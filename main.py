import os
import argparse

import bitsandbytes as bnb
import torch

from transformers import AutoConfig,  GenerationConfig
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from transformers import set_seed, Seq2SeqTrainer, BitsAndBytesConfig, EarlyStoppingCallback


from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, prepare_model_for_kbit_training

from arguments import Arguments, FinetuningArguments, GenerationArguments
from data import make_data_module
from model import BioEGAT
from model.rerank import build_rerank_enhancer

def get_accelerate_model(args, config, pretrained_model_class):
    device_map = 'auto' if os.environ.get('LOCAL_RANK') is None else {'': int(os.environ.get('LOCAL_RANK', '0'))}
    
   
    if args.use_quant:
        compute_dtype = torch.bfloat16 
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            device_map='auto',
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=torch.bfloat16,
        )
    else:
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path, 
            config=config,
            low_cpu_mem_usage=True, 
            device_map=device_map, 
        )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_quant)
    
    if args.model_type == "llama":
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif args.model_type == "mistral":
        config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
            ],
)
        
    model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model

        

class BioEGATTrainer(Seq2SeqTrainer):
    """Trainer that saves/reloads the custom BioEGAT model (LoRA adapter + graph_model.bin) so
    that `save_total_limit`, `load_best_model_at_end`, and EarlyStoppingCallback all work."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # unwrap any accelerate/DDP wrapper so we reach BioEGAT.save_pretrained
        model = self.model
        if hasattr(self, "accelerator"):
            model = self.accelerator.unwrap_model(model)
        # BioEGAT.save_pretrained writes the LoRA adapter + graph_model.bin only (no full-model dump)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:
            super()._save(output_dir, state_dict)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_best_model(self):
        ckpt = self.state.best_model_checkpoint
        if ckpt is None:
            print("No best checkpoint recorded; keeping last weights.")
            return
        print(f"Loading best checkpoint from: {ckpt}")
        llm = self.model.llm_model
        if hasattr(llm, "load_adapter"):
            adapter = "default"
            if getattr(llm, "active_adapters", None):
                adapter = llm.active_adapters[0]
            llm.load_adapter(ckpt, adapter)
        graph_path = os.path.join(ckpt, "graph_model.bin")
        if os.path.exists(graph_path):
            self.model.graph_model.load_state_dict(torch.load(graph_path, map_location="cpu"))




def train():
    set_seed(3407)

    hfparser = HfArgumentParser((Arguments, FinetuningArguments, GenerationArguments))
    (data_args, training_args, generation_args, _) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(data_args), **vars(training_args))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Load LLM: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(['[QUERY]', '[ENTITY]', '[RELATION]'])

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    if args.model_type == "llama":
        model = get_accelerate_model(args, model_config, LlamaForCausalLM)
    elif args.model_type == "mistral":
        model = get_accelerate_model(args, model_config, AutoModelForCausalLM)
    model.config.use_cache = False

    llm_config = model.config
    # Option A — reuse Stage-1 BioEGAT: load frozen KBGAT+InteractE backbone + fine-tuned
    # entity/relation embeddings; only adapter.fc2 (projection -> LLM hidden size) is trainable.
    embed_model, enh_info = build_rerank_enhancer(
        ent_path=args.kge_embedding_path,
        rel_path=args.rel_embedding_path,
        graph_weights_path=args.graph_weights_path,
        llm_hidden_size=llm_config.hidden_size,
        adapter_size=args.adapter_size,
        load_stage1=True,
    )
    print(f"[BioEGAT] emb_dim={enh_info['emb_dim']} num_relations={enh_info['num_relations']} "
          f"rel_dim={enh_info['rel_dim']} adapter_size={enh_info['adapter_size']} "
          f"-> projecting to LLM hidden_size={llm_config.hidden_size}")

    # Ensure gradients reach the trainable projection through LLM gradient checkpointing.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = BioEGAT(tokenizer, model, embed_model)

    data_module = make_data_module(args, tokenizer)
    
    callbacks = []
    patience = getattr(args, "early_stopping_patience", 0) or 0
    if patience > 0:
        print(f"Early stopping enabled (patience={patience}, metric={training_args.metric_for_best_model})")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = BioEGATTrainer(
        model=model,
        processing_class=tokenizer,   # transformers 5.x: `tokenizer=` was removed
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    # Training
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    final_dir = os.path.join(args.output_dir, "checkpoint-final")
    print(f"Saving final checkpoint to: {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

if __name__ == '__main__':
    train()

