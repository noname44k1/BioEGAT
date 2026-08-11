import os
import json
import numpy as np
from time import time
from tqdm import trange, tqdm
import argparse
from pathlib import Path

import bitsandbytes as bnb
import torch

import transformers
from transformers import AutoConfig,  GenerationConfig
from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from transformers import set_seed, Seq2SeqTrainer, BitsAndBytesConfig

from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, prepare_model_for_kbit_training, PeftModel

from arguments import Arguments, FinetuningArguments, GenerationArguments
from data import DataModule, QueryCollator
from model import BioEGAT
from model.rerank import build_rerank_enhancer

from torch.amp import autocast  # torch 2.x: torch.cuda.amp.autocast is deprecated

import torch
torch.cuda.empty_cache()


class Evaluator:
    def __init__(self, args, tokenizer, model, data_module, generation_config):
        self.args = args
        self.generation_config = generation_config

        self.tokenizer = tokenizer
        self.model = model
        self.data_module = data_module

        self.output_dir = os.path.dirname(args.checkpoint_dir)
        self.log_file_path = os.path.join(self.output_dir, 'metrics.txt')


    @torch.no_grad()
    def ranking_metrics(self, dataset):
        self.model.eval()

        preds = []
        ranks = np.array([])

        generated = []
        for ex_idx, ex in enumerate(tqdm(dataset)):
            prompt = ex['input']

            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = inputs.input_ids.cuda() 
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id 

            subgraph = [ex['subgraph']] if 'subgraph' in ex else None
            
            output = self.model.generate(
                input_ids=input_ids, 
                query_ids=torch.LongTensor([ex['query_entity_id']]).to(input_ids.device), 
                entity_ids=torch.LongTensor([ex['rank_entities_id']]).to(input_ids.device), 
                subgraph=subgraph, 
                generation_config=self.generation_config,
            )
            generated.append(output.sequences[0].cpu().numpy().tolist())
            ex.pop('input')
        
        batch_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        for ex_idx, ex in enumerate(dataset):
            target = ex.pop('output')
            rank = ex['rank']
            pred = str(batch_preds[ex_idx]).strip()

            topk_names = ex['rank_entities']
            if target == pred:
                rank = 1
            else:    
                if pred not in set(topk_names) or topk_names.index(pred) >= rank:
                    rank += 1
            
            ex['target'] = target
            ex['pred_rank'] = rank
            ex['pred'] = pred
            preds.append(ex)
            ranks = np.append(ranks, rank)
        
        metrics = {
        'mrr': np.mean(1. / ranks),
        'hits1': np.mean(ranks <= 1),
        'hits3': np.mean(ranks <= 3),
        'hits10': np.mean(ranks <= 10),
        }
        metrics = {k: round(v, 8) for k, v in metrics.items()}
        
        print("ranking metrics:")
        print(metrics)
        
        with open(self.log_file_path, 'w', encoding='utf-8') as log_file:
            log_line = f'ranking metrics: {metrics}\n'
            log_file.write(log_line)

        
        return preds


def resolve_checkpoint(ckpt_dir):
    """Return a checkpoint dir that actually contains a LoRA adapter + graph_model.bin.
    If the requested dir is missing/incomplete, fall back to the newest sibling checkpoint-*."""
    def valid(d):
        return (os.path.isfile(os.path.join(d, "adapter_config.json"))
                and os.path.isfile(os.path.join(d, "graph_model.bin")))
    if valid(ckpt_dir):
        return ckpt_dir
    parent = os.path.dirname(os.path.normpath(ckpt_dir)) or "."
    cands = []
    if os.path.isdir(parent):
        for name in sorted(os.listdir(parent)):
            d = os.path.join(parent, name)
            if name.startswith("checkpoint-") and valid(d):
                cands.append(d)
    def step_key(d):
        tail = os.path.basename(d).split("-")[-1]
        return int(tail) if tail.isdigit() else -1
    cands.sort(key=step_key)
    if cands:
        print(f"[infer] '{ckpt_dir}' has no adapter; falling back to newest valid checkpoint: '{cands[-1]}'")
        return cands[-1]
    contents = os.listdir(parent) if os.path.isdir(parent) else "<parent missing>"
    raise FileNotFoundError(
        f"No valid checkpoint (needs adapter_config.json + graph_model.bin) at '{ckpt_dir}' "
        f"nor under '{parent}'. Did training finish and save? Found: {contents}")


if __name__ == '__main__':
    set_seed(3407)

    hfparser = HfArgumentParser((Arguments, GenerationArguments))
    (data_args, generation_args, _) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(data_args))
    args.checkpoint_dir = resolve_checkpoint(args.checkpoint_dir)
    print(f"Using checkpoint: {args.checkpoint_dir}")

    print(f"Load LLM: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(['[QUERY]', '[ENTITY]', '[RELATION]'])
    

    generation_config.bos_token_id = tokenizer.bos_token_id
    
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, device_map='auto')
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)

    # Half-precision the LLM only; the BioEGAT backbone stays fp32 and its outputs are cast
    # to the LLM dtype at injection time (see BioEGAT._replace_placeholders).
    model = model.half()

    llm_config = model.config
    # Build the identical Stage-2 architecture (output projection -> LLM hidden size), then
    # load the trained Stage-2 weights from graph_model.bin (carries the trained adapter.fc2).
    embed_model, enh_info = build_rerank_enhancer(
        ent_path=args.kge_embedding_path,
        rel_path=args.rel_embedding_path,
        graph_weights_path=args.graph_weights_path,
        llm_hidden_size=llm_config.hidden_size,
        adapter_size=args.adapter_size,
        load_stage1=False,
    )
    ckpt_dir = Path(args.checkpoint_dir)
    state = torch.load(ckpt_dir / "graph_model.bin", map_location="cpu")
    embed_model.load_state_dict(state)

    model = BioEGAT(tokenizer, model, embed_model)

    model.cuda()
    model.eval()

    data_module = DataModule(args, tokenizer)

    evaluator = Evaluator(args, tokenizer, model, data_module, generation_config)

    with autocast('cuda'):
        preds = evaluator.ranking_metrics(data_module.test_ds)
    output = {
        'args': vars(args),
        'generation_config': vars(generation_config),
        'prediction': preds,
    }
    output_path = os.path.join(os.path.dirname(args.checkpoint_dir), f'prediction.json')
    json.dump(output, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)