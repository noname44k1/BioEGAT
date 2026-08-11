from typing import Optional
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments


@dataclass
class Arguments:
    dataset_path: str = field(default=None, metadata={"help": "Path for dataset"})
    
    model_name_or_path: str = field(default="large language model name", metadata={"help": "Large language model name for huggingface download"})
    
    model_type: str = field(default="llama", metadata={"help": "The type of LLM, llama or mistral"})
    
    kge_embedding_path: str = field(default=None, metadata={ "help": "Path of fine-tuned ENTITY embeddings (Stage-1 BioEGAT output, frozen)"})

    rel_embedding_path: str = field(default=None, metadata={"help": "Path of fine-tuned RELATION embeddings (Stage-1 BioEGAT output, frozen) — needed to reconstruct the GAT relation dim"})

    graph_weights_path: str = field(default=None, metadata={"help": "Path of the Stage-1 reranker checkpoint (graph_only_best_{ds}.pt); its KBGAT+InteractE backbone is loaded and frozen"})

    adapter_size: int = field(default=512, metadata={"help": "InteractE adapter hidden size — MUST match the Stage-1 reranker (512) so fc1/conv weights load"})

    source_max_len: int = field(default=2048, metadata={"help": "Maximum source sequence length."},)
    target_max_len: int = field(default=64, metadata={"help": "Maximum target sequence length."},)
    
    checkpoint_dir: str = field(default=None, metadata={"help": "Checkpoint saveing directory"})


@dataclass
class FinetuningArguments(Seq2SeqTrainingArguments):
    use_quant: bool = field(default=False)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=4, metadata={"help": "How many bits to use."})

    output_dir: str = field(default='', metadata={"help": 'Directory where checkpoints are saved'})

    num_train_epochs: float = field(default=15.0)
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=32)

    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'Optimizer'})
    learning_rate: float = field(default=0.0002)
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Constant | Linear | Cosine'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Proportion of training to be dedicated to a linear warmup where learning rate gradually increases'})
    
    lora_r: int = field(default=32)
    lora_alpha: float = field(default=32)
    lora_dropout: float = field(default=0.1)
    remove_unused_columns: bool = field(default=False)

    early_stopping_patience: int = field(default=0, metadata={"help": "Stop if eval metric does not improve for N evals (0 = disabled). Needs eval_strategy + metric_for_best_model + load_best_model_at_end."})


@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field(default=64)
    min_new_tokens : Optional[int] = field(default=1)

    do_sample: Optional[bool] = field(default=False) 
    num_beams: Optional[int] = field(default=1) 
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    # NOTE: `use_cache` removed — newer transformers auto-adds a `--use-cache` kebab alias that
    # collides with the same option elsewhere in the parsed dataclasses. GenerationConfig defaults
    # use_cache=True, so generation behaviour is unchanged.

    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

    num_return_sequences: Optional[int] = field(default=1) 
    output_scores: Optional[bool] = field(default=False)
    return_dict_in_generate: Optional[bool] = field(default=True)

