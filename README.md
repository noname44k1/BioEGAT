# BioEGAT

A decoupled **GNN–LLM pipeline for biomedical knowledge-graph completion**. BioEGAT casts KGC as a cascade
of five modules: (1) KGE candidate retrieval, (2) dynamic subgraph retrieval, (3) an edge-featured GAT
reranker fused with an InteractE adapter, (4) LoRA LLM reranking with graph-embedding injection, and (5) an
explanatory layer for mechanistic reasoning. Modules 1–4 form the scored pipeline; Module 5 powers the public
demonstrator.

## Requirements
**Note:** install the appropriate PyTorch and DGL build with CUDA support.
```bash
# python 3.12
torch==2.3.1+cu118
dgl==2.4.0+cu118
ogb==1.3.6
networkx==3.4.2
transformers==4.38.2
peft==0.4.0
accelerate==0.27.2
bitsandbytes==0.40.2
safetensors==0.4.3
tokenizers==0.15.2
datasets==2.20.0
```

## 🚀 Quick Start

### 💾 Download the datasets
The four biomedical datasets — **DrugMechDB, PrimeKG, Hetionet, and PharmKG** — together with their
preprocessed artefacts (coarse top-20 candidate lists, retrieved subgraphs, and the fine-tuned Stage-1
entity/relation embeddings under each dataset's `rerank/` directory) are provided here:

📥 **[Download the 4 datasets](https://drive.google.com/drive/folders/1q4e256h6UL0j9kIGm1k39yOMgbZoXe-R?usp=sharing)**

**Original Dataset Sources:**
* **[DrugMechDB](https://github.com/SuLab/DrugMechDB)**
* **[PrimeKG](https://github.com/mims-harvard/PrimeKG)**
* **[Hetionet](https://github.com/hetio/hetionet)**
* **[PharmKG / PharmKG8k](https://zenodo.org/records/4525237)**

Place each dataset folder under [`data/`](data/) so the layout is:
```
data/
├── drugmechdb/   └── rerank/   # finetuned_*_drugmechdb.pt, graph_only_best_drugmechdb.pt
├── primekg/      └── rerank/
├── hetionet/     └── rerank/
├── pharmkg/      └── rerank/
├── dataset.py    # loaders (tracked in git)
├── collate.py
└── __init__.py
```
> The `*_name` variants (e.g. `hetionet_name`) hold human-name prompts but share the **same integer-id
> embeddings** as their base dataset, so they read embeddings/graph weights from the base `rerank/` directory.

The per-relation prediction lexicons for these datasets are already included under [`lexicon/`](lexicon/)
(`drugmechdb`, `primekg`, `hetionet`, `pharmkg8k`).

### ⚡ Run the full Stage-2 pipeline
[`run_bioegat.sh`](run_bioegat.sh) trains the LoRA LLM (reusing the frozen Stage-1 GAT+InteractE backbone) and
then runs inference + evaluation in one shot:
```bash
# ./run_bioegat.sh <dataset> [model_name]
#   dataset    : drugmechdb | primekg | hetionet  (or name variants: drugmechdb_name | hetionet_name)
#   model_name : HF id (default: meta-llama/Meta-Llama-3-8B, hidden_size 4096)
./run_bioegat.sh drugmechdb
```
Outputs land in `results/bioegat_<dataset>_<model>/` (`metrics.txt`, `prediction.json`).

### 🔧 Run the stages manually
**Train** (LoRA + the single InteractE projection layer; GAT backbone and KGE embeddings stay frozen):
```bash
python main.py \
  --dataset_path "data/drugmechdb" \
  --model_name_or_path "meta-llama/Meta-Llama-3-8B" --model_type llama \
  --kge_embedding_path "data/drugmechdb/rerank/finetuned_ent_embeddings_drugmechdb.pt" \
  --rel_embedding_path "data/drugmechdb/rerank/finetuned_rel_embeddings_drugmechdb.pt" \
  --graph_weights_path "data/drugmechdb/rerank/graph_only_best_drugmechdb.pt" \
  --adapter_size 512 --output_dir "results/bioegat_drugmechdb" \
  --use_quant True --bits 4 --double_quant True --quant_type nf4 --bf16 True \
  --num_train_epochs 5 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --source_max_len 2048 --target_max_len 64 --remove_unused_columns False
```

**Infer & evaluate:**
```bash
python infer.py \
  --dataset_path "data/drugmechdb" \
  --model_name_or_path "meta-llama/Meta-Llama-3-8B" --model_type llama \
  --kge_embedding_path "data/drugmechdb/rerank/finetuned_ent_embeddings_drugmechdb.pt" \
  --rel_embedding_path "data/drugmechdb/rerank/finetuned_rel_embeddings_drugmechdb.pt" \
  --graph_weights_path "data/drugmechdb/rerank/graph_only_best_drugmechdb.pt" \
  --adapter_size 512 --checkpoint_dir "results/bioegat_drugmechdb/checkpoint-final" \
  --source_max_len 2048 --target_max_len 64
```

## Repository layout
| Path | Description |
|------|-------------|
| `main.py` / `infer.py` | Stage-2 LoRA LLM training and inference/evaluation |
| `run_bioegat.sh` | End-to-end train → infer driver |
| `model/` | `rerank.py` (GAT+InteractE backbone), `bioegat.py` |
| `subgraph_construction/` | Module 2: Subgraph construction scripts |
| `data/` | Dataset loaders (the dataset blobs are git-ignored — download above) |
| `lexicon/` | Per-relation head/tail prediction lexicons for the 4 datasets |
| `hf_space/` | Public demonstrator (Module 5: explanatory context & mechanistic reasoning) |
| `*-hypo.py`, `run_bioegat-hypo.sh` | LLM-hypothesis-augmented variant (Module-1 candidate union) |

## Demostration
[Demo](https://github.com/user-attachments/assets/32e85e22-203f-437f-9903-3483bde621d7)

## Citation

This work builds on DrKGC:
```bibtex
@inproceedings{xiao-etal-2025-drkgc,
    title = "{D}r{KGC}: Dynamic Subgraph Retrieval-Augmented {LLM}s for Knowledge Graph Completion across General and Biomedical Domains",
    author = "Xiao, Yongkang and Zhang, Sinian and Dai, Yi and Zhou, Huixue and Hou, Jue and Ding, Jie and Zhang, Rui",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov, year = "2025", address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.892/",
    doi = "10.18653/v1/2025.findings-emnlp.892",
    pages = "16432--16445", ISBN = "979-8-89176-335-7",
}
```
