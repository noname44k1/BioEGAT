#!/usr/bin/env bash
# BioEGAT Stage-2 (LLM) — Option A: reuse the frozen Stage-1 KBGAT+InteractE backbone,
# train only the InteractE projection (adapter.fc2 -> LLM hidden size) + LoRA.
#
# Usage:  ./run_bioegat.sh <dataset> [model_name]
#   dataset    : drugmechdb | primekg | hetionet  (or the name variants drugmechdb_name | hetionet_name)
#   model_name : HF id (default meta-llama/Meta-Llama-3-8B, hidden_size 4096)
#
# The *_name datasets hold human-name prompts but share the SAME integer-id embeddings as the
# base dataset (entity ids are identical), so embeddings/graph weights are read from the base
# rerank/ dir (no duplication).
set -euo pipefail
cd "$(dirname "$0")"

DS="${1:-drugmechdb}"
MODEL="${2:-meta-llama/Meta-Llama-3-8B}"
BASE_DS="${DS%_name}"          # strip the _name suffix to locate the shared embeddings
MODEL_TAG="$(basename "$MODEL" | tr '/:' '__')"   # e.g. Meta-Llama-3-8B / Llama3-OpenBioLLM-8B
DATA="data/${DS}"
RERANK="data/${BASE_DS}/rerank"
OUT="results/bioegat_${DS}_${MODEL_TAG}"          # model-aware so backbones don't overwrite each other

echo "=== BioEGAT Stage-2 | dataset=${DS} | model=${MODEL} | out=${OUT} ==="

# -------- Train --------
python main.py \
  --dataset_path "${DATA}" \
  --model_name_or_path "${MODEL}" \
  --model_type llama \
  --kge_embedding_path "${RERANK}/finetuned_ent_embeddings_${BASE_DS}.pt" \
  --rel_embedding_path "${RERANK}/finetuned_rel_embeddings_${BASE_DS}.pt" \
  --graph_weights_path "${RERANK}/graph_only_best_${BASE_DS}.pt" \
  --adapter_size 512 \
  --output_dir "${OUT}" \
  --use_quant True --bits 4 --double_quant True --quant_type nf4 \
  --bf16 True \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --source_max_len 2048 --target_max_len 64 \
  --logging_steps 10 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --early_stopping_patience 3 \
  --remove_unused_columns False

# -------- Infer / evaluate --------
python infer.py \
  --dataset_path "${DATA}" \
  --model_name_or_path "${MODEL}" \
  --model_type llama \
  --kge_embedding_path "${RERANK}/finetuned_ent_embeddings_${BASE_DS}.pt" \
  --rel_embedding_path "${RERANK}/finetuned_rel_embeddings_${BASE_DS}.pt" \
  --graph_weights_path "${RERANK}/graph_only_best_${BASE_DS}.pt" \
  --adapter_size 512 \
  --checkpoint_dir "${OUT}/checkpoint-final" \
  --source_max_len 2048 --target_max_len 64

echo "=== Done. Metrics: ${OUT}/metrics.txt | predictions: ${OUT}/prediction.json ==="
