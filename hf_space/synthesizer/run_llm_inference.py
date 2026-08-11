"""
Step 2: Run LLM Inference.

Reads the augmented JSON dataset (produced by Step 1), extracts the `llm_prompt`,
runs inference using the Llama 3 model, and saves the final result.
"""

import json
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import sys

# Add parent directory to sys.path to allow importing synthesizer
sys.path.append(str(Path(__file__).resolve().parent.parent))

from synthesizer import LLMReranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def default_json_serializer(o):
    import numpy as np
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    if isinstance(o, np.float32) or isinstance(o, np.float64):
        return float(o)
    raise TypeError(f"Type {o.__class__.__name__} not serializable")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Run LLM Inference")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file from Step 1 (e.g. test_with_prompts.json)")
    parser.add_argument("--output", type=str, required=True, help="Final Output JSON file with LLM reasoning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="LLM for reranking")
    parser.add_argument("--limit", type=int, default=-1, help="Max number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    logger.info(f"Initializing LLM Reranker with model: {args.model_name}")
    reranker = LLMReranker(model_name=args.model_name)
    
    # Pre-load model to avoid lazy loading per item
    reranker._load_model()
    
    logger.info(f"Reading data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if args.limit > 0:
        data = data[:args.limit]
        logger.info(f"Limiting processing to {args.limit} samples")
        
    logger.info("Starting LLM inference process...")
    for item in tqdm(data, desc="Running Inference"):
        if "llm_prompt" not in item:
            logger.warning("Missing 'llm_prompt' field in item. Skipping inference.")
            continue
            
        prebuilt_prompt = item["llm_prompt"]
        query_entity = item.get("query_entity", "")
        relation_name = item.get("triple", ["", "", ""])[1]
        
        # Run LLM Reranking reasoning using prebuilt prompt
        llm_reasoning = reranker.rerank(
            query_entity=query_entity,
            relation=relation_name,
            prebuilt_prompt=prebuilt_prompt
        )
        
        item["llm_reasoning"] = llm_reasoning
        
    logger.info(f"Saving final dataset to {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=default_json_serializer)
        
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
