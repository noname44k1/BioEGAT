"""
Step 1: Build Evidence Dataset.

Reads a DrKGC JSON dataset, evaluates the structural evidence for each candidate,
computes PSR and Fail-Safe statistics, translates paths to natural language,
and generates the final `llm_prompt`.
Saves the augmented dataset for the next step (LLM inference).
Filters the dataset to only include examples where the correct answer is within the top-K.
"""

import json
import argparse
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
import sys
from collections import defaultdict

# Add parent directory to sys.path to allow importing synthesizer
sys.path.append(str(Path(__file__).resolve().parent.parent))

from synthesizer import (
    PathUnroller,
    EvidenceBuilder,
    LLMReranker
)

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
    parser = argparse.ArgumentParser(description="Step 1: Build Evidence Dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file (e.g. test.json)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file with evidence and prompt")
    parser.add_argument("--id2relation", type=str, required=True, help="Path to id2relation.pkl")
    parser.add_argument("--top_k", type=int, default=10, help="Number of candidates to evaluate")
    parser.add_argument("--limit", type=int, default=-1, help="Max number of samples to process (for testing)")
    parser.add_argument("--no_filter", action="store_true", help="Do not filter out samples where ground truth is not in top_k")
    parser.add_argument("--entity_embeddings", type=str, default=None, help="Path to pre-trained KGE entity embeddings tensor (.pt)")
    parser.add_argument("--relation_embeddings", type=str, default=None, help="Path to pre-trained KGE relation embeddings tensor (.pt)")
    parser.add_argument("--kge_alpha", type=float, default=0.5, help="Scaling factor for negative exponential distance in KGE score")
    parser.add_argument("--supporting_threshold", type=float, default=0.5, help="Threshold to consider a path as supporting")
    parser.add_argument("--fallback_penalty", type=float, default=0.5, help="Penalty factor for disconnected 1-hop fallback edges")
    parser.add_argument("--p_value_threshold", type=float, default=1e-5, help="p-value significance threshold for fail-safe")
    parser.add_argument("--config", type=str, default=None, help="Path to dataset-specific hyperparameter config JSON")
    parser.add_argument("--model", type=str, default="TransE", help="KGE model type (e.g. TransE, DistMult)")
    
    args = parser.parse_args()

    if args.config:
        logger.info(f"Loading configurations from config file: {args.config}")
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        for key, val in config_data.items():
            if hasattr(args, key):
                setattr(args, key, val)
                logger.info(f"Overrode argument {key} with config value: {val}")
    
    logger.info(f"Loading id2relation from {args.id2relation}")
    with open(args.id2relation, 'rb') as f:
        id2relation = pickle.load(f)
        
    # Attempt to load id2entity.pkl from same directory
    id2entity_path = Path(args.id2relation).parent / "id2entity.pkl"
    id2entity = None
    if id2entity_path.exists():
        logger.info(f"Loading id2entity from {id2entity_path}")
        with open(id2entity_path, 'rb') as f:
            id2entity = pickle.load(f)
            
    # Build a lookup from biological identifiers (e.g. "MESH:D010146") to common names (e.g. "Pain")
    id_to_common_name = {}
    dmdb_paths_file = Path("DrugMechDB/DrugMechDB-2.0.1/SuLab-DrugMechDB-caa8f78/indication_paths.json")
    if dmdb_paths_file.exists():
        logger.info(f"Building ID-to-Name mapping from {dmdb_paths_file}...")
        try:
            with open(dmdb_paths_file, 'r', encoding='utf-8') as f:
                dmdb_data = json.load(f)
            
            if isinstance(dmdb_data, list):
                graphs = dmdb_data
            elif isinstance(dmdb_data, dict) and "graphs" in dmdb_data:
                graphs = dmdb_data["graphs"]
            else:
                graphs = []
                
            for graph in graphs:
                for node in graph.get("nodes", []):
                    node_id = node.get("id")
                    node_name = node.get("name")
                    if node_id and node_name:
                        id_to_common_name[node_id] = node_name
            logger.info(f"Successfully built mapping for {len(id_to_common_name)} DrugMechDB entities.")
        except Exception as e:
            logger.error(f"Error parsing indication_paths.json: {e}")

    # Map id2entity identifiers to their original common English names
    if id2entity and id_to_common_name:
        logger.info("Mapping entity database identifiers to their original common English names...")
        mapped_count = 0
        for ent_id, db_identifier in id2entity.items():
            if db_identifier in id_to_common_name:
                id2entity[ent_id] = id_to_common_name[db_identifier]
                mapped_count += 1
        logger.info(f"Mapped {mapped_count} entities in id2entity database.")
            
    # Attempt to load rules.json from the input dataset's directory
    input_dir = Path(args.input).parent
    rules_path = input_dir / "rules.json"
    ncrl_rules = {}
    if rules_path.exists():
        logger.info(f"Loading NCRL rules from {rules_path}")
        with open(rules_path, 'r', encoding='utf-8') as f:
            ncrl_rules = json.load(f)
            
    # Attempt to load train.txt to calculate training node degrees
    train_path = input_dir / "train.txt"
    if not train_path.exists():
        # Fallback search paths for train.txt
        possible_paths = [
            input_dir.parent / "drugmechdb" / "train.txt",
            input_dir.parent / "train.txt",
            Path("dataset/drugmechdb/train.txt")
        ]
        for p in possible_paths:
            if p.exists():
                train_path = p
                break

    train_stats = {"global_node_degrees": {}}
    if train_path.exists() and id2entity_path.exists():
        logger.info(f"Computing global training node degrees from {train_path}...")
        
        # Load entity2id.pkl to map train.txt string names to IDs
        entity2id_path = Path(args.id2relation).parent / "entity2id.pkl"
        if entity2id_path.exists():
            with open(entity2id_path, 'rb') as f:
                entity2id = pickle.load(f)
            
            degrees = defaultdict(int)
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                        h_id = entity2id.get(h)
                        t_id = entity2id.get(t)
                        if h_id is not None:
                            degrees[h_id] += 1
                        if t_id is not None:
                            degrees[t_id] += 1
            train_stats["global_node_degrees"] = dict(degrees)
            logger.info(f"Successfully indexed degrees for {len(degrees)} training entities.")
            
    # Load KGE embeddings if paths are provided
    ent_embeddings = None
    rel_embeddings = None
    if args.entity_embeddings:
        logger.info(f"Loading entity embeddings from {args.entity_embeddings}")
        import torch
        loaded = torch.load(args.entity_embeddings, map_location='cpu')
        if isinstance(loaded, dict):
            if 'ent_emb.weight' in loaded:
                ent_embeddings = loaded['ent_emb.weight']
                logger.info("Extracted ent_emb.weight from unified checkpoint dictionary")
            if 'rel_emb.weight' in loaded and rel_embeddings is None:
                rel_embeddings = loaded['rel_emb.weight']
                logger.info("Extracted rel_emb.weight from unified checkpoint dictionary")
            
            # support other naming variants
            if 'entity_embeddings.weight' in loaded and ent_embeddings is None:
                ent_embeddings = loaded['entity_embeddings.weight']
                logger.info("Extracted entity_embeddings.weight from unified checkpoint dictionary")
            if 'relation_embeddings.weight' in loaded and rel_embeddings is None:
                rel_embeddings = loaded['relation_embeddings.weight']
                logger.info("Extracted relation_embeddings.weight from unified checkpoint dictionary")
        else:
            ent_embeddings = loaded
            
    if args.relation_embeddings:
        logger.info(f"Loading relation embeddings from {args.relation_embeddings}")
        import torch
        loaded = torch.load(args.relation_embeddings, map_location='cpu')
        if isinstance(loaded, dict):
            if 'rel_emb.weight' in loaded:
                rel_embeddings = loaded['rel_emb.weight']
                logger.info("Extracted rel_emb.weight from model checkpoint dictionary")
            elif 'relation_embeddings.weight' in loaded:
                rel_embeddings = loaded['relation_embeddings.weight']
                logger.info("Extracted relation_embeddings.weight from model checkpoint dictionary")
        else:
            rel_embeddings = loaded

    logger.info("Initializing components...")
    path_unroller = PathUnroller(id2relation=id2relation, id2entity=id2entity, reverse_for_head=True)
    evidence_builder = EvidenceBuilder(
        path_unroller=path_unroller,
        train_stats=train_stats,
        ncrl_rules=ncrl_rules,
        ent_embeddings=ent_embeddings,
        rel_embeddings=rel_embeddings,
        kge_alpha=args.kge_alpha,
        supporting_threshold=args.supporting_threshold,
        fallback_penalty=args.fallback_penalty,
        p_value_threshold=args.p_value_threshold,
        model_name=args.model
    )
    
    # We initialize LLMReranker just to use its prompt building capabilities (no model loaded)
    reranker = LLMReranker()
    
    logger.info(f"Reading data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Filter dataset: Keep only examples where the true entity rank is within top_k
    if not args.no_filter:
        filtered_data = [item for item in data if item.get("rank", 9999) <= args.top_k]
        logger.info(f"Filtered dataset from {len(data)} to {len(filtered_data)} examples where ground truth is in top {args.top_k}.")
        data = filtered_data
    else:
        logger.info(f"Using full dataset with {len(data)} examples (--no_filter is enabled).")
        
    if args.limit > 0:
        data = data[:args.limit]
        logger.info(f"Limiting processing to {args.limit} samples")
        
    logger.info("Starting evidence generation process...")
    for item in tqdm(data, desc="Building evidence"):
        # Map query entity, candidates, and output in the dataset item to common names if available
        if id_to_common_name:
            if item.get("query_entity") in id_to_common_name:
                item["query_entity"] = id_to_common_name[item["query_entity"]]
            
            mapped_rank_entities = []
            for name in item.get("rank_entities", []):
                mapped_rank_entities.append(id_to_common_name.get(name, name))
            item["rank_entities"] = mapped_rank_entities
            
            if item.get("output") in id_to_common_name:
                item["output"] = id_to_common_name[item["output"]]
                
            triple = item.get("triple", ["", "", ""])
            if len(triple) >= 3:
                triple[0] = id_to_common_name.get(triple[0], triple[0])
                triple[2] = id_to_common_name.get(triple[2], triple[2])
                item["triple"] = triple

        query_entity = item.get("query_entity", "")
        relation_name = item.get("triple", ["", "", ""])[1]
        
        # Build evidence profiles for the top-k candidates
        profiles = evidence_builder.build_profiles(item, top_k=args.top_k)
        
        # Inject GNN Rerank Certainty scores if present in the data item
        rerank_scores = item.get("rerank_scores", [])
        for idx, profile in enumerate(profiles):
            if idx < len(rerank_scores):
                profile.rerank_certainty = rerank_scores[idx]
                
        # Save structured evidence to the item
        item["evidence_profiles"] = [p.to_dict() for p in profiles]
        
        # Build the final LLM prompt and store it in the item
        llm_prompt = reranker.build_prompt(
            query_entity=query_entity,
            relation=relation_name,
            evidence_profiles=profiles
        )
        item["llm_prompt"] = llm_prompt
        
        # Map the local subgraph triples from integer IDs to text names
        resolved_subgraph = []
        for h, r, t in item.get("subgraph", []):
            h_name = id2entity.get(h, str(h)) if id2entity else str(h)
            r_name = id2relation.get(r, str(r))
            t_name = id2entity.get(t, str(t)) if id2entity else str(t)
            resolved_subgraph.append([h_name, r_name, t_name])
        item["subgraph"] = resolved_subgraph
        
    logger.info(f"Saving augmented dataset to {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=default_json_serializer)
        
    logger.info("Done! Dataset is ready for LLM inference.")


if __name__ == "__main__":
    main()
