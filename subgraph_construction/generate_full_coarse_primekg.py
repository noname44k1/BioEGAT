import torch
import json
import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_true_dict(all_triples):
    """ Generates dictionaries of true head and tail completions for filtered ranking. """
    heads = {}
    tails = {}
    for s, r, o in all_triples:
        if (r, o) not in heads: heads[(r, o)] = []
        if (s, r) not in tails: tails[(s, r)] = []
        heads[(r, o)].append(s)
        tails[(s, r)].append(o)
    return heads, tails

def generate_full_coarse_json_primekg(csv_path, emb_path, entity2id_path, id2entity_path, id2relation_path, output_json, batch_size=512):
    # 1. Load Mappings
    print(f"Loading retriever mappings (System B) from {os.path.dirname(entity2id_path)}...")
    with open(entity2id_path, 'rb') as f:
        n2i_b = pickle.load(f) # Maps (str) Name -> retriever_index
    with open(id2relation_path, 'rb') as f:
        id2relation = pickle.load(f)
    relation2id_b = {v: k for k, v in id2relation.items()}
    
    # Load Training Loader Mappings (System A)
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
    from primekg_loader import load_primekg_data
    base_dir = os.path.dirname(csv_path)
    (n2i_a, _), (r2i_a, _), train_triples, test_triples, all_triples_a, _ = load_primekg_data(base_dir, use_test_set=True)
    # n2i_a maps (str) Name -> training_index
    
    # 2. Load Embeddings
    print(f"Loading entity embeddings from {emb_path} to {DEVICE}...")
    embeddings = torch.load(emb_path, map_location=DEVICE)
    if isinstance(embeddings, dict) and 'weight' in embeddings:
        embeddings = embeddings['weight']
        
    rel_emb_path = os.path.join(os.path.dirname(emb_path), "relation_embeddings.pt")
    print(f"Loading relation embeddings from {rel_emb_path} to {DEVICE}...")
    rel_embeddings = torch.load(rel_emb_path, map_location=DEVICE)

    # 3. Prepare Metadata
    print("Building index to metadata mapping...")
    idx2name = {}
    idx2type = {}
    for split_csv in ["train_final.csv", "valid.csv", "test.csv"]:
        split_path = os.path.join(base_dir, split_csv)
        if os.path.exists(split_path):
            chunk_df = pd.read_csv(split_path, usecols=['x_index', 'x_name', 'x_type', 'y_index', 'y_name', 'y_type'])
            for _, row in chunk_df.iterrows():
                idx2name[int(row['x_index'])] = str(row['x_name'])
                idx2type[int(row['x_index'])] = str(row['x_type'])
                idx2name[int(row['y_index'])] = str(row['y_name'])
                idx2type[int(row['y_index'])] = str(row['y_type'])
    
    print("Generating true triple dictionary for filtered ranking...")
    heads_true_a, tails_true_a = generate_true_dict(all_triples_a)

    type_to_indices_a = {}
    for idx_orig, t in idx2type.items():
        if t not in type_to_indices_a: type_to_indices_a[t] = []
        name = idx2name.get(idx_orig)
        if name in n2i_a:
            type_to_indices_a[t].append(n2i_a[name])
    
    num_nodes_a = embeddings.shape[0]
    type_masks_a = {}
    for t, indices_a in type_to_indices_a.items():
        mask = torch.zeros(num_nodes_a, dtype=torch.bool).to(DEVICE)
        if indices_a:
            mask[torch.LongTensor(indices_a)] = True
            type_masks_a[t] = mask

    # 4. Build Tasks with dual ID mapping
    print(f"Reading current split: {csv_path}...")
    df = pd.read_csv(csv_path)
    all_tasks = []
    for _, row in df.iterrows():
        h_name, r_name, t_name = str(row['x_name']), str(row['relation']), str(row['y_name'])
        
        # Verify both mappings exist (using names)
        if h_name not in n2i_a or t_name not in n2i_a: continue
        if h_name not in n2i_b or t_name not in n2i_b: continue
        if r_name not in r2i_a: continue
        
        h_idx_a, t_idx_a, r_idx_a = n2i_a[h_name], n2i_a[t_name], r2i_a[r_name]
        h_idx_b, t_idx_b = n2i_b[h_name], n2i_b[t_name]
        r_idx_b = relation2id_b.get(r_name, -1)
        
        # (h, r, ?) -> predicted_tail
        all_tasks.append({
            "triple": [h_name, r_name, t_name],
            "triple_id_b": [h_idx_b, r_idx_b, t_idx_b],
            "type": "predicted_tail",
            "query_entity": h_name,
            "query_idx_a": h_idx_a,
            "query_idx_b": h_idx_b,
            "target_idx_a": t_idx_a,
            "target_type": str(row['y_type']),
            "rel_idx_a": r_idx_a
        })
        # (?, r, t) -> predicted_head
        all_tasks.append({
            "triple": [h_name, r_name, t_name],
            "triple_id_b": [h_idx_b, r_idx_b, t_idx_b],
            "type": "predicted_head",
            "query_entity": t_name,
            "query_idx_a": t_idx_a,
            "query_idx_b": t_idx_b, # Corrected: t is the query entity
            "target_idx_a": h_idx_a,
            "target_type": str(row['x_type']),
            "rel_idx_a": r_idx_a
        })

    # 5. Execute Ranking with Filtered MRR calculation and Dual-ID Candidate lookup
    results = []
    num_tasks = len(all_tasks)
    # Bridge from model index A back to original ID, then to retriever index B
    a2name = {v: k for k, v in n2i_a.items()}
    a2b = {v: n2i_b[k] for k, v in n2i_a.items() if k in n2i_b}
    
    for start_idx in tqdm(range(0, num_tasks, batch_size), desc=f"Ranking {os.path.basename(csv_path)}"):
        end_idx = min(start_idx + batch_size, num_tasks)
        batch = all_tasks[start_idx:end_idx]
        
        batch_query_ids_a = torch.LongTensor([t['query_idx_a'] for t in batch]).to(DEVICE)
        batch_rel_ids_a = torch.LongTensor([t['rel_idx_a'] for t in batch]).to(DEVICE)
        
        head_emb = embeddings[batch_query_ids_a]
        rel_emb = rel_embeddings[batch_rel_ids_a]
        query_vecs = head_emb * rel_emb
        raw_scores = torch.mm(query_vecs, embeddings.t()) # Model scope A
        
        for i in range(len(batch)):
            task = batch[i]
            target_idx_a = task['target_idx_a']
            query_idx_a = task['query_idx_a']
            rel_idx_a = task['rel_idx_a']
            
            # --- Filtered Ranking Logic ---
            scores = raw_scores[i].clone()
            
            # Identify other ground truth entities to mask (filtering)
            if task['type'] == "predicted_tail":
                others = tails_true_a.get((query_idx_a, rel_idx_a), [])
            else:
                others = heads_true_a.get((rel_idx_a, query_idx_a), [])
            
            # Mask competing correct entities (set to -inf)
            mask_indices = [idx for idx in others if idx != target_idx_a]
            if mask_indices:
                scores[torch.LongTensor(mask_indices).to(DEVICE)] = float('-inf')
            
            # Calculate Filtered Rank with tie-breaking (Standard KGE protocol)
            target_score = scores[target_idx_a].item()
            num_greater = (scores > target_score).sum().item()
            num_equal = (scores == target_score).sum().item()
            
            # Mean Rank formula: R = num_greater + 1 + (num_equal - 1) / 2
            rank = num_greater + 1 + (num_equal - 1) / 2.0
            
            # --- Top-20 Candidates (Selected from filtered scores) ---
            _, top_indices_a = torch.topk(scores, k=20)
            top_indices_a = top_indices_a.tolist()
            
            # Map candidates to names and System B IDs
            rank_entities = [a2name.get(idx, str(idx)) for idx in top_indices_a]
            rank_entities_id_b = [a2b.get(idx, -1) for idx in top_indices_a]
            
            results.append({
                "triple": task['triple'],
                "triple_id": task['triple_id_b'], # Store System B IDs
                "type": task['type'],
                "query_entity": task['query_entity'],
                "query_entity_id": task['query_idx_b'], # Use System B ID
                "rank_entities": rank_entities,
                "rank_entities_id": rank_entities_id_b, # Use System B IDs
                "rank": float(rank) # Save as float to preserve mean rank
            })
            
    print(f"Saving {len(results)} entries to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, help='Input CSV file path')
    parser.add_argument('--output_json', type=str, help='Output JSON file path')
    parser.add_argument('--emb_path', type=str, help='Path to entity embeddings .pt file')
    parser.add_argument('--base_dir', type=str, default='dataset/primekg/', help='Base directory for data')
    parser.add_argument('--ncrl_dir', type=str, default='dataset/primekg/primekg_ncrl', help='NCRL directory for mappings')
    args = parser.parse_args()

    if args.input_csv and args.output_json and args.emb_path:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        generate_full_coarse_json_primekg(
            args.input_csv,
            args.emb_path,
            os.path.join(args.ncrl_dir, "entity2id.pkl"),
            os.path.join(args.ncrl_dir, "id2entity.pkl"),
            os.path.join(args.ncrl_dir, "id2relation.pkl"),
            args.output_json
        )
    else:
        # Default behavior for existing splits
        output_dir = "dataset/primekg_rgcn_256"
        os.makedirs(output_dir, exist_ok=True)
        splits = [
            ("test.csv", "test_coarse.json"),
            ("valid.csv", "valid_coarse.json")
        ]
        for csv_in, json_out in splits:
            generate_full_coarse_json_primekg(
                os.path.join(args.base_dir, csv_in),
                os.path.join(output_dir, "entity_embeddings.pt"),
                os.path.join(args.ncrl_dir, "entity2id.pkl"),
                os.path.join(args.ncrl_dir, "id2entity.pkl"),
                os.path.join(args.ncrl_dir, "id2relation.pkl"),
                os.path.join(output_dir, json_out)
            )
