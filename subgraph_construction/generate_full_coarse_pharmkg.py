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

def load_pharmkg_mappings(data_dir):
    print(f"Loading PharmKG System A mappings (Training) from {data_dir}...")
    with open(os.path.join(data_dir, 'entities.txt'), 'r') as f:
        ent_names = [line.strip() for line in f if line.strip()]
    n2i_a = {name: i for i, name in enumerate(ent_names)}
    
    with open(os.path.join(data_dir, 'relations.txt'), 'r') as f:
        rel_names = [line.strip() for line in f if line.strip()]
    r2i_a = {name: i for i, name in enumerate(rel_names)}
    
    # Load all triples for filtering
    all_triples_a = []
    for filename in ['train.txt', 'valid.txt']:
        p = os.path.join(data_dir, filename)
        if os.path.exists(p):
            print(f"Loading {filename} for filtering...")
            df = pd.read_csv(p, sep='\t', header=None, names=['sub', 'rel', 'obj'])
            for _, row in df.iterrows():
                h_name, r_name, t_name = str(row['sub']), str(row['rel']), str(row['obj'])
                if h_name in n2i_a and r_name in r2i_a and t_name in n2i_a:
                    all_triples_a.append((n2i_a[h_name], r2i_a[r_name], n2i_a[t_name]))
    
    return n2i_a, r2i_a, all_triples_a

def generate_full_coarse_json_pharmkg(input_path, emb_path, entity2id_path, id2entity_path, id2relation_path, output_json, batch_size=512):
    # 1. Load Mappings
    print(f"Loading retriever mappings (System B) from {os.path.dirname(entity2id_path)}...")
    with open(entity2id_path, 'rb') as f:
        n2i_b = pickle.load(f) # Maps (str) Name -> retriever_index
    with open(id2relation_path, 'rb') as f:
        id2relation = pickle.load(f)
    relation2id_b = {v: k for k, v in id2relation.items()}
    
    # Load Training Loader Mappings (System A)
    base_dir = os.path.dirname(entity2id_path) # Assume ncrl_dir contains entities.txt
    n2i_a, r2i_a, all_triples_a = load_pharmkg_mappings(base_dir)
    
    # 2. Load Embeddings
    print(f"Loading entity embeddings from {emb_path} to {DEVICE}...")
    embeddings = torch.load(emb_path, map_location=DEVICE)
    if isinstance(embeddings, dict) and 'weight' in embeddings:
        embeddings = embeddings['weight']
        
    # Attempt to find relation embeddings
    rel_emb_path = emb_path.replace("embeddings.pt", "rel_embeddings.pt")
    if not os.path.exists(rel_emb_path):
        rel_emb_path = os.path.join(os.path.dirname(emb_path), "hrgat_rel_embeddings.pt")
    
    print(f"Loading relation embeddings from {rel_emb_path} to {DEVICE}...")
    rel_embeddings = torch.load(rel_emb_path, map_location=DEVICE)

    # 3. Prepare Metadata for filtering
    print("Generating true triple dictionary for filtered ranking...")
    heads_true_a, tails_true_a = generate_true_dict(all_triples_a)

    # 4. Build Tasks with dual ID mapping
    print(f"Reading current split: {input_path}...")
    # Support both .txt (tab) and .csv (comma)
    sep = '\t' if input_path.endswith('.txt') or input_path.endswith('.tsv') else ','
    df = pd.read_csv(input_path, sep=sep, header=None, names=['sub', 'rel', 'obj'])
    
    all_tasks = []
    for _, row in df.iterrows():
        h_name, r_name, t_name = str(row['sub']), str(row['rel']), str(row['obj'])
        
        # Verify both mappings exist
        if h_name not in n2i_a or t_name not in n2i_a or r_name not in r2i_a: continue
        if h_name not in n2i_b or t_name not in n2i_b: continue
        
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
            "rel_idx_a": r_idx_a
        })
        # (?, r, t) -> predicted_head
        all_tasks.append({
            "triple": [h_name, r_name, t_name],
            "triple_id_b": [h_idx_b, r_idx_b, t_idx_b],
            "type": "predicted_head",
            "query_entity": t_name,
            "query_idx_a": t_idx_a,
            "query_idx_b": t_idx_b,
            "target_idx_a": h_idx_a,
            "rel_idx_a": r_idx_a
        })

    # 5. Execute Ranking
    results = []
    num_tasks = len(all_tasks)
    a2name = {v: k for k, v in n2i_a.items()}
    a2b = {v: n2i_b[k] for k, v in n2i_a.items() if k in n2i_b}
    
    for start_idx in tqdm(range(0, num_tasks, batch_size), desc=f"Ranking {os.path.basename(input_path)}"):
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
            
            scores = raw_scores[i].clone()
            
            if task['type'] == "predicted_tail":
                others = tails_true_a.get((query_idx_a, rel_idx_a), [])
            else:
                others = heads_true_a.get((rel_idx_a, query_idx_a), [])
            
            mask_indices = [idx for idx in others if idx != target_idx_a]
            if mask_indices:
                scores[torch.LongTensor(mask_indices).to(DEVICE)] = float('-inf')
            
            target_score = scores[target_idx_a].item()
            num_greater = (scores > target_score).sum().item()
            num_equal = (scores == target_score).sum().item()
            
            rank = num_greater + 1 + (num_equal - 1) / 2.0
            
            _, top_indices_a = torch.topk(scores, k=20)
            top_indices_a = top_indices_a.tolist()
            
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
                "rank": float(rank)
            })
            
    print(f"Saving {len(results)} entries to {output_json}...")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, help='Input file path (.txt or .csv)')
    parser.add_argument('--output_json', type=str, help='Output JSON file path')
    parser.add_argument('--emb_path', type=str, help='Path to entity embeddings .pt file')
    parser.add_argument('--ncrl_dir', type=str, default='dataset/PharmKG-8k/pharmkg8k_ncrl', help='NCRL directory for mappings')
    args = parser.parse_args()

    if args.input_csv and args.output_json and args.emb_path:
        generate_full_coarse_json_pharmkg(
            args.input_csv,
            args.emb_path,
            os.path.join(args.ncrl_dir, "entity2id.pkl"),
            os.path.join(args.ncrl_dir, "id2entity.pkl"),
            os.path.join(args.ncrl_dir, "id2relation.pkl"),
            args.output_json
        )
    else:
        # Default behavior for PharmKG
        base_dir = "dataset/PharmKG-8k/"
        ncrl_dir = os.path.join(base_dir, "pharmkg8k_ncrl")
        output_dir = os.path.join(base_dir, "pharmkg_final")
        
        splits = [
            ("valid.txt", "valid_coarse.json"),
            ("test.txt", "test_coarse.json")
        ]
        for inp, out in splits:
            generate_full_coarse_json_pharmkg(
                os.path.join(ncrl_dir, inp),
                os.path.join(base_dir, "hrgat_embeddings.pt"),
                os.path.join(ncrl_dir, "entity2id.pkl"),
                os.path.join(ncrl_dir, "id2entity.pkl"),
                os.path.join(ncrl_dir, "id2relation.pkl"),
                os.path.join(output_dir, out)
            )
