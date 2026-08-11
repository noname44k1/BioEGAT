import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import igraph as ig
import pickle
import argparse
import multiprocessing as mp
from functools import partial
import os
import gc

def add_prompt(raw, relation_questions_A_to_B, relation_questions_B_to_A, bkg):
    rel = raw['triple'][1]
    query_entity = raw['query_entity']
    rank_entities = raw['rank_entities']
    pred_type = raw['type']
    answer_options = "(" + ", ".join([f"'{name}'" for name in rank_entities]) + ")"
    refer_parts = []
    refer_parts.append(f"'{query_entity}': [QUERY]")
    for name in rank_entities:
        refer_parts.append(f"'{name}': [ENTITY]")
    if len(refer_parts) > 2:
        refer_str = ", ".join(refer_parts[:2]) + "," + ", ".join(refer_parts[2:])
    else:
        refer_str = ", ".join(refer_parts)
    if pred_type == "predicted_tail":
        question_template = relation_questions_A_to_B.get(rel, "What is related to {}?")
    elif pred_type == "predicted_head":
        question_template = relation_questions_B_to_A.get(rel, "What is related to {}?")
    question = question_template.format(query_entity)

    if bkg:
        prompt = ("You are a biomedical scientist. The task is to predict the answer based on the given question, and you only need to answer one entity. The answer must be in " + answer_options + ".\nYou can refer to the entity embeddings: " + refer_str + ".\n\nQuestion: " + question + "\nAnswer: ")
    else:
        prompt = ("You are an excellent linguist. The task is to predict the answer based on the given question, and you only need to answer one entity. The answer must be in " + answer_options + ".\nYou can refer to the entity embeddings: " + refer_str + ".\n\nQuestion: " + question + "\nAnswer: ")

    if pred_type == "predicted_tail":
        answer = raw['triple'][2]
    elif pred_type == "predicted_head":
        answer = raw['triple'][0]
    raw['input'] = prompt
    raw['output'] = answer


def map_graph(df, entity2id, relation2id):
    df_mapped = pd.DataFrame()
    if 'x_name' in df.columns and 'y_name' in df.columns:
        # PrimeKG CSV format
        df_mapped[0] = df['x_name'].map(entity2id)
        df_mapped[1] = df['relation'].map(relation2id)
        df_mapped[2] = df['y_name'].map(entity2id)
    else:
        # Fallback to index-based for headerless tab-separated files
        df_mapped[0] = df[0].map(entity2id)
        df_mapped[1] = df[1].map(relation2id)
        df_mapped[2] = df[2].map(entity2id)
    return df_mapped


# Globals for multiprocessing inheritance
G_GLOBAL = None
NAME_TO_IDX = None
NODE_NAMES = None
ADJ_GLOBAL = None
RULES_GLOBAL = None
GRAPH_SIZE_GLOBAL = None

def apply_rule_sequence(start_node_idx, rule_tuple, ADJ):
    temp_path = []
    current_node = start_node_idx
    for step_relation in rule_tuple:
        neighbors = ADJ[current_node].get(step_relation)
        if not neighbors:
            return False, []
        neighbor = neighbors[0]
        temp_path.append([current_node, step_relation, neighbor])
        current_node = neighbor
    return True, temp_path


def process_item_retrieval(item):
    global G_GLOBAL, NAME_TO_IDX, NODE_NAMES, ADJ_GLOBAL, RULES_GLOBAL, GRAPH_SIZE_GLOBAL
    rel_id = item['triple_id'][1]
    query_entity_id = item['query_entity_id']
    rank_entities_id = item['rank_entities_id']
    subg_indices = [] # Store as (u_idx, rel, v_idx)
    exp = 0

    source_idx = NAME_TO_IDX.get(query_entity_id)
    if source_idx is None:
        return [], 0
    
    target_indices = []
    for rid in rank_entities_id:
        t_idx = NAME_TO_IDX.get(int(rid))
        if t_idx is not None:
            target_indices.append(t_idx)
    target_indices_set = set(target_indices)
    
    # 1. Shortest paths
    if target_indices:
        try:
            paths = G_GLOBAL.get_shortest_paths(source_idx, to=target_indices, mode="all", output="vpath")
            for path in paths:
                if len(path) > 1:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        found_rel = None
                        for r, neighbors in ADJ_GLOBAL[u].items():
                            if v in neighbors:
                                found_rel = r
                                break
                        if found_rel is not None:
                            subg_indices.append((u, found_rel, v))
        except:
            exp += len(rank_entities_id)
    
    # 2. Rule-based paths
    if len(subg_indices) < GRAPH_SIZE_GLOBAL:
        rule_sequences = RULES_GLOBAL.get(rel_id, [])
        for rule_tuple in rule_sequences:
            if len(subg_indices) >= GRAPH_SIZE_GLOBAL: break
            success, temp_path = apply_rule_sequence(source_idx, rule_tuple, ADJ_GLOBAL)
            if success:
                reached_idx = temp_path[-1][2]
                if reached_idx in target_indices_set:
                    for triple in temp_path:
                        subg_indices.append(tuple(triple))
                        if len(subg_indices) >= GRAPH_SIZE_GLOBAL: break
    
    # 3. Size control
    if len(subg_indices) < GRAPH_SIZE_GLOBAL:
        allowed_indices = [source_idx] + target_indices
        for u in allowed_indices:
            for v in allowed_indices:
                if u == v: continue
                found_rel = None
                for r, neighbors in ADJ_GLOBAL[u].items():
                    if v in neighbors:
                        found_rel = r
                        break
                if found_rel is not None:
                    subg_indices.append((u, found_rel, v))
                    if len(subg_indices) >= GRAPH_SIZE_GLOBAL: break
            if len(subg_indices) >= GRAPH_SIZE_GLOBAL: break

    # Final cleanup and name mapping
    final_subg = []
    seen = set()
    for u, r, v in subg_indices:
        t_tuple = (u, r, v)
        if t_tuple not in seen:
            final_subg.append([NODE_NAMES[u], r, NODE_NAMES[v]])
            seen.add(t_tuple)
    
    return final_subg[:GRAPH_SIZE_GLOBAL], exp


def subgraph_func(dataset, graph_size, G, rules, num_processes=None):
    global G_GLOBAL, NAME_TO_IDX, NODE_NAMES, ADJ_GLOBAL, RULES_GLOBAL, GRAPH_SIZE_GLOBAL
    G_GLOBAL = G
    RULES_GLOBAL = rules
    GRAPH_SIZE_GLOBAL = graph_size
    
    if NAME_TO_IDX is None:
        print("Pre-computing vertex indexing and adjacency by relation...")
        NAME_TO_IDX = {v['name']: v.index for v in G.vs}
        NODE_NAMES = G.vs['name']
        
        ADJ_GLOBAL = [ {} for _ in range(G.vcount()) ]
        for edge in G.es:
            u, v = edge.source, edge.target
            r = edge['relation']
            if r not in ADJ_GLOBAL[u]: ADJ_GLOBAL[u][r] = []
            if r not in ADJ_GLOBAL[v]: ADJ_GLOBAL[v][r] = []
            ADJ_GLOBAL[u][r].append(v)
            ADJ_GLOBAL[v][r].append(u)
    
    if num_processes is None:
        num_processes = min(4, mp.cpu_count())
    
    print(f"Using {num_processes} processes for parallel subgraph retrieval...")
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_item_retrieval, dataset, chunksize=10), total=len(dataset)))
    
    all_subgraphs = [r[0] for r in results]
    del results
    gc.collect()
    return all_subgraphs


def default(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def main(args):
    print(f"Loading entity mappings...")
    with open(args.entity2id_path, 'rb') as file:
        entity2id = pickle.load(file)
    with open(args.id2entity_path, 'rb') as file:
        id2entity = pickle.load(file)
    with open(args.id2relation_path, 'rb') as file:
        id2relation = pickle.load(file)
    relation2id = {v: k for k, v in id2relation.items()}

    print(f"Loading lexicons and rules...")
    with open(args.tail_pred_lex, 'r', encoding='utf-8') as json_file:
        relation_questions_A_to_B = json.load(json_file)
    with open(args.head_pred_lex, 'r', encoding='utf-8') as json_file:
        relation_questions_B_to_A = json.load(json_file)
    with open(args.rules_path, 'r', encoding='utf-8') as json_file:
        rules_name = json.load(json_file)
    
    rules = {}
    for key, value in rules_name.items():
        rel_id = relation2id.get(key, 'Unknown')
        rule_list = []
        for rule in value:
            if isinstance(rule, dict) and 'body' in rule:
                body = rule['body']
            else:
                body = rule
            rule_list.append([relation2id.get(item, 'Unknown') for item in body])
        rules[rel_id] = rule_list

    def load_raw(path):
        if not path: return pd.DataFrame()
        try:
            # Try PrimeKG CSV first
            df = pd.read_csv(path)
            if 'x_name' in df.columns:
                return df
        except:
            pass
        # Fallback to tab-separated headerless
        return pd.read_csv(path, sep="\t", header=None)

    # Process splits
    if args.split_valid:
        print(f"--- repartitioning mode ---")
        with open(args.valid_json_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        import random
        random.seed(42)
        random.shuffle(full_data)

        split_idx = int(len(full_data) * args.train_ratio)
        train_data = full_data[:split_idx]
        valid_data = full_data[split_idx:]
        del full_data
        gc.collect()

        splits_to_process = [
            ('train', train_data, args.train_path_saved),
            ('valid', valid_data, args.valid_path_saved)
        ]
        
        # Test remains separate
        with open(args.test_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        splits_to_process.append(('test', test_data, args.test_path_saved))
    else:
        splits_to_process = []
        if args.valid_json_path:
            splits_to_process.append(('valid', args.valid_json_path, args.valid_path_saved))
        if args.test_json_path:
            splits_to_process.append(('test', args.test_json_path, args.test_path_saved))
        if args.train_json_path:
            splits_to_process.append(('train', args.train_json_path, args.train_path_saved))

    # Import inside main to clean globals
    global NAME_TO_IDX, NODE_NAMES, ADJ_GLOBAL

    for split_name, data_source, output_path in splits_to_process:
        if os.path.exists(output_path):
            print(f"Skipping {split_name} split as {output_path} already exists.")
            continue
            
        if isinstance(data_source, str):
            print(f"Loading {data_source}...")
            with open(data_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = data_source

        print(f"\n=== Processing {split_name} split ({len(data)} items) ===")
        
        # Build split-specific graph
        dfs_to_concat = []
        if split_name == 'valid':
            print(f"Building graph for valid split using ONLY training edges from {args.train_raw}...")
            train_df = load_raw(args.train_raw)
            if not train_df.empty:
                dfs_to_concat.append(map_graph(train_df, entity2id, relation2id))
        elif split_name == 'test':
            print(f"Building graph for test split using train + valid edges...")
            train_df = load_raw(args.train_raw)
            valid_df = load_raw(args.valid_raw)
            if not train_df.empty:
                dfs_to_concat.append(map_graph(train_df, entity2id, relation2id))
            if not valid_df.empty:
                dfs_to_concat.append(map_graph(valid_df, entity2id, relation2id))
        else:
            # Default fallback for other splits (e.g. train)
            train_df = load_raw(args.train_raw)
            if not train_df.empty:
                dfs_to_concat.append(map_graph(train_df, entity2id, relation2id))

        if not dfs_to_concat:
            print(f"Error: No graph data for {split_name}.")
            continue

        tv_id = pd.concat(dfs_to_concat)
        edges_df = tv_id[[0, 2, 1]]
        G = ig.Graph.TupleList(edges_df.itertuples(index=False), directed=False, edge_attrs=['relation'])
        
        # Reset globals to force rebuild for this G
        NAME_TO_IDX = None
        NODE_NAMES = None
        ADJ_GLOBAL = None
        
        batch_size = 500 # Small batch
        total_items = len(data)
        num_procs = min(4, mp.cpu_count())
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write("[\n")
            for i in range(0, total_items, batch_size):
                batch_data = data[i : i + batch_size]
                
                for raw in batch_data:
                    add_prompt(raw, relation_questions_A_to_B, relation_questions_B_to_A, args.bkg)
                
                subgraphs = subgraph_func(batch_data, args.graph_size, G, rules, num_processes=num_procs)
                
                for j in range(len(batch_data)):
                    batch_data[j]['subgraph'] = subgraphs[j]
                
                for k, item in enumerate(batch_data):
                    is_last_item = (i + k == total_items - 1)
                    json_str = json.dumps(item, ensure_ascii=False, indent=4, default=default)
                    indented_json = "\n".join(["    " + line for line in json_str.split("\n")])
                    f_out.write(indented_json)
                    if not is_last_item:
                        f_out.write(",\n")
                    else:
                        f_out.write("\n")
                
                del batch_data, subgraphs
                gc.collect()
            f_out.write("]")
        
        del data, G, tv_id, edges_df
        gc.collect()
        print(f"Finished {split_name} split.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_raw", required=True, help="Raw KG training set.")
    parser.add_argument("--valid_raw", help="Raw KG validation set.")
    parser.add_argument("--test_raw", help="Raw KG testing set.")
    parser.add_argument("--entity2id_path", required=True, help="Entity mapping file.")
    parser.add_argument("--id2entity_path", required=True, help="Entity mapping file.")
    parser.add_argument("--id2relation_path", required=True, help="Relation mapping file.")
    parser.add_argument("--train_json_path", help="Coarse ranks.")
    parser.add_argument("--valid_json_path", help="Coarse ranks.")
    parser.add_argument("--test_json_path", help="Coarse ranks.")
    parser.add_argument("--train_path_saved", help="Training set for LLM.")
    parser.add_argument("--valid_path_saved", help="Validation set for LLM.")
    parser.add_argument("--test_path_saved", help="Testing set for LLM.")
    parser.add_argument("--tail_pred_lex", required=True, help="Tail lexicon.")
    parser.add_argument("--head_pred_lex", required=True, help="Head lexicon.")
    parser.add_argument("--rules_path", required=True, help="Logic rules.")
    parser.add_argument("--graph_size", type=int, required=True, help="Graph size limit.")
    parser.add_argument("--bkg", action="store_true", help="Biomedical KG?")
    parser.add_argument("--split_valid", action="store_true", help="Split valid_json into train/valid.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio.")

    args = parser.parse_args()
    main(args)
