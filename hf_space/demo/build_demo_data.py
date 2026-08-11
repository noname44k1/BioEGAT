#!/usr/bin/env python3
"""
build_demo_data.py
~~~~~~~~~~~~~~~~~~
Offline builder: extracts 20 showcase queries per dataset from
prediction.json and produces compact <ds>_demo.json files for the
BioEGAT Gradio demo.

Usage:
    python -m demo.build_demo_data --dataset primekg
    python -m demo.build_demo_data --dataset all
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from collections import defaultdict

# Add project root to path for synthesizer imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Showcase triple_id lists (deterministic) ─────────────────────────
# Indices into prediction["prediction"] list for each dataset.
# Selected for:
#   - Gold recovered at rank 1 (clean wins)
#   - Full-pipeline lift cases (gold low in KGE → rank 1 after LLM)
#   - Mix of predicted_tail and predicted_head
#   - Recognisable biomedical entities

SHOWCASES = {
    "primekg": [
        # Lift cases (coarse rank >= 5, final rank = 1)
        70,   # Atenolol → hypertension (coarse=32, tail)
        169,  # aspergillosis → Voriconazole (coarse=38, head)
        274,  # Pramiracetam → Alzheimer disease (coarse=1569, tail)
        289,  # asthma → Prednisone (coarse=46, head)
        351,  # otitis externa → Ciprofloxacin (coarse=34, head)
        410,  # Leflunomide → rheumatoid arthritis (coarse=48, tail)
        512,  # Hydroxychloroquine → rheumatoid arthritis (coarse=41, tail)
        679,  # absence epilepsy → Valproic acid (coarse=53, head)
        # Clean wins (pred_rank=1, recognisable drugs/diseases)
        4,    # Metyrosine → adrenal gland pheochromocytoma (tail)
        6,    # Prednisolone → Hodgkins lymphoma (tail)
        10,   # Carbutamide → type 2 diabetes mellitus (tail)
        16,   # Fosamprenavir → AIDS (tail)
        20,   # Carmustine → glioblastoma (tail)
        7,    # Hodgkins lymphoma → Prednisolone (head)
        11,   # type 2 diabetes mellitus → Carbutamide (head)
        13,   # Crohn disease → Prednisolone (head)
        # A few more for variety
        388,  # Allantoin → psoriasis (coarse=30, tail, lift)
        575,  # hereditary persistence of fetal hemoglobin → Hydroxyurea (coarse=3291, head, lift)
        789,  # schistosomiasis → Praziquantel (coarse=49, head, lift)
        835,  # T-cell/histiocyte rich large B-cell lymphoma → Dexamethasone (head, lift)
    ],
}

# ── Dataset configurations ───────────────────────────────────────────

DATASET_CONFIGS = {
    "primekg": {
        "label": "PrimeKG",
        "prediction_path": "dataset-logos/primekg/prediction.json",
        "id2entity_path": "dataset-logos/primekg/id2entity.pkl",
        "id2relation_path": "dataset-logos/primekg/id2relation.pkl",
        "entity2id_path": "dataset-logos/primekg/entity2id.pkl",
    },
    "drugmechdb": {
        "label": "DrugMechDB",
        "prediction_path": "dataset-logos/drugmechdb/prediction.json",
        "id2entity_path": "dataset-logos/drugmechdb/id2entity.pkl",
        "id2relation_path": "dataset-logos/drugmechdb/id2relation.pkl",
        "entity2id_path": "dataset-logos/drugmechdb/entity2id.pkl",
    },
    "hetionet": {
        "label": "Hetionet",
        "prediction_path": "dataset-logos/hetionet/prediction.json",
        "id2entity_path": "dataset-logos/hetionet/id2entity.pkl",
        "id2relation_path": "dataset-logos/hetionet/id2relation.pkl",
        "entity2id_path": "dataset-logos/hetionet/entity2id.pkl",
    },
    "pharmkg": {
        "label": "PharmKG",
        "prediction_path": "dataset-logos/pharmkg/prediction.json",
        "id2entity_path": "dataset-logos/pharmkg/id2entity.pkl",
        "id2relation_path": "dataset-logos/pharmkg/id2relation.pkl",
        "entity2id_path": "dataset-logos/pharmkg/entity2id.pkl",
    },
}


def load_mappings(config: dict) -> tuple:
    """Load id2entity, id2relation, entity2id pickles."""
    with open(config["id2entity_path"], "rb") as f:
        id2entity = pickle.load(f)
    with open(config["id2relation_path"], "rb") as f:
        id2relation = pickle.load(f)
    with open(config["entity2id_path"], "rb") as f:
        entity2id = pickle.load(f)
    return id2entity, id2relation, entity2id


def resolve_subgraph_names(subgraph: list, id2entity: dict, id2relation: dict) -> list:
    """Convert subgraph triples from IDs to entity/relation names.
    
    Returns list of [head_name, relation_name, tail_name].
    Skips triples where any ID can't be resolved.
    """
    named = []
    for triple in subgraph:
        if len(triple) != 3:
            continue
        h_id, r_id, t_id = triple
        h_name = id2entity.get(h_id) if isinstance(h_id, int) else h_id
        r_name = id2relation.get(r_id) if isinstance(r_id, int) else r_id
        t_name = id2entity.get(t_id) if isinstance(t_id, int) else t_id
        if h_name is not None and r_name is not None and t_name is not None:
            named.append([str(h_name), str(r_name), str(t_name)])
    return named


def compute_evidence_profiles(
    record: dict,
    named_subgraph: list,
    id2entity: dict,
) -> list:
    """Compute evidence profiles for top-10 candidates.
    
    For each candidate, compute:
    - mechanism_texts: unrolled paths from query to candidate
    - support: number of ≤3-hop paths
    - psr_score, pvalue, tier (basic computation)
    """
    import networkx as nx

    query = record.get("query_entity", "")
    candidates = record.get("rank_entities", [])[:10]
    rerank_scores = record.get("rerank_scores", [])

    # Build graph from named subgraph
    G = nx.DiGraph()
    for triple in named_subgraph:
        if len(triple) == 3:
            G.add_edge(triple[0], triple[2], relation=triple[1])

    profiles = []
    total_paths_in_subgraph = len(named_subgraph)

    for i, candidate in enumerate(candidates):
        # Find paths from query to candidate (up to 3 hops)
        mechanism_texts = []
        path_count = 0

        try:
            # Use undirected graph for path finding
            G_undirected = G.to_undirected()
            for path in nx.all_simple_paths(G_undirected, query, candidate, cutoff=3):
                path_count += 1
                # Build mechanism text
                parts = []
                for j in range(len(path) - 1):
                    src, dst = path[j], path[j + 1]
                    # Try both directions for the relation
                    rel = "?"
                    if G.has_edge(src, dst):
                        rel = G[src][dst].get("relation", "?")
                    elif G.has_edge(dst, src):
                        rel = G[dst][src].get("relation", "?")
                    if j == 0:
                        parts.append(f"{src} --[{rel}]--> {dst}")
                    else:
                        parts.append(f" --[{rel}]--> {dst}")
                if parts:
                    mechanism_texts.append("".join(parts))
                if len(mechanism_texts) >= 5:
                    break
        except (nx.NodeNotFound, nx.NetworkXError):
            pass

        # Simple PSR approximation
        psr_score = path_count / max(total_paths_in_subgraph, 1)

        # Simple p-value approximation (binomial-like)
        if path_count > 0 and total_paths_in_subgraph > 0:
            expected = total_paths_in_subgraph / max(len(candidates), 1)
            if path_count > expected * 2:
                pvalue = 1e-6
                tier = "HIGH CONFIDENCE"
            elif path_count > expected:
                pvalue = 0.01
                tier = "MEDIUM CONFIDENCE"
            else:
                pvalue = 0.1
                tier = "LOW CONFIDENCE"
        else:
            pvalue = 1.0
            tier = "LOW CONFIDENCE"

        certainty = rerank_scores[i] if rerank_scores and i < len(rerank_scores) else 0.0

        profile = {
            "candidate_name": candidate,
            "gnn_rank": i + 1,
            "mechanism_texts": mechanism_texts,
            "psr_score": round(psr_score, 6),
            "failsafe": {
                "entity_name": candidate,
                "p_value": pvalue,
                "is_significant": pvalue < 0.05,
                "confidence_flag": tier,
                "provenance": {
                    "supporting_paths": path_count,
                    "total_subgraph_paths": total_paths_in_subgraph,
                    "paths_to_candidate": len(mechanism_texts),
                    "total_candidates": len(candidates),
                },
            },
            "candidate_id": record.get("rank_entities_id", [None] * 10)[i] if i < len(record.get("rank_entities_id", [])) else -1,
            "path_scores": [],
            "rerank_certainty": certainty,
        }
        ev = EVIDENCE.get((query, candidate))
        if ev:
            profile["evidence_source"], profile["evidence_id"] = ev
            profile["evidence_url"] = evidence_url(ev[1])
        profiles.append(profile)

    return profiles


def build_demo_record(pred_record: dict, rr_record: dict | None,
                      id2entity: dict, id2relation: dict) -> dict:
    """Merge the LLM prediction (from prediction.json) with the BioEGAT-reranked
    candidate list + subgraph (from reranked_test.json, joined by triple_id).

    - Hero / ``pred`` / ``pred_rank`` / ``target``  : prediction.json
    - The candidate list (``rank_entities``) + subgraph : reranked_test.json (when available)
    """
    src = rr_record or pred_record          # candidate list + subgraph: prefer reranked_test

    # Synthesize a descending rerank-score proxy when the file has none
    rerank_scores = src.get("rerank_scores", [])
    if not rerank_scores:
        K = len(src.get("rank_entities", []))
        raw = [1.0 / (i + 1) for i in range(K)]
        s = sum(raw) or 1.0
        rerank_scores = [round(x / s, 4) for x in raw]
        src["rerank_scores"] = rerank_scores      # so evidence profiles can read it

    # Resolve subgraph IDs to names (bounded for file size)
    named_subgraph = resolve_subgraph_names(src.get("subgraph", []), id2entity, id2relation)
    if len(named_subgraph) > 120:
        named_subgraph = named_subgraph[:120]

    evidence_profiles = compute_evidence_profiles(src, named_subgraph, id2entity)

    # Real-style prompt (graph context is injected as embeddings, not serialised as text)
    q = pred_record["query_entity"]
    relation = pred_record["triple"][1]
    slot = "tail" if pred_record["type"] == "predicted_tail" else "head"
    cand_lines = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(src.get("rank_entities", [])[:10]))
    llm_prompt = (
        "You are a biomedical scientist performing knowledge-graph completion.\n"
        f"Query: ({q}, {relation}, ?)  -- predict the most likely {slot} entity.\n"
        "Candidates (BioEGAT rerank order):\n"
        f"{cand_lines}\n"
        "The retrieved subgraph is injected as graph embeddings at the [QUERY]/[ENTITY] tokens.\n"
        "Answer with the single most likely entity name."
    )

    return {
        "triple": pred_record["triple"],
        "triple_id": pred_record["triple_id"],
        "type": pred_record["type"],
        "query_entity": q,
        "query_entity_id": pred_record.get("query_entity_id"),
        "relation": relation,
        "rank_entities": src.get("rank_entities", [])[:20],          # from reranked_test.json
        "rank_entities_id": src.get("rank_entities_id", [])[:20],
        "rank": pred_record.get("rank", -1),
        "pred_rank": pred_record.get("pred_rank", -1),               # from prediction.json
        "pred": pred_record.get("pred", ""),                         # LLM prediction (hero)
        "target": pred_record.get("target", ""),
        "subgraph": named_subgraph,                                  # from reranked_test.json
        "rerank_scores": rerank_scores[:20],
        "evidence_profiles": evidence_profiles,
        "llm_prompt": llm_prompt,
    }


def auto_select_showcases(predictions: list, n: int = 20) -> list:
    """Automatically select n showcase queries from predictions.
    
    Strategy:
    1. All lift cases (coarse rank >= 5, pred_rank = 1)
    2. Fill remaining with correct predictions (pred_rank = 1)
    3. Mix of predicted_tail and predicted_head
    """
    lifts = []
    correct_tail = []
    correct_head = []
    
    for i, d in enumerate(predictions):
        pr = d.get("pred_rank", 999)
        cr = d.get("rank", 999)
        ptype = d.get("type", "")
        
        if isinstance(cr, float):
            cr = int(cr)
        if isinstance(pr, float):
            pr = int(pr)
        
        if cr >= 5 and pr == 1:
            lifts.append(i)
        elif pr == 1 and ptype == "predicted_tail":
            correct_tail.append(i)
        elif pr == 1 and ptype == "predicted_head":
            correct_head.append(i)
    
    selected = list(lifts[:min(len(lifts), n // 2)])
    remaining = n - len(selected)
    
    # Alternate tail and head
    tail_idx = 0
    head_idx = 0
    while len(selected) < n:
        if tail_idx < len(correct_tail) and len(selected) < n:
            idx = correct_tail[tail_idx]
            if idx not in selected:
                selected.append(idx)
            tail_idx += 1
        if head_idx < len(correct_head) and len(selected) < n:
            idx = correct_head[head_idx]
            if idx not in selected:
                selected.append(idx)
            head_idx += 1
        if tail_idx >= len(correct_tail) and head_idx >= len(correct_head):
            break
    
    return selected[:n]


# ── Verified external evidence (ClinicalTrials.gov / PubMed) ─────────
# Curated + verified for pharmkg/imatinib (FuseLinker-style). (query, candidate) -> (source, id)
EVIDENCE = {
    ("imatinib", "myeloproliferative disorders"): ("Clinical trial", "NCT00006343"),
    ("imatinib", "inflammation"):                  ("Clinical trial", "NCT04394416"),
    ("imatinib", "infarction"):                    ("Literature", "PMID:36639597"),
    ("imatinib", "urination disorders"):           ("Literature", "PMID:35528149"),
    ("imatinib", "uterine diseases"):              ("Literature", "PMID:38279510"),
    ("imatinib", "thrombocytopenia"):              ("Literature", "PMID:36374396"),
    ("imatinib", "neuroectodermal tumors"):        ("Clinical trial", "NCT00062205"),
    ("imatinib", "infections"):                    ("Clinical trial", "NCT04394416"),
    ("imatinib", "lupus erythematosus systemic"):  ("Literature", "PMID:16688113"),
    ("imatinib", "ovarian epithelial cancer"):     ("Clinical trial", "NCT00510653"),
}


def evidence_url(ident: str) -> str:
    if ident.startswith("NCT"):
        return f"https://clinicaltrials.gov/study/{ident}"
    if ident.startswith("PMID:"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{ident[5:]}/"
    return ""


def stream_array(path):
    """Yield objects from a large JSON array without loading it all (bounded memory)."""
    dec = json.JSONDecoder()
    buf = ""
    started = False
    with open(path, encoding="utf-8") as f:
        while True:
            ch = f.read(1 << 20)
            if not ch and not buf.strip():
                break
            buf += ch
            if not started:
                i = buf.find("[")
                if i == -1:
                    continue
                buf = buf[i + 1:]
                started = True
            while True:
                buf = buf.lstrip()
                if not buf or buf[0] == "]":
                    break
                if buf[0] == ",":
                    buf = buf[1:].lstrip()
                    continue
                try:
                    obj, idx = dec.raw_decode(buf)
                except ValueError:
                    break
                yield obj
                buf = buf[idx:]
            if not ch:
                break


def collect_records(pred_path, ds_name: str, n: int = 20) -> list:
    """Stream prediction.json (handles 1 GB+ files) and return selected raw records."""
    if ds_name in SHOWCASES:
        want = set(SHOWCASES[ds_name])
        found = {}
        for idx, rec in enumerate(stream_array(pred_path)):
            if idx in want:
                found[idx] = rec
                if len(found) == len(want):
                    break
        return [found[i] for i in SHOWCASES[ds_name] if i in found]
    # streaming auto-select: forced queries first, then lift cases, then clean wins
    force = {"pharmkg": {"imatinib", "methotrexate"}}.get(ds_name, set())
    forced, lifts, ctail, chead = {}, [], [], []
    seen_q = set()
    for d in stream_array(pred_path):
        q = d.get("query_entity")
        try:
            pr = int(float(d.get("pred_rank", 999)))
        except (TypeError, ValueError):
            pr = 999
        try:
            cr = int(float(d.get("rank", 999)))
        except (TypeError, ValueError):
            cr = 999
        ptype = d.get("type", "")
        if q in force and q not in forced and pr <= 3:
            forced[q] = d
        if q not in seen_q:
            if cr >= 5 and pr == 1 and len(lifts) < n:
                lifts.append(d); seen_q.add(q)
            elif pr == 1 and ptype == "predicted_tail" and len(ctail) < n:
                ctail.append(d); seen_q.add(q)
            elif pr == 1 and ptype == "predicted_head" and len(chead) < n:
                chead.append(d); seen_q.add(q)
        full = len(lifts) >= n and len(ctail) >= n and len(chead) >= n
        if full and (not force or len(forced) == len(force)):
            break
    sel = [forced[q] for q in force if q in forced]
    chosen_q = {r.get("query_entity") for r in sel}
    sel += [d for d in lifts[: n // 2] if d.get("query_entity") not in chosen_q]
    ti = hi = 0
    while len(sel) < n and (ti < len(ctail) or hi < len(chead)):
        if ti < len(ctail):
            if ctail[ti].get("query_entity") not in {r.get("query_entity") for r in sel}:
                sel.append(ctail[ti])
            ti += 1
        if len(sel) < n and hi < len(chead):
            if chead[hi].get("query_entity") not in {r.get("query_entity") for r in sel}:
                sel.append(chead[hi])
            hi += 1
    return sel[:n]


def build_dataset(ds_name: str, project_root: Path) -> None:
    """Build demo data for a single dataset."""
    config = DATASET_CONFIGS.get(ds_name)
    if not config:
        logger.error(f"Unknown dataset: {ds_name}")
        return

    # Resolve paths relative to project root
    pred_path = project_root / config["prediction_path"]
    if not pred_path.exists():
        logger.error(f"Prediction file not found: {pred_path}")
        return

    logger.info(f"=== Building demo data for {config['label']} ===")
    
    # Load mappings
    id2entity, id2relation, entity2id = load_mappings({
        k: str(project_root / v) for k, v in config.items()
        if k.endswith("_path") and k != "prediction_path"
    })
    logger.info(f"Loaded mappings: {len(id2entity)} entities, {len(id2relation)} relations")

    # Stream + select showcase records from prediction.json (bounded memory; handles 1 GB+ pharmkg)
    logger.info(f"Streaming predictions from {pred_path} ...")
    records = collect_records(pred_path, ds_name, n=20)
    logger.info(f"Selected {len(records)} showcase records")

    # Join the reranked candidate list + subgraph from reranked_test.json.
    # Key on (triple_id, type): head- and tail-prediction share a triple_id but have
    # different candidate lists, so type must disambiguate them.
    rr_path = project_root / config["prediction_path"].replace("prediction.json", "reranked_test.json")
    want_keys = {(tuple(r["triple_id"]), r.get("type")) for r in records}
    rr_by_key = {}
    if rr_path.exists():
        logger.info(f"Joining reranked candidates from {rr_path} ...")
        for o in stream_array(rr_path):
            key = (tuple(o.get("triple_id", [])), o.get("type"))
            if key in want_keys:
                rr_by_key[key] = o
                if len(rr_by_key) == len(want_keys):
                    break
        logger.info(f"Matched {len(rr_by_key)}/{len(want_keys)} reranked records")
    else:
        logger.warning(f"reranked_test.json not found ({rr_path}); using prediction.json candidate order")

    demo_records = []
    for record in records:
        rr = rr_by_key.get((tuple(record["triple_id"]), record.get("type")))
        demo = build_demo_record(record, rr, id2entity, id2relation)
        demo_records.append(demo)
        logger.info(
            f"  {demo['query_entity']} -> {demo.get('target', '?')} "
            f"(pred_rank={demo.get('pred_rank')}, coarse={demo.get('rank')}, type={demo['type']})"
        )

    # Write output
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ds_name}_demo.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(demo_records, f, ensure_ascii=False, indent=2)
    
    file_size = output_path.stat().st_size
    logger.info(f"Wrote {len(demo_records)} records to {output_path} ({file_size / 1024:.1f} KB)")
    
    if file_size > 512 * 1024:
        logger.warning(f"File size {file_size / 1024:.1f} KB exceeds 500 KB target!")

    return ds_name, config["label"], output_path.name, len(demo_records)


def build_index(results: list) -> None:
    """Write data/index.json listing available datasets."""
    output_dir = Path(__file__).parent / "data"
    index = {}
    for ds_name, label, filename, count in results:
        index[ds_name] = {
            "label": label,
            "file": filename,
            "n": count,
        }
    
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Wrote index.json with {len(index)} datasets")


def main():
    parser = argparse.ArgumentParser(description="Build demo data for BioEGAT HF Space")
    parser.add_argument(
        "--dataset", type=str, default="primekg",
        help="Dataset name (primekg, drugmechdb, hetionet, pharmkg) or 'all'"
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    
    if args.dataset == "all":
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset.lower()]

    results = []
    for ds in datasets:
        result = build_dataset(ds, project_root)
        if result:
            results.append(result)

    if results:
        build_index(results)


if __name__ == "__main__":
    main()
