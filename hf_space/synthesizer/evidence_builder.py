"""
Evidence Builder.

Connects the raw DrKGC JSON items (with subgraphs) to the
evidence synthesis components (PSR, Failsafe, Path Unrolling).
Supports NCRL multi-hop rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import logging
import math
import torch
from .psr_scorer import compute_psr_score, compute_path_confidence, compute_confidence_from_stats
from .failsafe import evaluate_failsafe, FailSafeResult
from .llm_reranker import EvidenceProfile

logger = logging.getLogger(__name__)


class EvidenceBuilder:
    def __init__(
        self,
        path_unroller,
        train_stats: dict | None = None,
        ncrl_rules: dict | None = None,
        ent_embeddings: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
        kge_alpha: float = 0.5,
        supporting_threshold: float = 0.5,
        fallback_penalty: float = 0.5,
        p_value_threshold: float = 1e-5,
        model_name: str = "TransE",
    ):
        """
        Parameters
        ----------
        path_unroller:
            Module to convert edges/subgraphs into human-readable text.
        train_stats:
            Dictionary containing global training graph statistics (degrees, etc.).
        ncrl_rules:
            Dictionary containing mined rules: {target_rel: [{"body": [...], "score": float}, ...]}
        ent_embeddings:
            Pre-trained KGE entity embeddings tensor.
        rel_embeddings:
            Pre-trained KGE relation embeddings tensor.
        kge_alpha:
            Scaling factor for negative exponential distance: exp(-alpha * distance).
        supporting_threshold:
            Confidence threshold to count a path as supporting.
        p_value_threshold:
            p-value significance threshold for fail-safe.
        """
        self.path_unroller = path_unroller
        self.train_stats = train_stats or {}
        self.ncrl_rules = ncrl_rules or {}
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        self.kge_alpha = kge_alpha
        self.supporting_threshold = supporting_threshold
        self.fallback_penalty = fallback_penalty
        self.p_value_threshold = p_value_threshold
        self.model_name = model_name
        
    def _compute_kge_path_score(self, path: list) -> float:
        """Compute KGE-based path scoring using TransE L2 vector distance or DistMult dot product similarity."""
        if self.ent_embeddings is None or self.rel_embeddings is None or not path:
            return 0.0
            
        if self.model_name.lower() == "distmult":
            total_score = 1.0
            for edge in path:
                if len(edge) == 4:
                    h, r, t, is_rev = edge
                    if is_rev:
                        h, t = t, h
                else:
                    h, r, t = edge
                if (h < 0 or h >= self.ent_embeddings.shape[0] or
                    r < 0 or r >= self.rel_embeddings.shape[0] or
                    t < 0 or t >= self.ent_embeddings.shape[0]):
                    continue
                h_emb = self.ent_embeddings[h]
                r_emb = self.rel_embeddings[r]
                t_emb = self.ent_embeddings[t]
                # DistMult score: sigmoid of dot product sum(h * r * t)
                score = torch.sigmoid(torch.sum(h_emb * r_emb * t_emb)).item()
                total_score *= score
            return total_score

        if self.model_name.lower() == "complex":
            total_score = 1.0
            dim = self.ent_embeddings.shape[1] // 2
            for edge in path:
                if len(edge) == 4:
                    h, r, t, is_rev = edge
                    if is_rev:
                        h, t = t, h
                else:
                    h, r, t = edge
                if (h < 0 or h >= self.ent_embeddings.shape[0] or
                    r < 0 or r >= self.rel_embeddings.shape[0] or
                    t < 0 or t >= self.ent_embeddings.shape[0]):
                    continue
                # Split real and imaginary parts
                h_re = self.ent_embeddings[h, :dim]
                h_im = self.ent_embeddings[h, dim:]
                r_re = self.rel_embeddings[r, :dim]
                r_im = self.rel_embeddings[r, dim:]
                t_re = self.ent_embeddings[t, :dim]
                t_im = self.ent_embeddings[t, dim:]
                
                # ComplEx score = sum(h_re * r_re * t_re + h_im * r_re * t_im + h_re * r_im * t_im - h_im * r_im * t_re)
                raw_score = torch.sum(
                    h_re * r_re * t_re + 
                    h_im * r_re * t_im + 
                    h_re * r_im * t_im - 
                    h_im * r_im * t_re
                )
                score = torch.sigmoid(raw_score).item()
                total_score *= score
            return total_score

        # Default legacy TransE distance: ||h + r - t||_2
        total_dist = 0.0
        for edge in path:
            if len(edge) == 4:
                h, r, t, is_rev = edge
                if is_rev:
                    h, t = t, h
            else:
                h, r, t = edge
            if (h < 0 or h >= self.ent_embeddings.shape[0] or
                r < 0 or r >= self.rel_embeddings.shape[0] or
                t < 0 or t >= self.ent_embeddings.shape[0]):
                continue
                
            h_emb = self.ent_embeddings[h]
            r_emb = self.rel_embeddings[r]
            t_emb = self.ent_embeddings[t]
            
            dist = torch.norm(h_emb + r_emb - t_emb, p=2).item()
            total_dist += dist
            
        return math.exp(-self.kge_alpha * total_dist)

    def _find_paths(self, head, target_node, subgraph, max_hops=4):
        """Find paths from head to target_node in the local subgraph.
        Returns a list of paths, where each path is a list of (h, r, t, is_reverse) tuples.
        """
        adj = defaultdict(list)
        for h, r, t in subgraph:
            adj[h].append((r, t, False)) # Forward transition
            adj[t].append((r, h, True))  # Reverse transition
            
        found_paths = []
        
        def dfs(curr, target, current_path, visited_nodes, depth):
            if depth > max_hops:
                return
            if curr == target and depth > 0:
                found_paths.append(list(current_path))
                return
            
            for rel, neighbor, is_reverse in adj[curr]:
                if neighbor not in visited_nodes:
                    current_path.append((curr, rel, neighbor, is_reverse))
                    visited_nodes.add(neighbor)
                    dfs(neighbor, target, current_path, visited_nodes, depth + 1)
                    visited_nodes.remove(neighbor)
                    current_path.pop()

        dfs(head, target_node, [], {head}, 0)
        return found_paths

    def build_profiles(self, item: dict, top_k: int = 10) -> list[EvidenceProfile]:
        """Build evidence profiles for the top-K candidates in a JSON item."""
        query_rel = item.get("relation")
        is_head_prediction = item.get("type") == "predicted_head"
        
        # 1. Safely resolve Query fixed entity ID
        query_entity_id = item.get("query_entity_id")
        if query_entity_id is None:
            triple_id = item.get("triple_id")
            if triple_id and len(triple_id) >= 3:
                query_entity_id = triple_id[0] if not is_head_prediction else triple_id[2]
        
        fixed_entity_id = query_entity_id
        
        candidates = item.get("rank_entities", [])[:top_k]
        candidate_ids = item.get("rank_entities_id", [])[:top_k]
        raw_subgraph = item.get("subgraph", [])
        subgraph = []
        # Dynamically resolve name-based subgraphs back to integer IDs
        entity2id = {}
        if self.path_unroller.id2entity:
            for eid, name in self.path_unroller.id2entity.items():
                entity2id[str(name)] = int(eid)
        relation2id = {}
        if self.path_unroller.id2relation:
            for rid, rname in self.path_unroller.id2relation.items():
                relation2id[str(rname)] = int(rid)
                
        for h, r, t in raw_subgraph:
            h_id = int(h) if (isinstance(h, int) or str(h).isdigit()) else entity2id.get(str(h))
            r_id = int(r) if (isinstance(r, int) or str(r).isdigit()) else relation2id.get(str(r))
            t_id = int(t) if (isinstance(t, int) or str(t).isdigit()) else entity2id.get(str(t))
            if h_id is not None and r_id is not None and t_id is not None:
                subgraph.append([h_id, r_id, t_id])
        
        # Pre-index rules for the query relation for faster lookup
        active_rules = self.ncrl_rules.get(query_rel, [])
        # Map rule body tuple -> score
        rule_map = {tuple(r["body"]): r["score"] for r in active_rules}
        
        profiles = []
        total_paths_in_subgraph = len(subgraph)
        total_candidates = len(item.get("rank_entities", []))
        
        for rank, cand_name in enumerate(candidates, 1):
            cand_id = candidate_ids[rank - 1] if rank - 1 < len(candidate_ids) else None
            
            paths = []
            if fixed_entity_id is not None and cand_id is not None:
                if is_head_prediction:
                    # Head prediction: candidates are heads, fixed entity is tail.
                    # DFS path from candidate (head) to fixed (tail)
                    paths = self._find_paths(cand_id, fixed_entity_id, subgraph, max_hops=5)
                else:
                    # Tail prediction: fixed entity is head, candidates are tails.
                    # DFS path from fixed (head) to candidate (tail)
                    paths = self._find_paths(fixed_entity_id, cand_id, subgraph, max_hops=5)
            
            # Pair each path with its calculated confidence score and text representation
            path_info = []
            has_ncrl_match = False
            for path in paths:
                path_names = [self.path_unroller._get_relation_name(edge[1]) for edge in path]
                path_tuple = tuple(path_names)
                
                if path_tuple in rule_map:
                    score = rule_map[path_tuple]
                    has_ncrl_match = True
                elif self.ent_embeddings is not None and self.rel_embeddings is not None:
                    score = self._compute_kge_path_score(path)
                else:
                    degrees = self.train_stats.get("global_node_degrees", {})
                    node_ids = set()
                    for edge in path:
                        node_ids.add(edge[0])
                        node_ids.add(edge[2])
                    node_degrees = [degrees.get(nid, 0) for nid in node_ids]
                    score = compute_path_confidence(path_length=len(path), node_degrees=node_degrees, max_path_length=5)
                
                text = self.path_unroller.unroll_path(path, is_head_prediction)
                if text:
                    path_info.append((score, text))
                    
            # Sort paths by their score descending to prioritize high-confidence paths
            path_info.sort(key=lambda x: x[0], reverse=True)
            
            path_scores = [info[0] for info in path_info]
            cand_mechanism_texts = [info[1] for info in path_info]
            
            # Fallback: if no multi-hop paths are found at all, look at 1-hop edges connected to candidate
            has_fallback_match = False
            if not path_scores and cand_id is not None:
                for i, edge in enumerate(subgraph):
                    if len(edge) == 3 and (cand_id == edge[0] or cand_id == edge[2]):
                        # If KGE embeddings are available, use KGE scoring for the single edge with fallback penalty
                        if self.ent_embeddings is not None and self.rel_embeddings is not None:
                            score = self.fallback_penalty * self._compute_kge_path_score([edge])
                        else:
                            # Use global stats heuristic for 1-hop edges
                            score = compute_confidence_from_stats(edge, self.train_stats)
                        path_scores.append(score)
                        text = self.path_unroller.unroll_triple(edge[0], edge[1], edge[2], is_head_prediction)
                        if text:
                            cand_mechanism_texts.append(text)
                            has_fallback_match = True
            
            psr_score = compute_psr_score(path_scores)
            
            # Calculate supporting paths (k) vs total paths to candidate (n)
            # Use customizable self.supporting_threshold instead of static 0.35
            num_paths_to_candidate = len(path_scores)
            num_supporting_paths = sum(1 for s in path_scores if s >= self.supporting_threshold)
            
            # Determine dynamic source modules that contributed evidence
            active_modules = ["M1"] # M1 (coarse ranker) is always active
            if psr_score > 0.0:
                active_modules.append("M2") # M2 (PSR symbolic scorer)
            if has_ncrl_match:
                active_modules.append("M3") # M3 (NCRL logic rules)
            
            # Fail-safe statistical evaluation
            failsafe = evaluate_failsafe(
                candidate_name=cand_name,
                num_supporting_paths=num_supporting_paths,
                total_paths_in_subgraph=total_paths_in_subgraph,
                num_paths_to_candidate=num_paths_to_candidate,
                total_candidates=total_candidates,
                p_value_threshold=self.p_value_threshold,
                candidate_id=cand_id,
                source_modules=active_modules
            )
            
            # Append M4 to active modules if the hypergeometric filter deemed it significant
            if failsafe.is_significant:
                active_modules.append("M4")
                failsafe.provenance["source_modules"] = list(active_modules)
            
            profiles.append(EvidenceProfile(
                candidate_name=cand_name,
                gnn_rank=rank,
                mechanism_texts=cand_mechanism_texts[:5],
                psr_score=psr_score,
                failsafe=failsafe,
                candidate_id=cand_id,
                path_scores=path_scores[:5]
            ))
            
        return profiles
