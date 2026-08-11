"""
logatkg.module5_synthesizer.path_unroller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dịch thuật Cơ chế (Path Unrolling — SG-RAG 2024).

Trích xuất đường đi từ G_sub, dịch thành văn bản có hướng:
    Subject --[Relation]--> Object

Tự động đảo chiều ngữ pháp nếu truy vấn là Head Prediction.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PathUnroller:
    """Convert graph paths into human-readable mechanism text.

    Parameters
    ----------
    id2relation : dict[int, str]
        Relation ID → name mapping.
    id2entity : dict[int, str], optional
        Entity ID → name mapping (if entities in paths are represented by IDs).
    max_path_length : int
        Maximum path length to process.
    reverse_for_head : bool
        Whether to reverse grammar for Head Prediction queries.
    """

    def __init__(
        self,
        id2relation: dict[int, str],
        id2entity: dict[int, str] | None = None,
        max_path_length: int = 5,
        reverse_for_head: bool = True,
    ):
        self.id2relation = id2relation
        self.id2entity = id2entity or {}
        self.max_path_length = max_path_length
        self.reverse_for_head = reverse_for_head

    def _get_entity_name(self, ent) -> str:
        """Resolve entity ID to name if mapping is available, else return the entity name as string."""
        if self.id2entity and (isinstance(ent, int) or str(ent).isdigit()):
            return self.id2entity.get(int(ent), f"Entity_{ent}")
        return str(ent)

    def _get_relation_name(self, rel) -> str:
        """Resolve relation ID to name if mapping is available, else return the relation as string."""
        if isinstance(rel, (int, float)) or str(rel).isdigit():
            return self.id2relation.get(int(rel), f"Relation_{rel}")
        return str(rel)

    def unroll_triple(
        self,
        head: str | int,
        rel: str | int,
        tail: str | int,
        is_head_prediction: bool = False,
    ) -> str:
        """Convert a single triple to natural language text, always using entity names."""
        h_name = self._get_entity_name(head)
        t_name = self._get_entity_name(tail)
        r_name = self._get_relation_name(rel)

        if is_head_prediction and self.reverse_for_head:
            return f"{t_name} --[{r_name}]--> {h_name}"
        return f"{h_name} --[{r_name}]--> {t_name}"

    def unroll_path(
        self,
        path,
        is_head_prediction: bool = False,
    ) -> str:
        """Convert a single path to natural language text, always using entity names.

        Handles both LoGATKG RawPath objects (with .edges) and list of edges.
        """
        # Detect if it is a RawPath object or standard list
        edges = getattr(path, "edges", None)
        if edges is None:
            # Reconstruct from standard sequence of triples or nodes
            if isinstance(path, list):
                edges = path
            else:
                return ""

        if not edges:
            nodes = getattr(path, "nodes", None)
            if nodes:
                return self._get_entity_name(nodes[0])
            return ""

        edges = edges[:self.max_path_length]

        if is_head_prediction and self.reverse_for_head:
            reversed_edges = []
            for edge in reversed(edges):
                if len(edge) == 4:
                    curr, rel, neighbor, is_rev = edge
                    reversed_edges.append((neighbor, rel, curr, not is_rev))
                elif len(edge) == 3:
                    curr, rel, neighbor = edge
                    reversed_edges.append((neighbor, rel, curr))
            edges = reversed_edges

        parts: list[str] = []
        for i, edge in enumerate(edges):
            if len(edge) == 4:
                curr, rel, neighbor, is_rev = edge
                curr_name = self._get_entity_name(curr)
                neighbor_name = self._get_entity_name(neighbor)
                r_name = self._get_relation_name(rel)
                
                if i == 0:
                    if is_rev:
                        parts.append(f"{curr_name} <--[{r_name}]-- {neighbor_name}")
                    else:
                        parts.append(f"{curr_name} --[{r_name}]--> {neighbor_name}")
                else:
                    if is_rev:
                        parts.append(f"<--[{r_name}]-- {neighbor_name}")
                    else:
                        parts.append(f"--[{r_name}]--> {neighbor_name}")
            elif len(edge) == 3:
                h, r_or_t, t_or_r = edge[0], edge[1], edge[2]
                if str(r_or_t).isdigit() or r_or_t in self.id2relation:
                    h_name = self._get_entity_name(h)
                    t_name = self._get_entity_name(t_or_r)
                    r_name = self._get_relation_name(r_or_t)
                else:
                    h_name = self._get_entity_name(h)
                    t_name = self._get_entity_name(r_or_t)
                    r_name = self._get_relation_name(t_or_r)
                if i == 0:
                    parts.append(f"{h_name} --[{r_name}]--> {t_name}")
                else:
                    parts.append(f"--[{r_name}]--> {t_name}")
            else:
                continue

        return " ".join(parts)

    def unroll_subgraph(
        self,
        subgraph,
        is_head_prediction: bool = False,
    ) -> list[str]:
        """Convert all paths/edges in a subgraph to natural language text.

        Handles both LoGATKG Subgraph objects (with .paths) and list of triples.
        """
        texts = []
        paths = getattr(subgraph, "paths", None)
        if paths is not None:
            for path in paths:
                text = self.unroll_path(path, is_head_prediction)
                if text:
                    texts.append(text)
        elif isinstance(subgraph, list):
            for edge in subgraph:
                if len(edge) == 3:
                    h, r, t = edge
                    texts.append(self.unroll_triple(h, r, t, is_head_prediction))
        
        logger.debug(f"Unrolled {len(texts)} paths into mechanism text")
        return texts
