"""
logatkg.module5_synthesizer.psr_scorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Probabilistic Soft Reranking (PSR) — MedKGent 2025.

Đo lường sức nặng của chứng cứ văn bản (evidence weight):
    S_PSR(t) = 1 - Π_i∈P_t (1 - s_i)

Trong đó s_i là confidence score của mỗi đường đi dẫn tới ứng viên t.
"""

from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)


def compute_psr_score(
    path_scores: list[float],
    min_threshold: float = 0.01,
) -> float:
    """Compute PSR (Probabilistic Soft Reranking) score for a candidate.

    S_PSR(t) = 1 - Π(1 - s_i) for all paths leading to candidate t.

    Intuition: even if individual paths are weak (low s_i),
    multiple independent paths accumulate strong evidence.

    Parameters
    ----------
    path_scores:
        Confidence scores for each path to this candidate.
        Each score should be in [0, 1].
    min_threshold:
        Minimum threshold for individual path scores.

    Returns
    -------
    float
        PSR score ∈ [0, 1]. Higher = stronger cumulative evidence.
    """
    if not path_scores:
        return 0.0

    # Clamp scores to [min_threshold, 1.0]
    clamped = [max(min(s, 1.0), min_threshold) for s in path_scores]

    # S_PSR = 1 - Π(1 - s_i)
    product = 1.0
    for s in clamped:
        product *= (1.0 - s)

    psr = 1.0 - product
    return psr


def compute_path_confidence(
    path_length: int,
    node_degrees: list[int] | None = None,
    max_path_length: int = 5,
) -> float:
    """Heuristic confidence score for a single path.

    Shorter paths and paths through lower-degree nodes get higher scores.

    Parameters
    ----------
    path_length:
        Number of edges in the path.
    node_degrees:
        Degrees of intermediate nodes (optional).
    max_path_length:
        Maximum expected path length for normalization.

    Returns
    -------
    float
        Confidence score ∈ (0, 1].
    """
    # Length penalty: shorter paths are more confident
    length_score = 1.0 / (1.0 + path_length / max_path_length)

    # Degree penalty: paths through hub nodes are less specific
    if node_degrees and len(node_degrees) > 0:
        avg_degree = sum(node_degrees) / len(node_degrees)
        degree_score = 1.0 / (1.0 + math.log1p(avg_degree) / 10.0)
    else:
        degree_score = 1.0

    return length_score * degree_score


def batch_psr_scores(
    candidate_path_scores: dict[str | int, list[float]],
    min_threshold: float = 0.01,
) -> dict[str | int, float]:
    """Compute PSR scores for a batch of candidates.

    Parameters
    ----------
    candidate_path_scores:
        Mapping from candidate (entity name or ID) → list of path confidence scores.

    Returns
    -------
    dict[str | int, float]
        Mapping from candidate (entity name or ID) → PSR score.
    """
    result = {}
    for key, scores in candidate_path_scores.items():
        result[key] = compute_psr_score(scores, min_threshold)

    logger.debug(f"Computed PSR scores for {len(result)} candidates")
    return result


def compute_confidence_from_stats(
    edge: list,
    global_stats: dict,
) -> float:
    """Compute confidence for a single edge using global training statistics.

    Parameters
    ----------
    edge: [head, relation, tail]
    global_stats: Dictionary containing 'global_node_degrees'.
    """
    if not edge or len(edge) < 3:
        return 0.0

    head, rel, tail = edge[0], edge[1], edge[2]
    degrees = global_stats.get("global_node_degrees", {})

    # Get degrees of head and tail from training graph
    h_deg = degrees.get(head, 0)
    t_deg = degrees.get(tail, 0)

    # A path of length 1 (a single edge)
    return compute_path_confidence(
        path_length=1,
        node_degrees=[h_deg, t_deg]
    )
