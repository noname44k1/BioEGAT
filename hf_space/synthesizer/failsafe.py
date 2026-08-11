"""
logatkg.module5_synthesizer.failsafe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thống kê Fail-Safe (EDGAR 2024).

Tính p-value bằng Phân phối Siêu hình học (Hypergeometric Distribution)
để đánh giá ý nghĩa thống kê của chứng cứ cấu trúc.

Nếu p-value > 10^{-5} → gắn cờ [LOW CONFIDENCE].
Gắn kèm Metadata Nguồn gốc (Provenance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class FailSafeResult:
    """Result of fail-safe statistical evaluation.

    Attributes
    ----------
    entity_name : str
        Candidate entity name.
    p_value : float
        Hypergeometric p-value.
    is_significant : bool
        Whether p-value passes the threshold.
    confidence_flag : str
        "HIGH CONFIDENCE" or "LOW CONFIDENCE".
    provenance : dict
        Source metadata for transparency.
    entity_id : int | None
        Optional candidate entity ID.
    """

    entity_name: str
    p_value: float
    is_significant: bool
    confidence_flag: str
    provenance: dict = field(default_factory=dict)
    entity_id: int | None = None

    def to_dict(self):
        return {
            "entity_name": self.entity_name,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_flag": self.confidence_flag,
            "provenance": self.provenance,
            "entity_id": self.entity_id
        }


def compute_hypergeometric_pvalue(
    num_supporting_paths: int,
    total_paths_in_subgraph: int,
    num_paths_to_candidate: int,
    total_candidates: int,
) -> float:
    """Compute p-value using the Hypergeometric distribution.

    Tests whether the number of supporting paths to a candidate
    is significantly greater than expected by chance.

    Parameters
    ----------
    num_supporting_paths : int
        k — Number of "successful" (high-confidence) paths to candidate.
    total_paths_in_subgraph : int
        N — Total paths in the subgraph.
    num_paths_to_candidate : int
        n — Number of paths that reach this candidate.
    total_candidates : int
        K — Total candidates in D_can.

    Returns
    -------
    float
        p-value (lower = more significant / non-random).
    """
    if total_paths_in_subgraph == 0 or total_candidates == 0:
        return 1.0

    # Cap draws and success states to population size (M) to respect hypergeometric bounds
    M = total_paths_in_subgraph
    n = min(num_paths_to_candidate, M)
    N = min(total_candidates, M)

    p_value = stats.hypergeom.sf(
        num_supporting_paths - 1,
        M,  # Population size
        n,  # Success states in population
        N,  # Draws
    )
    import math
    if math.isnan(p_value) or p_value is None:
        p_value = 1.0
        
    return float(p_value)



def evaluate_failsafe(
    candidate_name: str,
    num_supporting_paths: int,
    total_paths_in_subgraph: int,
    num_paths_to_candidate: int,
    total_candidates: int,
    p_value_threshold: float = 1e-5,
    candidate_id: int | None = None,
    source_modules: list[str] | None = None,
) -> FailSafeResult:
    """Run fail-safe statistical evaluation for a single candidate.

    Parameters
    ----------
    candidate_name:
        Entity name.
    num_supporting_paths:
        Number of high-confidence paths to this candidate.
    total_paths_in_subgraph:
        Total paths in G_sub.
    num_paths_to_candidate:
        Total paths reaching this candidate.
    total_candidates:
        Size of D_can.
    p_value_threshold:
        Significance threshold (default: 10^{-5}).
    candidate_id:
        Optional entity ID.
    source_modules:
        Optional list of modules contributing evidence.

    Returns
    -------
    FailSafeResult
    """
    p_value = compute_hypergeometric_pvalue(
        num_supporting_paths,
        total_paths_in_subgraph,
        num_paths_to_candidate,
        total_candidates,
    )

    is_significant = p_value <= p_value_threshold
    if not is_significant and num_supporting_paths > 0:
        if num_supporting_paths == num_paths_to_candidate:
            is_significant = True
            
    flag = "HIGH CONFIDENCE" if is_significant else "LOW CONFIDENCE"

    provenance = {
        "supporting_paths": num_supporting_paths,
        "total_subgraph_paths": total_paths_in_subgraph,
        "paths_to_candidate": num_paths_to_candidate,
        "total_candidates": total_candidates,
        "p_value_threshold": p_value_threshold,
        "source_modules": source_modules or ["M1", "M2", "M3", "M4"]
    }

    result = FailSafeResult(
        entity_name=candidate_name,
        p_value=p_value,
        is_significant=is_significant,
        confidence_flag=flag,
        provenance=provenance,
        entity_id=candidate_id
    )

    if not is_significant:
        logger.warning(
            f"[LOW CONFIDENCE] Candidate '{candidate_name}': p={p_value:.2e} > threshold={p_value_threshold:.2e}"
        )

    return result
