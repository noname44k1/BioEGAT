"""Module 5: Mechanistic Synthesizer & LLM Reranker — White-box reasoning."""

from .psr_scorer import compute_psr_score, compute_path_confidence, batch_psr_scores
from .failsafe import FailSafeResult, evaluate_failsafe
from .path_unroller import PathUnroller
from .evidence_builder import EvidenceBuilder
from .llm_reranker import LLMReranker, EvidenceProfile

__all__ = [
    "compute_psr_score",
    "compute_path_confidence",
    "batch_psr_scores",
    "FailSafeResult",
    "evaluate_failsafe",
    "PathUnroller",
    "EvidenceBuilder",
    "LLMReranker",
    "EvidenceProfile"
]
