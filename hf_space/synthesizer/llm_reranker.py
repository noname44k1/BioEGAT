"""
logatkg.module5_synthesizer.llm_reranker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LLM Reranker — In-Context Learning Prompt Builder.

Chuyển hóa biểu diễn Hộp đen (Black-box) của GAT thành
Hộp trắng (White-box) ngôn ngữ tự nhiên.

Đóng gói Evidence Profile (GNN score, Mechanism text, PSR, p-value,
Provenance) → inject vào Prompt → LLM (frozen) reasoning →
kết luận y khoa cuối cùng.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from .failsafe import FailSafeResult

logger = logging.getLogger(__name__)


@dataclass
class EvidenceProfile:
    """Complete evidence package for a candidate entity.

    Assembled from outputs of Modules 1–4 and Module 5 sub-components.
    """
    candidate_name: str
    gnn_rank: int
    mechanism_texts: list[str]
    psr_score: float
    failsafe: FailSafeResult
    candidate_id: int | None = None
    path_scores: list[float] | None = None
    rerank_certainty: float | None = None

    def to_dict(self):
        return {
            "candidate_name": self.candidate_name,
            "gnn_rank": self.gnn_rank,
            "mechanism_texts": self.mechanism_texts,
            "psr_score": self.psr_score,
            "failsafe": self.failsafe.to_dict(),
            "candidate_id": self.candidate_id,
            "path_scores": self.path_scores,
            "rerank_certainty": self.rerank_certainty
        }


# Default reranking prompt incorporating SG-RAG and EDGAR principles (ID-based few-shots)
RERANKER_PROMPT_TEMPLATE_ID = """You are a biomedical expert evaluating candidates for knowledge graph completion using Subgraph Retrieval-Augmented Generation (SG-RAG) and the Explainable Enrichment-Driven Graph Reasoner (EDGAR).

You are given a query and a set of candidate entities. Each candidate has an Evidence Profile with the following metrics:
- **Initial Rank**: The candidate's rank prior to path retrieval (lower rank is better).
- **GNN Rerank Certainty**: The GNN's structural confidence score (higher values indicate stronger neighborhood alignment).
- **PSR (Path Support Ratio) Score**: Cumulative confidence of all supporting paths (ranges from 0.0 to 1.0; higher is better).
- **p-value and Confidence Flag**: A hypergeometric test result. [HIGH CONFIDENCE] and [MID CONFIDENCE] indicate statistically significant supporting paths. [LOW CONFIDENCE] indicates weak statistical evidence.
- **SG-RAG Mechanism Paths**: Natural language representations of multi-hop paths from the query entity to the candidate, along with their individual confidence scores (product of sigmoided edge scores, range [0, 1]).

## Few-Shot Example 1: High Confidence ➜ Target Rank 1
### Example Query
Head entity: Compound::DB00959
Relation: treats
Task: Predict the most likely tail entity.

### Example Candidate Evidence Profiles
### Candidate 1: Disease::DOID:9074
- Initial Rank: #1
- GNN Rerank Certainty: 4.8915
- PSR Score: 0.9991
- p-value: 1.25e-06 [HIGH CONFIDENCE]
- EDGAR Statistics: 1/1 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  1. Compound::DB00959 --[treats]--> Disease::DOID:9074 [Score: 0.9991]

### Candidate 2: Disease::DOID:10763
- Initial Rank: #2
- GNN Rerank Certainty: 1.2062
- PSR Score: 0.0000
- p-value: 1.00e+00 [LOW CONFIDENCE]
- EDGAR Statistics: 0/0 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  (No direct paths found in subgraph)
  ⚠️ WARNING: Statistical evidence is weak (p=1.00e+00)

### Example Analysis and Final Answer:
Candidate 1 (Disease::DOID:9074) is ranked #1 by the GNN with a high certainty score of 4.8915. It is also strongly supported by a high-confidence direct therapeutic path (Compound::DB00959 --[treats]--> Disease::DOID:9074 [Score: 0.9991]) and passes the statistical failsafe check as HIGH CONFIDENCE (p = 1.25e-06). Candidate 2 has low structural confidence, no mechanism paths, and is flagged as LOW CONFIDENCE. Therefore, Disease::DOID:9074 is the most likely target.

Disease::DOID:9074

---

## Few-Shot Example 2: High Confidence ➜ Target NOT Rank 1 (Bypassing Distractor)
### Example Query
Head entity: Compound::DB00459
Relation: treats
Task: Predict the most likely tail entity.

### Example Candidate Evidence Profiles
### Candidate 1: Disease::DOID:2531
- Initial Rank: #1
- GNN Rerank Certainty: 1.1501
- PSR Score: 0.0000
- p-value: 1.00e+00 [LOW CONFIDENCE]
- EDGAR Statistics: 0/0 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  (No direct paths found in subgraph)
  ⚠️ WARNING: Statistical evidence is weak (p=1.00e+00)

### Candidate 2: Disease::DOID:8893
- Initial Rank: #2
- GNN Rerank Certainty: 3.9682
- PSR Score: 0.9969
- p-value: 1.25e-06 [HIGH CONFIDENCE]
- EDGAR Statistics: 1/1 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  1. Compound::DB00459 --[treats]--> Disease::DOID:8893 [Score: 0.9969]

### Example Analysis and Final Answer:
Although Candidate 1 (Disease::DOID:2531) is initially ranked higher (#1), it has a low GNN Certainty (1.1501), no supporting mechanism paths, and is flagged as LOW CONFIDENCE. Candidate 2 (Disease::DOID:8893, Rank #2) has a much higher GNN Certainty (3.9682) and is strongly supported by a high-scoring path (Compound::DB00459 --[treats]--> Disease::DOID:8893 [Score: 0.9969]), passing the statistical override as HIGH CONFIDENCE (p = 1.25e-06). Therefore, we bypass Candidate 1 and select Disease::DOID:8893.

Disease::DOID:8893

---

## Few-Shot Example 3: Mid Confidence ➜ Target Rank 1
### Example Query
Head entity: Compound::DB00997
Relation: treats
Task: Predict the most likely tail entity.

### Example Candidate Evidence Profiles
### Candidate 1: Disease::DOID:2531
- Initial Rank: #1
- GNN Rerank Certainty: 1.8504
- PSR Score: 0.1502
- p-value: 1.45e-02 [MID CONFIDENCE]
- EDGAR Statistics: 1/5 supporting paths (Subgraph complexity: 51 paths)
- SG-RAG Mechanism Paths:
  1. Compound::DB00997 --[binds]--> Gene::10320 --[associates]--> Disease::DOID:2531 [Score: 0.0152]

### Candidate 2: Disease::DOID:0060073
- Initial Rank: #2
- GNN Rerank Certainty: 0.9502
- PSR Score: 0.0000
- p-value: 1.00e+00 [LOW CONFIDENCE]
- EDGAR Statistics: 0/0 supporting paths (Subgraph complexity: 51 paths)
- SG-RAG Mechanism Paths:
  (No direct paths found in subgraph)
  ⚠️ WARNING: Statistical evidence is weak (p=1.00e+00)

### Example Analysis and Final Answer:
Candidate 1 (Disease::DOID:2531) is initially ranked #1 by the GNN and is supported by a multi-hop path with a statistically significant p-value of 1.45e-02, qualifying as MID CONFIDENCE. Candidate 2 has no supporting mechanism paths, low structural certainty, and is flagged as LOW CONFIDENCE. Therefore, Disease::DOID:2531 is the most likely target.

Disease::DOID:2531

---

## Few-Shot Example 4: Low Confidence ➜ Fallback to Structural Rank
### Example Query
Head entity: Compound::DB01204
Relation: treats
Task: Predict the most likely tail entity.

### Example Candidate Evidence Profiles
### Candidate 1: Disease::DOID:10283
- Initial Rank: #7
- GNN Rerank Certainty: 0.8972
- PSR Score: 0.0000
- p-value: 1.00e+00 [LOW CONFIDENCE]
- EDGAR Statistics: 0/0 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  (No direct paths found in subgraph)
  ⚠️ WARNING: Statistical evidence is weak (p=1.00e+00)

### Candidate 2: Disease::DOID:3393
- Initial Rank: #8
- GNN Rerank Certainty: 0.5401
- PSR Score: 0.0000
- p-value: 1.00e+00 [LOW CONFIDENCE]
- EDGAR Statistics: 0/0 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  (No direct paths found in subgraph)
  ⚠️ WARNING: Statistical evidence is weak (p=1.00e+00)

### Example Analysis and Final Answer:
Both candidates fail to show statistically significant path evidence and are flagged as LOW CONFIDENCE. Under this condition of weak path evidence, we fall back to the GNN's structural ranking. Candidate 1 (Disease::DOID:10283) holds a higher initial rank (#7 vs #8) and a higher GNN Certainty (0.8972 vs 0.5401). Thus, Candidate 1 is the most likely target.

Disease::DOID:10283

---

## Query
Head entity: {head_entity}
Relation: {relation}
Task: Predict the most likely tail entity.

## Candidate Evidence Profiles
{evidence_block}

## Instructions
Based on the evidence profiles above:
1. Analyze the retrieved SG-RAG mechanism paths and EDGAR statistical significance for each candidate.
2. Compare the GNN Rerank Certainty, PSR cumulative evidence, and EDGAR hypergeometric p-value provenance.
3. Prioritize candidates based on the confidence hierarchy: [HIGH CONFIDENCE] > [MID CONFIDENCE] > [LOW CONFIDENCE] when they are backed by corresponding mechanism paths.
4. Bypassing distractors: If the candidate at Rank #1 is flagged as [LOW CONFIDENCE] with no path support, look down the list for a candidate with high certainty and strong statistical path evidence.
5. Fallback: If ALL candidates are flagged as [LOW CONFIDENCE], fallback to the candidate with the highest GNN Rerank Certainty OR the candidate you are most confident in based on your pre-trained biomedical knowledge.

## Formatting Instructions
At the very end of your response, you MUST provide the predicted correct entity name directly. Do not include any tags, quotes, explanation, or markdown formatting.
Example: If the predicted entity is Disease::DOID:9074, output:
Disease::DOID:9074

## Your Final Answer:"""

# Default reranking prompt incorporating SG-RAG and EDGAR principles (Name-based few-shots)
RERANKER_PROMPT_TEMPLATE_NAME = """You are a biomedical expert evaluating candidates for knowledge graph completion using Subgraph Retrieval-Augmented Generation (SG-RAG) and the Explainable Enrichment-Driven Graph Reasoner (EDGAR).

You are given a query and a set of candidate entities. Each candidate has an Evidence Profile with the following metrics:
- **Initial Rank**: The candidate's rank prior to path retrieval (lower rank is better).
- **GNN Rerank Certainty**: The GNN's structural confidence score (higher values indicate stronger neighborhood alignment).
- **PSR (Path Support Ratio) Score**: Cumulative confidence of all supporting paths (ranges from 0.0 to 1.0; higher is better).
- **p-value and Confidence Flag**: A hypergeometric test result. [HIGH CONFIDENCE] and [MID CONFIDENCE] indicate statistically significant supporting paths. [LOW CONFIDENCE] indicates weak statistical evidence.
- **SG-RAG Mechanism Paths**: Natural language representations of multi-hop paths from the query entity to the candidate, along with their individual confidence scores (product of sigmoided edge scores, range [0, 1]).

## Few-Shot Example 1: High Confidence ➜ Target Rank 1
### Example Query
Head entity: Methylprednisolone
Relation: treats
Task: Predict the most likely tail entity.

### Example Candidate Evidence Profiles
### Candidate 1: systemic lupus erythematosus
- Initial Rank: #1
- GNN Rerank Certainty: 4.8915
- PSR Score: 0.9991
- p-value: 1.25e-06 [HIGH CONFIDENCE]
- EDGAR Statistics: 1/1 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  1. Methylprednisolone --[treats]--> systemic lupus erythematosus [Score: 0.9991]

### Candidate 2: lupus erythematosus
- Initial Rank: #2
- GNN Rerank Certainty: 1.2062
- PSR Score: 0.0000
- p-value: 1.00e+00 [LOW CONFIDENCE]
- EDGAR Statistics: 0/0 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  (No direct paths found in subgraph)
  ⚠️ WARNING: Statistical evidence is weak (p=1.00e+00)

### Example Analysis and Final Answer:
Candidate 1 (systemic lupus erythematosus) is ranked #1 by the GNN with a high certainty score of 4.8915. It is also strongly supported by a high-confidence direct therapeutic path (Methylprednisolone --[treats]--> systemic lupus erythematosus [Score: 0.9991]) and passes the statistical failsafe check as HIGH CONFIDENCE (p = 1.25e-06). Candidate 2 has low structural confidence, no mechanism paths, and is flagged as LOW CONFIDENCE. Therefore, systemic lupus erythematosus is the most likely target.

systemic lupus erythematosus

---

## Few-Shot Example 2: High Confidence ➜ Target NOT Rank 1 (Bypassing Distractor)
### Example Query
Head entity: Acitretin
Relation: treats
Task: Predict the most likely tail entity.

### Example Candidate Evidence Profiles
### Candidate 1: myeloid leukemia
- Initial Rank: #1
- GNN Rerank Certainty: 1.1501
- PSR Score: 0.0000
- p-value: 1.00e+00 [LOW CONFIDENCE]
- EDGAR Statistics: 0/0 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  (No direct paths found in subgraph)
  ⚠️ WARNING: Statistical evidence is weak (p=1.00e+00)

### Candidate 2: psoriasis
- Initial Rank: #2
- GNN Rerank Certainty: 3.9682
- PSR Score: 0.9969
- p-value: 1.25e-06 [HIGH CONFIDENCE]
- EDGAR Statistics: 1/1 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  1. Acitretin --[treats]--> psoriasis [Score: 0.9969]

### Example Analysis and Final Answer:
Although Candidate 1 (myeloid leukemia) is initially ranked higher (#1), it has a low GNN Certainty (1.1501), no supporting mechanism paths, and is flagged as LOW CONFIDENCE. Candidate 2 (psoriasis, Rank #2) has a much higher GNN Certainty (3.9682) and is strongly supported by a high-scoring path (Acitretin --[treats]--> psoriasis [Score: 0.9969]), passing the statistical override as HIGH CONFIDENCE (p = 1.25e-06). Therefore, we bypass Candidate 1 and select psoriasis.

psoriasis

---

## Few-Shot Example 3: Mid Confidence ➜ Target Rank 1
### Example Query
Head entity: Paroxetine
Relation: treats
Task: Predict the most likely tail entity.

### Example Candidate Evidence Profiles
### Candidate 1: depression
- Initial Rank: #1
- GNN Rerank Certainty: 1.8504
- PSR Score: 0.1502
- p-value: 1.45e-02 [MID CONFIDENCE]
- EDGAR Statistics: 1/5 supporting paths (Subgraph complexity: 51 paths)
- SG-RAG Mechanism Paths:
  1. Paroxetine --[binds]--> Gene::10320 --[associates]--> depression [Score: 0.0152]

### Candidate 2: migraine
- Initial Rank: #2
- GNN Rerank Certainty: 0.9502
- PSR Score: 0.0000
- p-value: 1.00e+00 [LOW CONFIDENCE]
- EDGAR Statistics: 0/0 supporting paths (Subgraph complexity: 51 paths)
- SG-RAG Mechanism Paths:
  (No direct paths found in subgraph)
  ⚠️ WARNING: Statistical evidence is weak (p=1.00e+00)

### Example Analysis and Final Answer:
Candidate 1 (depression) is initially ranked #1 by the GNN and is supported by a multi-hop path with a statistically significant p-value of 1.45e-02, qualifying as MID CONFIDENCE. Candidate 2 (migraine) has no supporting mechanism paths, low structural certainty, and is flagged as LOW CONFIDENCE. Therefore, depression is the most likely target.

depression

---

## Few-Shot Example 4: Low Confidence ➜ Fallback to Structural Rank
### Example Query
Head entity: Mitoxantrone
Relation: treats
Task: Predict the most likely tail entity.

### Example Candidate Evidence Profiles
### Candidate 1: prostate cancer
- Initial Rank: #7
- GNN Rerank Certainty: 0.8972
- PSR Score: 0.0000
- p-value: 1.00e+00 [LOW CONFIDENCE]
- EDGAR Statistics: 0/0 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  (No direct paths found in subgraph)
  ⚠️ WARNING: Statistical evidence is weak (p=1.00e+00)

### Candidate 2: coronary artery disease
- Initial Rank: #8
- GNN Rerank Certainty: 0.5401
- PSR Score: 0.0000
- p-value: 1.00e+00 [LOW CONFIDENCE]
- EDGAR Statistics: 0/0 supporting paths (Subgraph complexity: 5 paths)
- SG-RAG Mechanism Paths:
  (No direct paths found in subgraph)
  ⚠️ WARNING: Statistical evidence is weak (p=1.00e+00)

### Example Analysis and Final Answer:
Both candidates fail to show statistically significant path evidence and are flagged as LOW CONFIDENCE. Under this condition of weak path evidence, we fall back to the GNN's structural ranking. Candidate 1 (prostate cancer) holds a higher initial rank (#7 vs #8) and a higher GNN Certainty (0.8972 vs 0.5401). Thus, Candidate 1 is the most likely target.

prostate cancer

---

## Query
Head entity: {head_entity}
Relation: {relation}
Task: Predict the most likely tail entity.

## Candidate Evidence Profiles
{evidence_block}

## Instructions
Based on the evidence profiles above:
1. Analyze the retrieved SG-RAG mechanism paths and EDGAR statistical significance for each candidate.
2. Compare the GNN Rerank Certainty, PSR cumulative evidence, and EDGAR hypergeometric p-value provenance.
3. Prioritize candidates based on the confidence hierarchy: [HIGH CONFIDENCE] > [MID CONFIDENCE] > [LOW CONFIDENCE] when they are backed by corresponding mechanism paths.
4. Bypassing distractors: If the candidate at Rank #1 is flagged as [LOW CONFIDENCE] with no path support, look down the list for a candidate with high certainty and strong statistical path evidence.
5. Fallback: If ALL candidates are flagged as [LOW CONFIDENCE], fallback to the candidate with the highest GNN Rerank Certainty OR the candidate you are most confident in based on your pre-trained biomedical knowledge.

## Formatting Instructions
At the very end of your response, you MUST provide the predicted correct entity name directly. Do not include any tags, quotes, explanation, or markdown formatting.
Example: If the predicted entity is systemic lupus erythematosus, output:
systemic lupus erythematosus

## Your Final Answer:"""

RERANKER_PROMPT_TEMPLATE = RERANKER_PROMPT_TEMPLATE_ID


class LLMReranker:
    """LLM-based reranker using in-context learning.

    Parameters
    ----------
    model_name:
        HuggingFace model ID for the reranker LLM.
    temperature:
        Sampling temperature (lower = more deterministic).
    max_new_tokens:
        Maximum tokens for the LLM response.
    device:
        Device for inference.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device

        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load the reranker LLM."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"Loading Reranker LLM: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()

    def build_evidence_block(self, profiles: list[EvidenceProfile]) -> str:
        """Format evidence profiles into a structured text block.

        Parameters
        ----------
        profiles:
            List of EvidenceProfile objects (one per candidate).

        Returns
        -------
        str
            Formatted evidence block for prompt injection.
        """
        blocks: list[str] = []

        for i, p in enumerate(profiles, 1):
            prov = getattr(p.failsafe, "provenance", {}) or {}
            sup_paths = prov.get("supporting_paths", 0)
            paths_to_cand = prov.get("paths_to_candidate", 0)
            tot_subg = prov.get("total_subgraph_paths", 0)

            lines = [
                f"### Candidate {i}: {p.candidate_name}",
                f"- Initial Rank: #{p.gnn_rank}",
            ]
            if p.rerank_certainty is not None:
                lines.append(f"- GNN Rerank Certainty: {p.rerank_certainty:.4f}")
            lines.extend([
                f"- PSR Score: {p.psr_score:.4f}",
                f"- p-value: {p.failsafe.p_value:.2e} [{p.failsafe.confidence_flag}]",
                f"- EDGAR Statistics: {sup_paths}/{paths_to_cand} supporting paths (Subgraph complexity: {tot_subg} paths)",
                f"- SG-RAG Mechanism Paths:",
            ])
            
            if not p.mechanism_texts:
                lines.append("  (No direct paths found in subgraph)")
            else:
                for j, text in enumerate(p.mechanism_texts[:5], 1):
                    score_str = ""
                    if p.path_scores and j - 1 < len(p.path_scores):
                        score_str = f" [Score: {p.path_scores[j-1]:.4f}]"
                    lines.append(f"  {j}. {text}{score_str}")

            if not p.failsafe.is_significant:
                lines.append(
                    f"  ⚠️ WARNING: Statistical evidence is weak (p={p.failsafe.p_value:.2e})"
                )

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def build_prompt(self, query_entity: str, relation: str, evidence_profiles: list[EvidenceProfile]) -> str:
        """Construct the raw prompt string from the evidence profiles."""
        evidence_block = self.build_evidence_block(evidence_profiles)

        # Selection logic based on presence of database ID indicators in query_entity or candidate names
        # e.g., '::', 'DOID:', 'MESH:', 'HP:', 'Gene::', 'Compound::'
        is_id = False
        if "::" in query_entity or ":" in query_entity:
            is_id = True
        elif evidence_profiles and any("::" in p.candidate_name or ":" in p.candidate_name for p in evidence_profiles):
            is_id = True

        template = RERANKER_PROMPT_TEMPLATE_ID if is_id else RERANKER_PROMPT_TEMPLATE_NAME

        prompt = template.format(
            head_entity=query_entity,
            relation=relation,
            evidence_block=evidence_block,
        )
        return prompt

    def rerank(
        self,
        query_entity: str,
        relation: str,
        evidence_profiles: list[EvidenceProfile] = None,
        prebuilt_prompt: str = None,
    ) -> str:
        """Run LLM reranking with evidence profiles or prebuilt prompt."""
        self._load_model()
        import torch

        if prebuilt_prompt:
            prompt = prebuilt_prompt
        elif evidence_profiles:
            prompt = self.build_prompt(query_entity, relation, evidence_profiles)
        else:
            raise ValueError("Must provide either evidence_profiles or prebuilt_prompt")

        # Chat template style if applicable, else raw prompt
        if "Instruct" in self.model_name:
            messages = [
                {"role": "system", "content": "You are a biomedical expert evaluating candidates for knowledge graph completion."},
                {"role": "user", "content": prompt}
            ]
            input_text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer(input_text, return_tensors="pt").to(self.device)
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.95,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        logger.info(
            f"LLM Reranker completed: {len(evidence_profiles) if evidence_profiles else 'prebuilt'} candidates evaluated for ({query_entity}, {relation}, ?)"
        )
        return response
