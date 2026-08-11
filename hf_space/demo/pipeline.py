"""
pipeline.py
~~~~~~~~~~~
Orchestrates the pipeline backend for the BioEGAT Gradio demo.
Links pre-computed data, synthesizer modules, and the Gemini API.
"""

import os
import re
import pickle
import logging
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

from demo.precomputed import DemoDataRegistry
from demo.visualizer import build_subgraph_html, build_candidate_path_html, build_candidate_table
from synthesizer.llm_reranker import LLMReranker, EvidenceProfile

logger = logging.getLogger(__name__)

# Load local environment variables
env_path = Path("notebook/.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
elif Path(".env").exists():
    load_dotenv(override=True)


class DemoPipeline:
    def __init__(self, data_dir: str | Path | None = None):
        self.registry = DemoDataRegistry(data_dir)
        self.gemini_client = None
        self.id2relation = None
        self.entity2id = None
        
        # Initialize Gemini API client if API key is set
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                self.gemini_client = genai.Client(api_key=api_key)
                logger.info("Initialized Gemini Client successfully.")
            except Exception as e:
                logger.error(f"Error initializing Gemini Client: {e}")
        else:
            logger.warning("GEMINI_API_KEY not found in environment variables.")

        # Try to load dictionaries for metadata fallback if present
        try:
            dict_path = Path("dataset/primekg/primekg_ncrl/id2relation.pkl")
            if dict_path.exists():
                with open(dict_path, "rb") as f:
                    self.id2relation = pickle.load(f)
            
            ent_path = Path("dataset/primekg/entity2id.pkl")
            if ent_path.exists():
                with open(ent_path, "rb") as f:
                    self.entity2id = pickle.load(f)
        except Exception as e:
            logger.debug(f"Failed to load supplementary dictionary maps: {e}")

    def is_api_available(self) -> bool:
        return self.gemini_client is not None

    def query_llm(self, prompt: str, model_name: str = "gemini-2.5-flash") -> str:
        """Query Gemini API for prediction and reasoning."""
        if not self.gemini_client:
            return "⚠️ Gemini API client is not configured. Please set GEMINI_API_KEY as a secret or environment variable."

        config = types.GenerateContentConfig(
            candidate_count=1,
            temperature=0.1,
        )

        try:
            response = self.gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            return response.text.strip()
        except APIError as e:
            return f"❌ Gemini API Error: {e.message}"
        except Exception as e:
            return f"❌ Unexpected Error: {str(e)}"

    def run(self, dataset_id: str, example_idx: int, run_live_llm: bool = False, model_name: str = "gemini-2.5-flash") -> dict:
        """
        Run pipeline step-by-step for a given dataset and example index.
        """
        item = self.registry.get_by_index(dataset_id, example_idx)
        if not item:
            return {"error": "Invalid dataset or example index"}

        query_ent = item.get("query_entity", "")
        triple = item.get("triple", ["", "", ""])
        rel = triple[1] if len(triple) > 1 else ""
        target = triple[2] if item.get("type") == "predicted_tail" else triple[0]

        # Extract values
        subgraph = item.get("subgraph", [])
        rank_entities = item.get("rank_entities", [])
        rerank_scores = item.get("rerank_scores", [])
        evidence_profiles = item.get("evidence_profiles")

        # 1. Synthesize Prompt if not pre-computed or if we need to regenerate
        llm_prompt = item.get("llm_prompt")
        if not llm_prompt and evidence_profiles:
            # Build prompt dynamically using loaded LLMReranker
            try:
                profiles = []
                for p in evidence_profiles:
                    profiles.append(EvidenceProfile(
                        candidate_name=p["candidate_name"],
                        gnn_rank=p["gnn_rank"],
                        mechanism_texts=p["mechanism_texts"],
                        psr_score=p["psr_score"],
                        failsafe=p["failsafe"],
                        candidate_id=p.get("candidate_id", -1),
                        path_scores=p.get("path_scores", []),
                        rerank_certainty=p.get("rerank_certainty", 0.0)
                    ))
                reranker = LLMReranker()
                llm_prompt = reranker.build_prompt(
                    query_entity=query_ent,
                    relation=rel,
                    evidence_profiles=profiles
                )
            except Exception as e:
                logger.error(f"Failed to dynamically construct LLM prompt: {e}")

        # 2. Get LLM prediction
        if run_live_llm and llm_prompt and self.is_api_available():
            status_msg = "Running live Gemini inference..."
            llm_response = self.query_llm(llm_prompt, model_name=model_name)
        else:
            status_msg = "Displaying pre-computed LLM decision (Gemini API skipped)."
            llm_response = item.get("pred", "No pre-computed LLM output available. Enable live Gemini API to predict.")

        # Extract decision tag from response
        decision = "N/A"
        match = re.search(r"<decision>\s*(.*?)\s*</decision>", llm_response, re.IGNORECASE | re.DOTALL)
        if match:
            decision = match.group(1).strip()

        # Build candidate dataframe for Gradio
        df = build_candidate_table(rank_entities, rerank_scores, evidence_profiles)

        # Build interactive subgraph HTML
        subgraph_html = build_subgraph_html(
            subgraph_triples=subgraph,
            query_entity=query_ent,
            candidates=rank_entities,
            rerank_scores=rerank_scores,
        )

        return {
            "query_entity": query_ent,
            "relation": rel,
            "prediction_type": item.get("type", "predicted_tail"),
            "target": target,
            "status": status_msg,
            "candidates_df": df,
            "llm_prompt": llm_prompt or "No prompt synthesized.",
            "llm_response": llm_response,
            "parsed_decision": decision,
            "subgraph_html": subgraph_html,
            "subgraph_triples": subgraph,
            "rank_entities": rank_entities,
        }
