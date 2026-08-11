from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import GenerationConfig

__all__ = ["BioEGAT"]


class BioEGAT(nn.Module):
    def __init__(self, tokenizer, llm_model, graph_model):
        super().__init__()
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.graph_model = graph_model
        self.query_token_id = self.tokenizer.convert_tokens_to_ids(['[QUERY]'])[0]
        self.entity_token_id = self.tokenizer.convert_tokens_to_ids(['[ENTITY]'])[0]


    def _replace_placeholders(self, input_ids: torch.Tensor, query_ids: torch.Tensor, entity_ids: torch.Tensor, subgraph=None):
        # BioEGAT enhancer: frozen KBGAT + InteractE backbone provides structure-aware vectors;
        # only the final projection (adapter.fc2 -> LLM hidden size) is trainable.
        query_embeds, entity_embeds = self.graph_model(query_ids, entity_ids, subgraph)

        clean_ids = input_ids.clone()
        clean_ids[clean_ids == self.query_token_id] = self.tokenizer.pad_token_id
        clean_ids[clean_ids == self.entity_token_id] = self.tokenizer.pad_token_id
        # get_input_embeddings() resolves the token embedding table across PEFT/quant wrapping
        inputs_embeds = self.llm_model.get_input_embeddings()(clean_ids).clone()

        query_pos = torch.nonzero(input_ids == self.query_token_id)
        entity_pos = torch.nonzero(input_ids == self.entity_token_id)
        # cast graph (fp32) outputs to the LLM embedding dtype (bf16/fp16 under QLoRA)
        inputs_embeds[query_pos[:, 0], query_pos[:, 1]] = query_embeds.to(inputs_embeds.dtype)
        inputs_embeds[entity_pos[:, 0], entity_pos[:, 1]] = entity_embeds.to(inputs_embeds.dtype)
        return inputs_embeds


    def forward(self,input_ids, attention_mask, labels, query_ids, entity_ids, subgraph):
        inputs_embeds = self._replace_placeholders(input_ids, query_ids, entity_ids, subgraph)

        return self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    def train(self, mode: bool = True):
        # Keep the frozen BioEGAT backbone (entity/relation embeddings, GAT, InteractE conv/fc1,
        # BatchNorms) in eval mode so its BN running stats stay fixed and dropout is disabled;
        # only adapter.fc2 + the LoRA adapters are trained. Gradients still flow to fc2 in eval mode.
        super().train(mode)
        self.graph_model.eval()
        return self

    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.llm_model.save_pretrained(save_dir)
        torch.save(self.graph_model.state_dict(), save_dir / "graph_model.bin")


    @torch.no_grad()
    def generate(
        self, input_ids, query_ids, entity_ids, subgraph = None, generation_config: GenerationConfig = None):
        inputs_embeds = self._replace_placeholders(input_ids, query_ids, entity_ids, subgraph)
        
        if generation_config is None:
            generation_config = GenerationConfig()
        
        return self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            generation_config=generation_config,
        )    