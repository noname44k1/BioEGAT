---
title: BioEGAT - Biomedical Link Prediction
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.19.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# BioEGAT: Biomedical Link Prediction via Edge-featured GAT and LLM-based Mechanistic Reasoning

This is the interactive Gradio demo interface for the **BioEGAT** framework.

## Features
- **Gallery**: Select and load pre-computed case studies from PrimeKG test splits.
- **Link Prediction**: Generate ranks and certainty scores using KGE and GAT.
- **Interactive Graph Visualization**: Explore the local subgraph and paths from query entity to candidate entities.
- **Explainable Reasoning**: Review the natural language prompts and the LLM's step-by-step reasoning.
