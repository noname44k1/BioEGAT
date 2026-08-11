"""
app.py
~~~~~~
Main entrypoint for the BioEGAT Gradio demo.
Biomedical Link Prediction via Edge-featured GAT and LLM-based Mechanistic Reasoning.
"""

import os
import sys
import gradio as gr
import pandas as pd
from pathlib import Path

# Add project root to sys.path to allow importing other modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from demo.pipeline import DemoPipeline
from demo.visualizer import build_candidate_path_html

# Initialize pipeline with DemoDataRegistry
DATA_PATH = Path(__file__).parent / "data"
pipeline = DemoPipeline(DATA_PATH)
AVAILABLE_DATASETS = pipeline.registry.get_available_datasets()


DIR_LABEL = {
    "predicted_tail": "Predict tail  (head + relation → ?)",
    "predicted_head": "Predict head  (? → relation + tail)",
}


def _rel_of(item):
    return item.get("relation") or (item.get("triple") or ["", "", ""])[1]


def get_query_entities(dataset_id: str):
    data = pipeline.registry.load_dataset(dataset_id)
    out = []
    for it in data:
        q = it.get("query_entity", "")
        if q and q not in out:
            out.append(q)
    return out


def get_directions(dataset_id: str, query: str):
    data = pipeline.registry.load_dataset(dataset_id)
    dirs = []
    for it in data:
        if it.get("query_entity") == query:
            d = it.get("type")
            if d and d not in dirs:
                dirs.append(d)
    return [(DIR_LABEL.get(d, d), d) for d in dirs]


def get_relations(dataset_id: str, query: str, direction: str):
    data = pipeline.registry.load_dataset(dataset_id)
    rels = []
    for it in data:
        if it.get("query_entity") == query and it.get("type") == direction:
            r = _rel_of(it)
            if r and r not in rels:
                rels.append(r)
    return rels


def resolve_index(dataset_id: str, query: str, direction: str, relation: str) -> int:
    data = pipeline.registry.load_dataset(dataset_id)
    for i, it in enumerate(data):
        if it.get("query_entity") == query and it.get("type") == direction and _rel_of(it) == relation:
            return i
    for i, it in enumerate(data):           # fallback: match query + direction
        if it.get("query_entity") == query and it.get("type") == direction:
            return i
    return -1


def on_dataset_change(dataset_id: str):
    ents = get_query_entities(dataset_id)
    first = ents[0] if ents else None
    dirs = get_directions(dataset_id, first) if first else []
    dval = dirs[0][1] if dirs else None
    rels = get_relations(dataset_id, first, dval) if dval else []
    return (
        gr.update(choices=ents, value=first),
        gr.update(choices=dirs, value=dval),
        gr.update(choices=rels, value=rels[0] if rels else None),
    )


def on_query_change(dataset_id: str, query: str):
    dirs = get_directions(dataset_id, query)
    dval = dirs[0][1] if dirs else None
    rels = get_relations(dataset_id, query, dval) if dval else []
    return (
        gr.update(choices=dirs, value=dval),
        gr.update(choices=rels, value=rels[0] if rels else None),
    )


def on_direction_change(dataset_id: str, query: str, direction: str):
    rels = get_relations(dataset_id, query, direction)
    return gr.update(choices=rels, value=rels[0] if rels else None)


def _ev_chip(p: dict) -> str:
    if p.get("evidence_id"):
        return (f'<a class="ev" href="{p.get("evidence_url", "#")}" target="_blank" '
                f'rel="noopener">{p.get("evidence_source", "")} · {p.get("evidence_id")}</a>')
    return ""


def render_premium_html(item: dict) -> tuple[str, str]:
    """Hero = LLM prediction; list = the remaining reranked candidates.
    Statistics (PSR, supporting paths) are kept as explanation; no HIGH/MED/LOW tiers."""
    if not item:
        return "<div>No data</div>", "<div>No data</div>"

    gold = item.get("target", "")
    pred = item.get("pred", "") or item.get("rank_entities", [""])[0]
    rank_entities = item.get("rank_entities", [])
    by_name = {p.get("candidate_name"): p for p in item.get("evidence_profiles", [])}

    # ── Hero: the LLM's predicted entity ──
    hp = by_name.get(pred, {})
    hero_paths = hp.get("mechanism_texts", [])
    hero_psr = hp.get("psr_score")
    gold_badge = ' <span class="gold-badge">✓ ground truth</span>' if pred == gold else ""
    psr_stat = f'<span><strong>PSR:</strong> {hero_psr:.3f}</span>' if hero_psr is not None else ""
    hero_html = f"""
    <div class="hero-card">
        <div class="hero-header">
            <span class="hero-rank">🧠 LLM prediction</span>
            {_ev_chip(hp)}
        </div>
        <h2 class="hero-title">{pred}{gold_badge}</h2>
        <div class="hero-stats">
            {psr_stat}
            <span><strong>Supporting paths:</strong> {len(hero_paths)}</span>
        </div>
        <div class="hero-paths">
            <h4>Top supporting paths</h4>
            <ul>
    """
    for p in hero_paths[:3]:
        hero_html += f"<li><code>{p}</code></li>"
    if not hero_paths:
        hero_html += "<li><em>No direct path in the retrieved subgraph</em></li>"
    hero_html += "</ul></div></div>"

    # ── List: the other 19 reranked candidates (excluding the LLM pick) ──
    others = [c for c in rank_entities if c != pred]
    list_html = '<div class="ranked-list"><div class="list-title">Reranked candidates (BioEGAT order)</div>'
    for i, name in enumerate(others, start=1):
        p = by_name.get(name, {})
        top_path = (p.get("mechanism_texts") or ["No direct path in subgraph"])[0]
        psr = p.get("psr_score")
        psr_html = f'<span class="item-psr">PSR {psr:.3f}</span>' if psr is not None else ""
        gold_mark = ' <span class="gold-badge">✓</span>' if name == gold else ""
        list_html += f"""
        <div class="ranked-item">
            <div class="item-header">
                <span class="item-rank">#{i}</span>
                <span class="item-name">{name}{gold_mark}</span>
                {_ev_chip(p)}
                {psr_html}
            </div>
            <div class="item-path"><code>{top_path}</code></div>
        </div>
        """
    list_html += '</div>'

    return hero_html, list_html


def _empty_outputs(msg):
    return (
        msg, "", "", "N/A", "N/A", "N/A", "N/A", "", "",
        "<p style='text-align:center; padding:40px; color:#64748B;'>No subgraph rendering available.</p>",
        gr.update(choices=[]),
        "<p style='text-align:center; color:#64748B;'>Select a candidate to show connection paths.</p>",
    )


def run_prediction_flow(dataset_id, query, direction, relation, run_live_llm, model_name):
    """Resolve the (query, head/tail, relation) selection to a precomputed record and render it."""
    idx = resolve_index(dataset_id, query, direction, relation)
    if idx < 0:
        return _empty_outputs("⚠️ No precomputed result for that query / relation combination.")

    res = pipeline.run(dataset_id, idx, run_live_llm=run_live_llm, model_name=model_name)
    if "error" in res:
        return _empty_outputs(f"❌ Error: {res['error']}")

    item = pipeline.registry.get_by_index(dataset_id, idx)
    hero_html, ranked_html = render_premium_html(item)

    # candidate-path dropdown: LLM prediction first, then the reranked rest
    pred = item.get("pred", "")
    candidates = res.get("rank_entities", [])
    ordered = ([pred] if pred and pred in candidates else []) + [c for c in candidates if c != pred]
    if pred and pred not in candidates:
        ordered = [pred] + candidates
    path_dropdown_update = gr.update(choices=ordered, value=ordered[0] if ordered else None)

    subgraph_triples = res.get("subgraph_triples", [])
    first_candidate_path = ""
    if ordered:
        first_candidate_path = build_candidate_path_html(
            subgraph_triples=subgraph_triples,
            query_entity=res["query_entity"],
            candidate=ordered[0],
        )

    dir_word = "tail" if item.get("type") == "predicted_tail" else "head"
    status = (f"✅ ({item.get('query_entity')}, {relation}, ?) — predict {dir_word}. "
              f"LLM prediction: {pred or 'N/A'} · ground truth: {item.get('target', 'N/A')}")
    target = res.get("target")
    
    kge_rank = item.get("rank", "N/A")
    if isinstance(kge_rank, float):
        kge_rank = int(kge_rank)
        
    gnn_rank = "N/A (not in top 20)"
    if target in candidates:
        gnn_rank = candidates.index(target) + 1

    return (
        status,
        hero_html,
        ranked_html,
        str(kge_rank),
        str(gnn_rank),
        pred or "N/A",                 # decision box = LLM's predicted entity
        target,
        res["llm_prompt"],
        res["llm_response"],
        res["subgraph_html"],
        path_dropdown_update,
        first_candidate_path,
    )


def update_candidate_path(dataset_id, query, direction, relation, candidate):
    """Callback to update candidate path visualization."""
    idx = resolve_index(dataset_id, query, direction, relation)
    if idx < 0 or not candidate:
        return "<p style='text-align:center; color:#64748B;'>Select a candidate and run prediction first.</p>"

    item = pipeline.registry.get_by_index(dataset_id, idx)
    if not item:
        return "<p style='text-align:center; color:#64748B;'>Example not found.</p>"

    return build_candidate_path_html(
        subgraph_triples=item.get("subgraph", []),
        query_entity=item.get("query_entity", ""),
        candidate=candidate,
    )


# Custom CSS for light-mode premium design aesthetics
CUSTOM_CSS = """
body, .gradio-container {
    background-color: #F8FAFC !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.premium-title {
    text-align: center;
    margin-bottom: 2rem;
}
.premium-title h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1E293B;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.premium-title p {
    font-size: 1.1rem;
    color: #475569;
}
.status-bar {
    background-color: #F1F5F9;
    border-left: 5px solid #4F46E5;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    font-weight: 500;
    color: #334155;
    margin-bottom: 1.5rem;
}
/* Hero Card */
.hero-card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
    border: 1px solid #E2E8F0;
    margin-bottom: 1.5rem;
    border-top: 4px solid #4F46E5;
}
.hero-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.hero-rank {
    font-weight: 800;
    color: #4F46E5;
    font-size: 1.1rem;
}
.hero-tier {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.hero-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0.5rem 0;
}
.hero-stats {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1rem;
    color: #64748B;
    font-size: 0.95rem;
}
.hero-paths h4 {
    margin: 0 0 0.5rem 0;
    color: #334155;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.hero-paths ul {
    list-style-type: none;
    padding: 0;
    margin: 0;
}
.hero-paths li {
    background: #F8FAFC;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    margin-bottom: 0.5rem;
    border: 1px solid #F1F5F9;
    font-size: 0.85rem;
    color: #475569;
}
/* Ranked List */
.ranked-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.ranked-item {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem;
    transition: all 0.2s ease;
}
.ranked-item:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    border-color: #CBD5E1;
}
.item-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.item-rank {
    font-weight: 700;
    color: #94A3B8;
    min-width: 2rem;
}
.item-name {
    font-weight: 600;
    color: #1E293B;
    flex-grow: 1;
}
.item-tier {
    font-size: 0.75rem;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    border: 1px solid;
    font-weight: 600;
}
.item-psr {
    font-size: 0.85rem;
    color: #64748B;
}
.item-path {
    background: #F8FAFC;
    padding: 0.4rem 0.6rem;
    border-radius: 4px;
    font-size: 0.8rem;
    color: #475569;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.list-title { font-weight: 700; color: #334155; margin: 0.25rem 0 0.6rem; font-size: 0.95rem; }
.gold-badge {
    background: #FEF3C7; color: #92400E; font-size: 0.7rem; font-weight: 700;
    padding: 0.1rem 0.45rem; border-radius: 9999px; vertical-align: middle;
}
.ev {
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.72rem; font-weight: 600;
    padding: 0.12rem 0.5rem; border-radius: 4px; text-decoration: none;
    background: #E0E7FF; color: #3730A3; border: 1px solid #C7D2FE;
}
.ev:hover { text-decoration: underline; }
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate"), css=CUSTOM_CSS) as demo:
    # Header section
    gr.HTML("""
        <div class="premium-title">
            <h1>BioEGAT 🧬</h1>
            <p>Biomedical Link Prediction via Edge-featured GAT and LLM-based Mechanistic Reasoning</p>
        </div>
    """)

    # Main layout
    with gr.Row():
        # LEFT COLUMN: Configuration
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configuration")
            
            dataset_dd = gr.Dropdown(
                choices=[(lbl, k) for k, lbl in AVAILABLE_DATASETS],
                label="Dataset",
                value=AVAILABLE_DATASETS[0][0] if AVAILABLE_DATASETS else None,
                interactive=True
            )
            
            query_entity_dd = gr.Dropdown(
                choices=[],
                label="Query entity",
                info="The head entity to complete from",
                interactive=True,
            )
            direction_dd = gr.Dropdown(
                choices=[],
                label="Prediction direction",
                info="Predict the head or the tail of the triple",
                interactive=True,
            )
            relation_dd = gr.Dropdown(
                choices=[],
                label="Relation",
                interactive=True,
            )
            
            with gr.Accordion("LLM Settings", open=False):
                enable_live_api = gr.Checkbox(
                    label="Enable Live Gemini API Call",
                    value=pipeline.is_api_available(),
                    info="Uncheck to display pre-computed LLM predictions instantly"
                )
                model_name_dd = gr.Dropdown(
                    choices=["gemini-2.5-flash", "gemini-2.5-pro"],
                    label="Gemini Model",
                    value="gemini-2.5-flash"
                )
            
            predict_btn = gr.Button("🔍 Run Pipeline & Predict Link", variant="primary")
            
            status_output = gr.Markdown("Ready to run. Select a dataset and query, then click Predict.", elem_classes="status-bar")
            
            gr.Markdown("### Pipeline Tracking")
            with gr.Row():
                kge_rank_box = gr.Textbox(
                    label="Rank from KGE",
                    placeholder="-",
                    interactive=False,
                )
                gnn_rank_box = gr.Textbox(
                    label="Rank from Rerank",
                    placeholder="-",
                    interactive=False,
                )
            with gr.Row():
                decision_box = gr.Textbox(
                    label="LLM Prediction",
                    placeholder="-",
                    interactive=False,
                )
                target_box = gr.Textbox(
                    label="Ground Truth",
                    placeholder="-",
                    interactive=False,
                )

        # RIGHT COLUMN: Results Tabs
        with gr.Column(scale=3):
            with gr.Tabs() as tabs:
                # Tab 1: Evidence & Ranked Candidates
                with gr.Tab("🏆 Ranked Candidates & Evidence", id=1):
                    hero_card_viz = gr.HTML(value="<p style='color:#64748B;'>Select a query and run pipeline to see top candidate.</p>")
                    ranked_list_viz = gr.HTML(value="")
                    
                # Tab 2: Subgraph Visualization
                with gr.Tab("🕸️ Neighborhood Subgraph", id=2):
                    gr.Markdown("Interactive view of the local network neighborhood. **Query entity** is highlighted in **Gold**.")
                    
                    subgraph_viz = gr.HTML(
                        value="<p style='text-align:center; padding:100px; color:#64748B;'>Select a query and run pipeline to visualize.</p>"
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Highlight Candidate Path")
                            path_candidate_dd = gr.Dropdown(
                                choices=[],
                                label="Select Candidate to Isolate",
                                info="Extract and visualize only paths connecting Query → Candidate"
                            )
                        with gr.Column(scale=2):
                            isolated_path_viz = gr.HTML(
                                value="<p style='text-align:center; color:#64748B; padding:40px;'>No isolated path rendering active.</p>"
                            )

                # Tab 3: LLM Reasoning
                with gr.Tab("🧠 API Call's Results", id=3):
                    gr.Markdown("Inspect the exact prompt constructed by the synthesizer and the LLM's full reasoning trace.")
                    with gr.Row():
                        prompt_inspector_box = gr.Textbox(
                            label="Synthesized Reranker Input Prompt",
                            lines=20,
                            interactive=False
                        )
                        llm_reasoning_box = gr.Textbox(
                            label="LLM Final Output",
                            lines=20,
                            interactive=False
                        )

    # Event Wireups
    
    # Cascade: dataset -> query entity -> direction (head/tail) -> relation
    dataset_dd.change(
        fn=on_dataset_change,
        inputs=[dataset_dd],
        outputs=[query_entity_dd, direction_dd, relation_dd],
    )
    query_entity_dd.change(
        fn=on_query_change,
        inputs=[dataset_dd, query_entity_dd],
        outputs=[direction_dd, relation_dd],
    )
    direction_dd.change(
        fn=on_direction_change,
        inputs=[dataset_dd, query_entity_dd, direction_dd],
        outputs=[relation_dd],
    )
    demo.load(
        fn=on_dataset_change,
        inputs=[dataset_dd],
        outputs=[query_entity_dd, direction_dd, relation_dd],
    )

    # Main prediction action
    predict_btn.click(
        fn=run_prediction_flow,
        inputs=[dataset_dd, query_entity_dd, direction_dd, relation_dd, enable_live_api, model_name_dd],
        outputs=[
            status_output,
            hero_card_viz,
            ranked_list_viz,
            kge_rank_box,
            gnn_rank_box,
            decision_box,
            target_box,
            prompt_inspector_box,
            llm_reasoning_box,
            subgraph_viz,
            path_candidate_dd,
            isolated_path_viz,
        ],
    )

    # Dynamic candidate path highlights
    path_candidate_dd.change(
        fn=update_candidate_path,
        inputs=[dataset_dd, query_entity_dd, direction_dd, relation_dd, path_candidate_dd],
        outputs=isolated_path_viz,
    )

if __name__ == "__main__":
    demo.launch()
