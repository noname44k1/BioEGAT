"""
visualizer.py
~~~~~~~~~~~~~
Graph visualization utilities for the BioEGAT demo.
Uses NetworkX for graph construction and PyVis for interactive HTML rendering.
"""

import html as html_module
import networkx as nx
import pandas as pd

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False


# ── Color palette (light-mode friendly) ──────────────────────────────────────
QUERY_COLOR = "#F59E0B"       # Amber – query entity
CANDIDATE_HIGH = "#10B981"    # Emerald – high certainty
CANDIDATE_MID = "#3B82F6"     # Blue – mid certainty
CANDIDATE_LOW = "#EF4444"     # Red – low certainty
INTERMEDIATE_COLOR = "#94A3B8" # Slate – intermediate entities
EDGE_COLOR = "#CBD5E1"         # Light slate – edges
HIGHLIGHT_EDGE = "#6366F1"     # Indigo – highlighted path edges

RELATION_COLORS = {
    "indication": "#6366F1",
    "contraindication": "#EF4444",
    "off-label use": "#F59E0B",
    "disease_protein": "#10B981",
    "drug_protein": "#3B82F6",
}


def _certainty_color(score: float) -> str:
    """Map a certainty score [0, 1] to a color."""
    if score >= 0.10:
        return CANDIDATE_HIGH
    elif score >= 0.05:
        return CANDIDATE_MID
    else:
        return CANDIDATE_LOW


def build_subgraph_html(
    subgraph_triples: list,
    query_entity: str,
    candidates: list[str],
    rerank_scores: list[float] | None = None,
    layout: str = "force",
    height: str = "520px",
    width: str = "100%",
) -> str:
    """
    Build an interactive PyVis graph from subgraph triples.

    Parameters
    ----------
    subgraph_triples : list of [h, r, t]
        Named triples from the subgraph.
    query_entity : str
        The query entity (highlighted as central node).
    candidates : list of str
        Candidate entity names.
    rerank_scores : list of float, optional
        Softmax-normalized GNN certainty scores, aligned with candidates.
    layout : str
        Layout algorithm: "force", "hierarchical", "circular".
    height, width : str
        CSS dimensions for the visualization.

    Returns
    -------
    str
        HTML string wrapped in an iframe for Gradio gr.HTML.
    """
    if not HAS_PYVIS:
        return "<p style='color:red'>PyVis is not installed. Run: pip install pyvis</p>"

    if not subgraph_triples:
        return "<p style='color:#64748B; text-align:center; padding:40px;'>No subgraph data available for this query.</p>"

    score_map = {}
    if rerank_scores and candidates:
        for name, score in zip(candidates, rerank_scores):
            score_map[name] = score

    # Build NetworkX graph
    G = nx.DiGraph()
    candidate_set = set(candidates) if candidates else set()

    for triple in subgraph_triples:
        if len(triple) < 3:
            continue
        h, r, t = str(triple[0]), str(triple[1]), str(triple[2])
        G.add_edge(h, t, relation=r)

    # Build PyVis network
    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        cdn_resources="remote",
        bgcolor="#FFFFFF",
        font_color="#1E293B",
    )

    # Configure physics
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -3000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09
            },
            "stabilization": {
                "enabled": true,
                "iterations": 200
            }
        },
        "edges": {
            "smooth": {
                "type": "curvedCW",
                "roundness": 0.15
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.6
                }
            },
            "font": {
                "size": 10,
                "color": "#64748B",
                "strokeWidth": 2,
                "strokeColor": "#FFFFFF"
            }
        },
        "nodes": {
            "font": {
                "size": 12,
                "face": "Inter, sans-serif"
            },
            "borderWidth": 2,
            "borderWidthSelected": 3,
            "shadow": {
                "enabled": true,
                "size": 5,
                "x": 2,
                "y": 2
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": true
        }
    }
    """)

    # Add nodes
    for node in G.nodes():
        if node == query_entity:
            net.add_node(
                node,
                label=_truncate(node, 25),
                title=f"🔍 Query: {node}",
                color=QUERY_COLOR,
                size=35,
                shape="diamond",
                font={"size": 14, "color": "#92400E", "bold": True},
            )
        elif node in candidate_set:
            score = score_map.get(node, 0.0)
            rank = candidates.index(node) + 1 if node in candidates else "?"
            net.add_node(
                node,
                label=f"#{rank} {_truncate(node, 20)}",
                title=f"Candidate #{rank}: {node}\nCertainty: {score:.4f}",
                color=_certainty_color(score),
                size=22 + score * 80,
                shape="dot",
                font={"size": 12},
            )
        else:
            # Simple heuristic: look at incident edges to guess type
            node_type = "unknown"
            for u, v, data in G.edges(node, data=True):
                r = data.get("relation", "").lower()
                if "protein" in r or "gene" in r or "enzym" in r:
                    node_type = "protein"
                elif "disease" in r or "indicat" in r or "symptom" in r:
                    node_type = "disease"
                elif "drug" in r or "compound" in r:
                    node_type = "drug"
            for u, v, data in G.in_edges(node, data=True):
                r = data.get("relation", "").lower()
                if "protein" in r or "gene" in r or "enzym" in r:
                    node_type = "protein"
                elif "disease" in r or "indicat" in r or "symptom" in r:
                    node_type = "disease"
                elif "drug" in r or "compound" in r:
                    node_type = "drug"
            
            color = INTERMEDIATE_COLOR
            if node_type == "protein":
                color = "#14B8A6" # Teal
            elif node_type == "disease":
                color = "#F43F5E" # Rose
            elif node_type == "drug":
                color = "#8B5CF6" # Violet

            net.add_node(
                node,
                label=_truncate(node, 18),
                title=f"{node} ({node_type})",
                color=color,
                size=14,
                shape="dot",
                font={"size": 10, "color": color},
            )

    # Add edges
    for h, t, data in G.edges(data=True):
        rel = data.get("relation", "")
        color = RELATION_COLORS.get(rel, EDGE_COLOR)
        net.add_edge(
            h, t,
            label=rel,
            color=color,
            width=1.5,
            title=f"{h} →[{rel}]→ {t}",
        )

    # Generate HTML
    raw_html = net.generate_html()

    # Inject fullscreen CSS and JS fix to override PyVis defaults
    css_fix = """
    <style>
        html, body { margin: 0 !important; padding: 0 !important; overflow: hidden !important; width: 100% !important; height: 100% !important; }
        #mynetwork { width: 100vw !important; height: 100vh !important; border: none !important; margin: 0 !important; }
    </style>
    <script>
        // Fallback: force styles via JS in case Gradio's CSP strips the <style> tag
        window.addEventListener('load', function() {
            var net = document.getElementById('mynetwork');
            if(net) {
                net.style.setProperty('width', '100vw', 'important');
                net.style.setProperty('height', '100vh', 'important');
                net.style.setProperty('border', 'none', 'important');
            }
            document.body.style.setProperty('margin', '0', 'important');
            document.body.style.setProperty('overflow', 'hidden', 'important');
        });
    </script>
    """
    if "</head>" in raw_html:
        raw_html = raw_html.replace("</head>", f"{css_fix}\n</head>")

    # Escape for srcdoc embedding
    escaped = html_module.escape(raw_html, quote=True)
    
    # We add a unique ID to the iframe so the button can target it, and wrap them in a container
    # allowfullscreen is required on the iframe to permit the Fullscreen API.
    import uuid
    uid = f"graph_iframe_{uuid.uuid4().hex[:8]}"
    
    iframe = (
        f'<div style="text-align: right; margin-bottom: 8px;">'
        f'<button onclick="document.getElementById(\'{uid}\').requestFullscreen()" '
        f'style="padding: 6px 12px; background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; '
        f'border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem; '
        f'transition: all 0.2s;" onmouseover="this.style.background=\'#E2E8F0\'" '
        f'onmouseout="this.style.background=\'#F1F5F9\'">⛶ View Fullscreen</button>'
        f'</div>'
        f'<iframe id="{uid}" style="width:100%; height:{height}; border:1px solid #E2E8F0; '
        f'background-color: white; border-radius:12px;" srcdoc="{escaped}" allowfullscreen></iframe>'
    )
    return iframe


def build_candidate_path_html(
    subgraph_triples: list,
    query_entity: str,
    candidate: str,
    mechanism_paths: list[str] | None = None,
    height: str = "400px",
) -> str:
    """
    Focused visualization: only paths connecting query → candidate.
    """
    if not subgraph_triples:
        return "<p style='text-align:center; color:#64748B;'>No path data available.</p>"

    # Filter triples to only those on paths between query and candidate
    G_full = nx.DiGraph()
    for triple in subgraph_triples:
        if len(triple) < 3:
            continue
        h, r, t = str(triple[0]), str(triple[1]), str(triple[2])
        G_full.add_edge(h, t, relation=r)

    # Find all simple paths (up to length 4)
    path_nodes = set()
    path_edges = set()
    try:
        for path in nx.all_simple_paths(G_full.to_undirected(), query_entity, candidate, cutoff=4):
            for node in path:
                path_nodes.add(node)
            for i in range(len(path) - 1):
                path_edges.add((path[i], path[i + 1]))
                path_edges.add((path[i + 1], path[i]))
    except (nx.NodeNotFound, nx.NetworkXError):
        path_nodes = {query_entity, candidate}

    if not path_nodes or len(path_nodes) <= 1:
        return f"<p style='text-align:center; color:#64748B;'>No paths found between <b>{query_entity}</b> and <b>{candidate}</b> in the subgraph.</p>"

    # Build filtered subgraph
    filtered_triples = []
    for triple in subgraph_triples:
        if len(triple) < 3:
            continue
        h, t = str(triple[0]), str(triple[2])
        if h in path_nodes and t in path_nodes:
            filtered_triples.append(triple)

    return build_subgraph_html(
        filtered_triples,
        query_entity,
        candidates=[candidate],
        rerank_scores=[0.15],  # highlight color
        height=height,
    )


def build_candidate_table(
    rank_entities: list[str],
    rerank_scores: list[float] | None = None,
    evidence_profiles: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Build a ranked candidates DataFrame for display.
    """
    rows = []
    for i, name in enumerate(rank_entities):
        row = {
            "Rank": i + 1,
            "Entity": name,
        }

        if rerank_scores and i < len(rerank_scores):
            row["GNN Certainty"] = f"{rerank_scores[i]:.4f}"

        if evidence_profiles and i < len(evidence_profiles):
            ep = evidence_profiles[i]
            row["PSR Score"] = f"{ep.get('psr_score', 0):.4f}"
            fs = ep.get("failsafe", {})
            row["Confidence"] = fs.get("confidence_flag", "N/A")
            row["Paths"] = len(ep.get("mechanism_texts", []))
        else:
            row["PSR Score"] = "—"
            row["Confidence"] = "—"
            row["Paths"] = "—"

        rows.append(row)

    return pd.DataFrame(rows)


def _truncate(text: str, max_len: int = 20) -> str:
    """Truncate text for node labels."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
