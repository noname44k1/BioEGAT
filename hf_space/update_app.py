import re

with open('/home/trung/code/Graduation_Thesis/DrKGC/BioEGAT/hf_space/demo/app.py', 'r') as f:
    content = f.read()

# Update the layout
layout_search = """            gr.Markdown("### Decision Overview")
            target_box = gr.Textbox(
                label="Ground Truth Target Entity",
                placeholder="Pending inference...",
                interactive=False,
            )
            decision_box = gr.Textbox(
                label="BioEGAT LLM Prediction Decision",
                placeholder="Pending inference...",
                interactive=False,
            )"""

layout_replace = """            gr.Markdown("### Pipeline Tracking")
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
                )"""

content = content.replace(layout_search, layout_replace)

# Update run_prediction_flow empty return
empty_search = """def _empty_outputs(msg):
    return (
        msg, "", "", "N/A", "N/A", "", "",
        "<p style='text-align:center; padding:40px; color:#64748B;'>No subgraph rendering available.</p>",
        gr.update(choices=[]),
        "<p style='text-align:center; color:#64748B;'>Select a candidate to show connection paths.</p>",
    )"""

empty_replace = """def _empty_outputs(msg):
    return (
        msg, "", "", "N/A", "N/A", "N/A", "N/A", "", "",
        "<p style='text-align:center; padding:40px; color:#64748B;'>No subgraph rendering available.</p>",
        gr.update(choices=[]),
        "<p style='text-align:center; color:#64748B;'>Select a candidate to show connection paths.</p>",
    )"""

content = content.replace(empty_search, empty_replace)

# Update run_prediction_flow return
return_search = """    return (
        status,
        hero_html,
        ranked_html,
        pred or "N/A",                 # decision box = LLM's predicted entity
        res["target"],
        res["llm_prompt"],
        res["llm_response"],
        res["subgraph_html"],
        path_dropdown_update,
        first_candidate_path,
    )"""

return_replace = """    target = res.get("target")
    
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
    )"""

content = content.replace(return_search, return_replace)

# Update outputs list in predict_btn.click
click_search = """        outputs=[
            status_output,
            hero_card_viz,
            ranked_list_viz,
            decision_box,
            target_box,
            prompt_inspector_box,
            llm_reasoning_box,
            subgraph_viz,
            path_candidate_dd,
            isolated_path_viz,
        ],"""

click_replace = """        outputs=[
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
        ],"""

content = content.replace(click_search, click_replace)

with open('/home/trung/code/Graduation_Thesis/DrKGC/BioEGAT/hf_space/demo/app.py', 'w') as f:
    f.write(content)

print("Updated app.py")
