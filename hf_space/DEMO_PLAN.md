# BioEGAT Explainable Demo — Implementation Plan

Goal: a HuggingFace Space that lets a reviewer pick a **dataset → query**, see the
**BioEGAT+LLM ranked candidates** with **structural + statistical evidence** (subgraph
paths, PSR, p-value tier) and **verified external references** (NCT/PMID). Same card
structure as the static mockup (`report/.../bioegat_demo_walkthrough.html`), but
deployed on the live Gradio scaffold and covering all four datasets.

---

## 0. Current state (reuse, don't rewrite)

`BioEGAT/hf_space/demo/`
- `app.py` — Gradio UI: gallery table + entity/relation dropdowns + select callback.
- `precomputed.py` — `PrecomputedResults`: loads one JSON, indexes by entity/relation
  (`get_all_entities`, `get_all_relations`, `get_gallery_rows`, `get_by_index`).
- `pipeline.py` — `DemoPipeline`: wraps precomputed + optional live Gemini + `synthesizer.LLMReranker`.
- `visualizer.py` — `build_subgraph_html` (pyvis), `build_candidate_path_html`, `build_candidate_table`, `_certainty_color`.
- `data/sample_results.json` — 20 PrimeKG records, schema:
  `triple, triple_id, type, query_entity(_id), rank_entities(_id), rank, subgraph,`
  `rerank_scores, evidence_profiles[], llm_prompt`.
- `synthesizer/psr_scorer.py` — `compute_psr_score`, `compute_path_confidence`,
  `batch_psr_scores`, `compute_confidence_from_stats`.
- `synthesizer/path_unroller.py` — turns subgraph edges into Subject→Rel→Object text.

Raw inputs available now: `dataset-logos/{primekg,drugmechdb,pharmkg}/prediction.json`
(+ `id2entity.pkl` each). **`hetionet/prediction.json` is missing** — either supply it
or build Hetionet from `reranked_test.json` (rerank-only).

The work splits cleanly into **(A) an offline data-prep builder** and **(B) UI changes**.

---

## A. Offline data-prep builder  `demo/build_demo_data.py`

Why: `prediction.json` is up to 1.1 GB (pharmkg) — far too big to ship/load in a Space.
We precompute a **small curated JSON per dataset** (~hundreds of KB) in the
`sample_results.json` schema, so the Space only ever loads tiny files.

Inputs per dataset `<ds>`:
- `dataset-logos/<ds>/prediction.json`  (stream — never `json.load` the whole pharmkg file;
  reuse the streaming `JSONDecoder.raw_decode` parser from `scratch/case_study.py`).
- `dataset-logos/<ds>/id2entity.pkl`, `id2relation.pkl`.
- `lexicon/` relation legend (PharmKG GNBR codes → names) for display.
- `demo/evidence/<ds>_evidence.json` — curated NCT/PMID map (see §C).

Steps:
1. **Select showcase queries** (8–12 per dataset), prioritising:
   - gold recovered at rank 1 (clean wins), plus 1–2 "full-pipeline lift" cases
     (gold low in KGE → rank 1 after LLM), plus 1–2 drug→disease *and* disease→drug.
   - Prefer recognisable entities (named drugs/diseases). Hard-code the chosen
     `triple_id`s in a `SHOWCASES[<ds>]` list so the build is deterministic.
2. For each selected record, emit one demo object:
   - `query_entity`, `relation` (decode code→name), `type`, `gold` (name), `pred_rank`, `pred`.
   - `rank_entities` (names, top-10) + `rerank_scores` (BioEGAT certainty if present;
     else derive a monotone proxy and flag it).
   - `subgraph` (keep, but **prune to the union of ≤3-hop paths to the top-10 candidates**
     to bound size; translate ids→names).
   - `evidence_profiles[]` per top-10 candidate:
     - `mechanism_texts` = top-k unrolled paths query→candidate (`path_unroller`),
     - `support` = #≤3-hop paths (compute via the BFS in `scratch/detail.py`),
     - `psr`, `pvalue`, `tier` (`psr_scorer.compute_psr_score` + `compute_confidence_from_stats`),
     - `evidence_ref` = `{source, id, url}` from the curated map (§C), or null.
3. Write `demo/data/<ds>_demo.json` (list of objects) + a `demo/data/index.json`
   listing `{dataset: {label, file, n}}`.
4. Keep each file < ~500 KB; assert and log if larger (drop longest paths first).

Memory: stream the big files; build dicts keyed by `triple_id`; only materialise the
selected records. (Mirror the patterns already in `scratch/case_study.py` / `detail.py`.)

---

## B. UI changes (Gradio)

### B1. `precomputed.py` → multi-dataset registry
- Replace single-file load with a `DemoData` class that reads `data/index.json` and
  lazily loads `data/<ds>_demo.json` on demand; keep the per-dataset entity/relation indices.
- API: `datasets()`, `queries(ds)`, `get(ds, query_id)`.

### B2. `app.py` → the card layout (match the mockup)
Top-to-bottom, reusing the mockup's visual language via `gr.HTML`:
1. **Header** — title + one-line description + the 4-stage pipeline strip.
2. **Dataset tabs / dropdown** — PrimeKG · DrugMechDB · Hetionet · PharmKG.
3. **Query picker** — dropdown of that dataset's showcase queries
   ("Imatinib — treats — ?"), or keep the existing gallery table.
4. **Hero card** (rank-1) — `gr.HTML`: gold badge, confidence tier, supporting-path count,
   evidence paths (mono, entity-coloured), verified reference chip.
5. **Ranked list** (#2–10) — `gr.HTML` rows: rank, candidate, evidence-strength bar,
   tier chip, top path, reference chip. (Port the CSS from the mockup into a `<style>`
   block returned by a `render_html(item)` helper, or `app.css`.)
6. **Interactive subgraph** — reuse `visualizer.build_subgraph_html` (pyvis) for the
   selected query; colour nodes by entity type (drug/gene/disease).
7. **Legend + "how to read"** note (ranking = BioEGAT+LLM; paths/PSR corroborate).

Selecting a dataset repopulates the query picker; selecting a query re-renders hero +
list + subgraph. Keep the existing **optional live-LLM toggle** (Gemini) as a secondary
tab/accordion — primary path is precomputed.

### B3. `visualizer.py`
- Reuse as-is; add entity-type colouring (pass a `node_type` map derived from the
  relation/entity metadata) and PSR/p-value into the candidate table.

---

## C. Verified evidence references (reuse the workflow output)

- The FuseLinker evidence curation already produced verified NCT/PMID for
  imatinib→diseases and RA→compounds (see `tasks/wgmzec72b.output`). Persist these as
  `demo/evidence/<ds>_evidence.json`: `{ "<query_id or candidate_name>": {source,id,url} }`.
- For other showcase queries, run the same verify step (a few candidates each):
  re-invoke the saved workflow `fuselinker-evidence-wf_*.js` with the new candidate list,
  or do manual lookups. **Only store identifiers verified to resolve**; otherwise omit.

---

## D. Statistic integration (PSR / p-value)

- Compute offline in the builder (§A.2) using `synthesizer/psr_scorer.py`:
  - PSR per path type from the training split (precompute once per dataset; cache).
  - Binomial p-value → tier via `compute_confidence_from_stats`.
- Store the resulting `psr`, `pvalue`, `tier` per candidate in the demo JSON.
- Show in the card as a chip; the live pipeline path can recompute on the fly for custom
  queries (existing `pipeline.run`).

---

## E. Deployment (HF Space)

- `requirements.txt`: keep `gradio`, `pandas`, `networkx`, `pyvis`, `numpy`; `google-genai`
  + `python-dotenv` only if the live tab is kept. Drop heavy deps (no torch/transformers —
  everything is precomputed).
- Ship only `demo/` + `demo/data/*_demo.json` + `demo/evidence/*` + `synthesizer/{psr_scorer,path_unroller}.py`.
  Do **not** ship the raw `prediction.json` files.
- Secrets: `GEMINI_API_KEY` (optional, only for live mode).
- `app.py` is the Space entrypoint; verify `sys.path` so `synthesizer` imports resolve
  (the demo currently appends the project root — keep, or vendor the two synthesizer files).

---

## F. Build & test order (suggested)

1. `build_demo_data.py` for **primekg** (smallest, predictions exist) → `primekg_demo.json`;
   eyeball it.
2. Multi-dataset `precomputed.py` + `index.json`; load primekg in the app.
3. Port the mockup card CSS/HTML into `app.py` (`render_html`) — hero + ranked list.
4. Wire dataset/query selectors + pyvis subgraph.
5. Add `drugmechdb` and `pharmkg` (curated subset) → their `_demo.json`.
6. Merge the curated NCT/PMID evidence chips.
7. **hetionet**: get `prediction.json`, else build from `reranked_test.json` (rerank-only,
   mark "no LLM re-ranking").
8. Smoke-test locally (`python -m demo.app`), check file sizes, then push to the Space.

---

## G. Open items / decisions
- **hetionet/prediction.json** is missing — supply it, or accept rerank-only for that tab.
- **rerank_scores / BioEGAT certainty**: confirm `prediction.json` carries per-candidate
  rerank scores; if not, the builder derives a rank-based proxy (flagged in the card note).
- **UI framework**: plan assumes staying on **Gradio** (existing scaffold + pyvis). If you'd
  rather ship the static HTML card app, swap §B for "serve the mockup HTML + inject
  `data/<ds>_demo.json` via a small JS loader" — same data contract from §A.
