# Experiment Log

Tracks retrieval-quality experiments on the full 288-doc corpus (`ingest_run_uri:
gs://atlasfin-raw-dev/run=20260808t221108z/`, offline fingerprint `v1-736f7b21f69b0662`,
141,375 chunks), scored against the 123 FinanceBench gold questions whose evidence doc is
in that corpus (`atlasfin.eval.gold.load_gold_set(doc_names=...)`).

Metrics come from `atlasfin.eval.score.score()` (recall@5/10/20, MRR, precision@5 — see
`atlasfin/eval/metrics.py` for exact definitions). Each row's config lives in
`atlasfin/experiments/configs/`. When comparing two rows, only look at the "knob(s) changed"
column as the isolated variable — everything else in that row's config matches the baseline
unless noted.

| Date | Experiment | Config | Knob(s) changed vs. baseline | recall@5 | recall@10 | recall@20 | MRR | precision@5 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-10 | baseline | [`baseline_dense_full.yaml`](../configs/baseline_dense_full.yaml) | — (baseline) | 0.301 | 0.439 | 0.528 | 0.208 | n/a | Dense-only, `voyage-4`, `top_k=20`, no reranking. |
| 2026-08-10 | dense + reranker | [`dense_rerank_full.yaml`](../configs/dense_rerank_full.yaml) | `reranking.enabled: false → true` (cross-encoder/ms-marco-MiniLM-L-6-v2, depth=20) | 0.244 | 0.358 | 0.528 | 0.141 | 0.050 | Retrieval unchanged (still dense, `top_k=20`) — only the reranker was added on top. recall@20 is identical to baseline by construction (rerank depth == top_k, so it only reorders the same 20 candidates, never changes the set). Every other metric got **worse**: recall@5 −0.057, recall@10 −0.081, MRR −0.067. The general-purpose cross-encoder (trained on short web-search snippets) appears to be mis-ranking financial-filing chunks rather than improving on dense similarity — worth swapping for a finance-tuned or larger reranker before trusting reranking in this pipeline. |
