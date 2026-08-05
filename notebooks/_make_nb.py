"""Generates exploration.ipynb programmatically so the JSON is always valid."""
import json
from pathlib import Path

def md(source: str):
    return {"cell_type": "markdown", "id": None, "metadata": {}, "source": source}

def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

cells = []

# ── Intro markdown ────────────────────────────────────────────────────────────
cells.append(md(
"""# AtlasFin RAG — Pipeline Exploration Notebook

This notebook walks through every stage of the pipeline in one place:

1. **Config** — load an experiment config and inspect all knobs
2. **Gold set** — explore FinanceBench questions and evidence
3. **Chunks** — inspect pre-built chunks from disk
4. **Embedding** — embed a query and inspect the vector (needs `VOYAGE_API_KEY`)
5. **Local retrieval** — dense + hybrid against `LocalBruteForceVectorStore` (no Vertex spend)
6. **Reranking** — cross-encoder before/after comparison
7. **Eval metrics** — recall@k, MRR, precision@5 against the gold set
8. **Experiment comparison** — diff two configs side-by-side
9. **Per-question deep-dive** — find misses and inspect them

---
### Prerequisites
```bash
uv sync --dev          # install deps including jupyter
export VOYAGE_API_KEY=your-voyage-key
export GEMINI_API_KEY=your-gemini-key   # only for contextual_prefix / query_transform
uv run jupyter lab
```
Cells that call external APIs are marked **[KEY REQUIRED]**."""
))

# ── Cell 0: sys.path ──────────────────────────────────────────────────────────
cells.append(code(
"""# Cell 0 — run this first; adds repo root to sys.path
import sys
from pathlib import Path

REPO_ROOT = Path("__file__").resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("Repo root:", REPO_ROOT)
print("Python:", sys.version)"""
))

# ── Section 1 ─────────────────────────────────────────────────────────────────
cells.append(md("## 1. Config — load an experiment and inspect all knobs"))

cells.append(code(
"""import dataclasses, json
from atlasfin.config.loader import load_experiment_config

CONFIG_PATH = REPO_ROOT / "atlasfin/experiments/configs/baseline_dense_3m.yaml"

spec = load_experiment_config(CONFIG_PATH)
cfg  = spec.config

print(f"Experiment name    : {cfg.name}")
print(f"Offline fingerprint: {cfg.offline_fingerprint()}")
print(f"Ingest run URI     : {spec.ingest_run_uri}")
print(f"Docs to index      : {spec.doc_names}")
print()
print("--- Full config (JSON) ---")
print(json.dumps(dataclasses.asdict(cfg), indent=2))"""
))

cells.append(code(
"""# Show the offline/online split — which knobs invalidate the cache
import pandas as pd

knobs = [
    ("parsing",   "backend",           cfg.parsing.backend,           "offline"),
    ("parsing",   "ocr",               cfg.parsing.ocr,               "offline"),
    ("parsing",   "table_mode",        cfg.parsing.table_mode,        "offline"),
    ("chunking",  "strategy",          cfg.chunking.strategy,         "offline"),
    ("chunking",  "max_tokens",        cfg.chunking.max_tokens,       "offline"),
    ("chunking",  "overlap_tokens",    cfg.chunking.overlap_tokens,   "offline"),
    ("chunking",  "contextual_prefix", cfg.chunking.contextual_prefix,"offline"),
    ("embedding", "model_id",          cfg.embedding.model_id,        "offline"),
    ("embedding", "dimension",         cfg.embedding.dimension,       "offline"),
    ("embedding", "normalize",         cfg.embedding.normalize,       "offline"),
    ("retrieval", "mode",              cfg.retrieval.mode,            "online"),
    ("retrieval", "fusion",            cfg.retrieval.fusion,          "online"),
    ("retrieval", "top_k",             cfg.retrieval.top_k,           "online"),
    ("retrieval", "query_transform",   cfg.retrieval.query_transform, "online"),
    ("retrieval", "parent_child",      cfg.retrieval.parent_child,    "online"),
    ("reranking", "enabled",           cfg.reranking.enabled,         "online"),
    ("reranking", "model_id",          cfg.reranking.model_id,        "online"),
    ("reranking", "depth",             cfg.reranking.depth,           "online"),
]
df = pd.DataFrame(knobs, columns=["stage", "knob", "value", "type"])
df.style.map(
    lambda v: "background-color: #ffe0b2" if v == "offline" else "background-color: #c8e6c9",
    subset=["type"]
)"""
))

# ── Section 2 ─────────────────────────────────────────────────────────────────
cells.append(md("## 2. Gold set — explore FinanceBench questions and evidence"))

cells.append(code(
"""from atlasfin.eval.gold import load_gold_set

GOLD_PATH = REPO_ROOT / "financebench_open_source.jsonl"

gold_all = load_gold_set(GOLD_PATH)
gold_3m  = load_gold_set(GOLD_PATH, doc_names=spec.doc_names)

print(f"Total FinanceBench questions : {len(gold_all)}")
print(f"Questions grounded in 3M docs: {len(gold_3m)}")
print()

from collections import Counter
companies = Counter(g.company for g in gold_all)
print("Top companies in gold set:")
for company, count in companies.most_common(10):
    print(f"  {company:<20} {count} questions")"""
))

cells.append(code(
"""# Inspect a single gold record in detail
sample = gold_3m[0]
print(f"ID            : {sample.financebench_id}")
print(f"Company       : {sample.company}")
print(f"Doc           : {sample.doc_name}")
print(f"Question      : {sample.question}")
print(f"Answer        : {sample.answer}")
print(f"Evidence pages: {[e.evidence_page_num for e in sample.evidence]}")
print()
print("Evidence text (first item, first 400 chars):")
print(sample.evidence[0].evidence_text[:400], "...")"""
))

# ── Section 3 ─────────────────────────────────────────────────────────────────
cells.append(md(
"""## 3. Build offline artifacts — parse → chunk → embed → BM25

This is the main offline step. It reads your already-parsed documents from GCS,
chunks them, embeds them with Voyage, and saves everything to a local cache keyed
by the offline fingerprint. It only rebuilds if the fingerprint changes.

**Set your GCP project below and run the cell.** It will take a few minutes the
first time (embedding 8 x 10-K filings). On subsequent runs it loads from cache
instantly.

> **[KEY REQUIRED]** `VOYAGE_API_KEY`"""
))

cells.append(code(
"""# [KEY REQUIRED] VOYAGE_API_KEY
# Set your GCP project here — used to access the GCS bucket where parsed docs live.
GCP_PROJECT  = "project-7dfc6f6c-1703-48a3-83b"   # your active gcloud project
GCP_LOCATION = "us-central1"

import logging, os
from atlasfin.pipeline.cache import offline_dir
from atlasfin.index.chunk_store import ChunkStore

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s")

fp          = cfg.offline_fingerprint()
off_dir     = offline_dir(fp)
chunks_path = off_dir / "chunks.jsonl"

print(f"Offline fingerprint : {fp}")
print(f"Cache dir           : {off_dir}")
print(f"Already cached      : {chunks_path.exists()}")
print()

if not chunks_path.exists():
    print("Cache MISS — running parse → chunk → embed → BM25 ...")
    print("(downloading parsed docs from GCS, embedding with Voyage — takes a few minutes)")

from atlasfin.pipeline.offline import build_offline

index = build_offline(
    cfg,
    spec.doc_names,
    ingest_run_uri=spec.ingest_run_uri,
    gcp_project=None,          # None = skip Vertex index creation (local smoke only)
    gcp_location=GCP_LOCATION,
    parse_device="auto",
)

store  = ChunkStore.load(index.chunks_path)
chunks = store.all_chunks()
print(f"\\nOffline build complete.")
print(f"  Chunks          : {index.n_chunks}")
print(f"  Cache dir       : {index.offline_dir}")"""
))

cells.append(code(
"""# Inspect chunk distribution now that the cache is populated
import matplotlib.pyplot as plt
from collections import Counter

token_counts = [c.token_count for c in chunks]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(token_counts, bins=40, edgecolor="white", color="#4a90d9")
axes[0].axvline(cfg.chunking.max_tokens, color="tomato", linestyle="--",
                label=f"max_tokens={cfg.chunking.max_tokens}")
axes[0].set_xlabel("Token count (proxy tokenizer)")
axes[0].set_ylabel("Number of chunks")
axes[0].set_title("Chunk token distribution")
axes[0].legend()

per_doc = Counter(c.doc_name for c in chunks)
docs, counts = zip(*sorted(per_doc.items()))
axes[1].barh(docs, counts, color="#4a90d9", edgecolor="white")
axes[1].set_xlabel("Chunks")
axes[1].set_title("Chunks per document")

plt.tight_layout()
plt.show()

print(f"Total chunks : {len(chunks)}")
print(f"Avg tokens   : {sum(token_counts)/len(token_counts):.0f}")
print(f"Max tokens   : {max(token_counts)}")
print(f"Min tokens   : {min(token_counts)}")"""
))

cells.append(code(
"""# Print a few sample chunks
for i, c in enumerate(chunks[:3]):
    print(f"--- Chunk {i} ---")
    print(f"  chunk_id    : {c.chunk_id}")
    print(f"  doc_name    : {c.doc_name}")
    print(f"  section     : {c.section}")
    print(f"  pages       : {c.pages}")
    print(f"  token_count : {c.token_count}")
    print(f"  text[:200]  : {c.text[:200]}")
    print()"""
))

# ── Section 4 ─────────────────────────────────────────────────────────────────
cells.append(md(
"""## 4. Embedding — embed a query and inspect the vector

> **[KEY REQUIRED]** `VOYAGE_API_KEY` must be set before running these cells."""
))

cells.append(code(
"""# [KEY REQUIRED] VOYAGE_API_KEY
import os

if not os.environ.get("VOYAGE_API_KEY"):
    raise EnvironmentError(
        "VOYAGE_API_KEY is not set.\\n"
        "Stop Jupyter, run: export VOYAGE_API_KEY=your-key, then restart."
    )

from atlasfin.embedding import build as build_embedder

embedder = build_embedder(cfg.embedding)
print(f"Embedder model : {cfg.embedding.model_id}")
print(f"Normalize      : {cfg.embedding.normalize}")
print(f"Dimension      : {cfg.embedding.dimension or 'model native'}")"""
))

cells.append(code(
"""import math

SAMPLE_QUERY = "What was 3M's capital expenditure in FY2018?"

query_vec = embedder.embed_query(SAMPLE_QUERY)
dim  = len(query_vec)
norm = math.sqrt(sum(x*x for x in query_vec))

print(f"Query        : {SAMPLE_QUERY}")
print(f"Vector dim   : {dim}")
print(f"L2 norm      : {norm:.6f}  (should be ~1.0 since normalize=True)")
print(f"First 8 dims : {[round(v, 4) for v in query_vec[:8]]}")"""
))

cells.append(code(
"""# Verify query vs doc vectors differ (Voyage uses task-specific prefixes)
doc_vec = embedder.embed_docs([SAMPLE_QUERY])[0]
dot = sum(q*d for q, d in zip(query_vec, doc_vec))
print(f"Dot product (query_vec · doc_vec of SAME text): {dot:.4f}")
print("Should be < 1.0 — Voyage prepends different task prefixes for query vs document.")"""
))

# ── Section 5 ─────────────────────────────────────────────────────────────────
cells.append(md(
"""## 5. Local retrieval — dense + hybrid (no Vertex spend)

Uses `LocalBruteForceVectorStore` (in-memory exact search) instead of Vertex AI.
Requires the offline cache from Section 3.

> **[KEY REQUIRED]** `VOYAGE_API_KEY`"""
))

cells.append(code(
"""# Wire up the local retriever from the index built in Section 3.
# Uses LocalBruteForceVectorStore (exact in-memory search, no Vertex spend).
import numpy as np
from atlasfin.index.local_vector_store import LocalBruteForceVectorStore
from atlasfin.index.bm25_store import BM25Store
from atlasfin.index.chunk_store import ChunkStore
from atlasfin.retrieval import build_local_smoke

chunk_store  = ChunkStore.load(index.chunks_path)
vector_store = LocalBruteForceVectorStore.load(index.embeddings_path, index.chunk_ids_path)
bm25_store   = BM25Store.load(index.bm25_path) if index.bm25_path.exists() else None

retriever = build_local_smoke(
    cfg.retrieval,
    embedding_cfg=cfg.embedding,
    chunking_cfg=cfg.chunking,
    chunk_store=chunk_store,
    vector_store=vector_store,
)
shape = vector_store._vectors.shape
print(f"Dense retriever ready — {len(chunk_store)} chunks, embedding matrix {shape}")
print(f"BM25 store loaded : {bm25_store is not None}")"""
))

cells.append(code(
"""# Run a single query and print the top-k candidates
if retriever is not None:
    query = "What was 3M's capital expenditure in FY2018?"
    candidates = retriever.retrieve(query, k=cfg.retrieval.top_k)

    print(f"Query: {query}")
    print(f"Top {len(candidates)} candidates:\\n")
    for rank, c in enumerate(candidates, 1):
        print(f"  [{rank:>2}] score={c.score:.4f}  pages={c.pages}  doc={c.chunk_id.split('#')[0]}")
        print(f"       {c.text[:120].strip()} ...")"""
))

cells.append(code(
"""# Compare dense vs hybrid (RRF) side-by-side
import dataclasses

if retriever is not None and bm25_store is not None:
    hybrid_cfg = dataclasses.replace(cfg.retrieval, mode="hybrid", fusion="rrf")
    hybrid_retriever = build_local_smoke(
        hybrid_cfg,
        embedding_cfg=cfg.embedding,
        chunking_cfg=cfg.chunking,
        chunk_store=chunk_store,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    dense_ids    = [c.chunk_id for c in candidates]
    hybrid_cands = hybrid_retriever.retrieve(query, k=cfg.retrieval.top_k)
    hybrid_ids   = [c.chunk_id for c in hybrid_cands]

    only_dense  = [cid for cid in dense_ids  if cid not in hybrid_ids]
    only_hybrid = [cid for cid in hybrid_ids if cid not in dense_ids]

    print(f"Dense  top-{cfg.retrieval.top_k}: {len(set(dense_ids))} unique chunks")
    print(f"Hybrid top-{cfg.retrieval.top_k}: {len(set(hybrid_ids))} unique chunks")
    print(f"In dense but not hybrid : {len(only_dense)}")
    print(f"In hybrid but not dense : {len(only_hybrid)}  ← BM25 surfaced these")
else:
    print("Skipped: retriever or BM25 store not available.")"""
))

# ── Section 6 ─────────────────────────────────────────────────────────────────
cells.append(md(
"""## 6. Reranking — cross-encoder before/after comparison

Runs locally via `sentence-transformers` — no API key required.
Green shift = moved up in ranking, red = moved down."""
))

cells.append(code(
"""from atlasfin.config.schema import RerankingConfig
from atlasfin.reranking import build as build_reranker

rerank_cfg = RerankingConfig(
    enabled=True,
    model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
    depth=20,
    max_pair_tokens=512,
)
reranker = build_reranker(rerank_cfg)
print("Reranker loaded (sentence-transformers, runs locally — no API key needed)")"""
))

cells.append(code(
"""if retriever is not None:
    reranked = reranker.rerank(query, candidates, depth=rerank_cfg.depth)

    before_rank = {c.chunk_id: i + 1 for i, c in enumerate(candidates)}
    rows = []
    for after_rank, c in enumerate(reranked, 1):
        before = before_rank.get(c.chunk_id, "?")
        shift  = (before - after_rank) if isinstance(before, int) else 0
        rows.append({
            "after_rank" : after_rank,
            "before_rank": before,
            "shift"      : shift,
            "score"      : round(c.score, 4),
            "pages"      : c.pages,
            "text_preview": c.text[:80].replace("\\n", " "),
        })

    df_rerank = pd.DataFrame(rows)

    def color_shift(val):
        if val > 0: return "color: green"
        if val < 0: return "color: red"
        return ""

    print(f"Query: {query}")
    df_rerank.style.map(color_shift, subset=["shift"])
else:
    print("Skipped: retriever not available.")"""
))

# ── Section 7 ─────────────────────────────────────────────────────────────────
cells.append(md(
"""## 7. Eval metrics — score recall@k, MRR, latency

> **[KEY REQUIRED]** `VOYAGE_API_KEY` | Requires offline cache."""
))

cells.append(code(
"""import time
from atlasfin.pipeline.answer import AnswerResult
from atlasfin.eval.score import score as eval_score

if retriever is None:
    print("Skipped: offline artifacts not found.")
else:
    print(f"Scoring {len(gold_3m)} questions ...")
    results = []
    for g in gold_3m:
        t0    = time.perf_counter()
        cands = retriever.retrieve(g.question, k=cfg.retrieval.top_k)
        ms    = (time.perf_counter() - t0) * 1000
        results.append(AnswerResult(question=g.question, candidates=cands,
                                    retrieve_ms=ms, rerank_ms=None))

    metrics = eval_score(results, gold_3m)
    print("\\n--- Metrics (dense, no rerank) ---")
    print(f"  recall@5    : {metrics.recall_at[5]:.3f}")
    print(f"  recall@10   : {metrics.recall_at[10]:.3f}")
    print(f"  recall@20   : {metrics.recall_at[20]:.3f}")
    print(f"  MRR         : {metrics.mrr:.3f}")
    print(f"  retrieve p50: {metrics.retrieve_latency_p50:.1f} ms")
    print(f"  retrieve p95: {metrics.retrieve_latency_p95:.1f} ms")"""
))

cells.append(code(
"""# Recall@k bar chart
if retriever is not None:
    import matplotlib.pyplot as plt

    ks      = sorted(metrics.recall_at.keys())
    recalls = [metrics.recall_at[k] for k in ks]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([str(k) for k in ks], recalls, color="#4a90d9", edgecolor="white")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title(f"{cfg.name} — Recall@k ({len(gold_3m)} 3M questions)")
    for i, (k, r) in enumerate(zip(ks, recalls)):
        ax.text(i, r + 0.01, f"{r:.2f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.show()"""
))

# ── Section 8 ─────────────────────────────────────────────────────────────────
cells.append(md(
"""## 8. Experiment comparison — diff two configs side-by-side

`baseline_dense_3m` vs `hybrid_rerank_3m`.
Both share the same offline fingerprint, so no rebuild is needed for B."""
))

cells.append(code(
"""from atlasfin.config.loader import load_experiment_config

spec_a = load_experiment_config(REPO_ROOT / "atlasfin/experiments/configs/baseline_dense_3m.yaml")
spec_b = load_experiment_config(REPO_ROOT / "atlasfin/experiments/configs/hybrid_rerank_3m.yaml")

fp_a, fp_b = spec_a.config.offline_fingerprint(), spec_b.config.offline_fingerprint()
print(f"Config A: {spec_a.config.name}  fingerprint: {fp_a}")
print(f"Config B: {spec_b.config.name}  fingerprint: {fp_b}")
print()
if fp_a == fp_b:
    print("✓ Same fingerprint — shared cached artifacts, no rebuild needed for B")
else:
    print("✗ Different fingerprints — B would trigger a rebuild")"""
))

cells.append(code(
"""# Show exactly which knobs differ
import dataclasses

def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(_flatten(v, key)) if isinstance(v, dict) else out.update({key: v})
    return out

fa, fb = _flatten(dataclasses.asdict(spec_a.config)), _flatten(dataclasses.asdict(spec_b.config))
diffs  = [(k, fa[k], fb[k]) for k in fa if fa[k] != fb.get(k)]

print(f"Knobs that differ ({len(diffs)} total):\\n")
for k, va, vb in diffs:
    print(f"  {k:<35} {str(va):<20} → {vb}")"""
))

cells.append(code(
"""# Side-by-side metrics comparison (requires offline cache)
if retriever is not None and bm25_store is not None:
    hybrid_retriever_b = build_local_smoke(
        spec_b.config.retrieval,
        embedding_cfg=spec_b.config.embedding,
        chunking_cfg=spec_b.config.chunking,
        chunk_store=chunk_store,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )
    reranker_b = build_reranker(spec_b.config.reranking)

    results_b = []
    for g in gold_3m:
        t0    = time.perf_counter()
        cands = hybrid_retriever_b.retrieve(g.question, k=spec_b.config.retrieval.top_k)
        r_ms  = (time.perf_counter() - t0) * 1000
        t1    = time.perf_counter()
        cands = reranker_b.rerank(g.question, cands, depth=spec_b.config.reranking.depth)
        rr_ms = (time.perf_counter() - t1) * 1000
        results_b.append(AnswerResult(question=g.question, candidates=cands,
                                      retrieve_ms=r_ms, rerank_ms=rr_ms))

    metrics_b = eval_score(results_b, gold_3m)

    comparison = pd.DataFrame([
        {"metric": "recall@5",     "A (dense)": f"{metrics.recall_at[5]:.3f}",   "B (hybrid+rerank)": f"{metrics_b.recall_at[5]:.3f}"},
        {"metric": "recall@10",    "A (dense)": f"{metrics.recall_at[10]:.3f}",  "B (hybrid+rerank)": f"{metrics_b.recall_at[10]:.3f}"},
        {"metric": "recall@20",    "A (dense)": f"{metrics.recall_at[20]:.3f}",  "B (hybrid+rerank)": f"{metrics_b.recall_at[20]:.3f}"},
        {"metric": "MRR",          "A (dense)": f"{metrics.mrr:.3f}",            "B (hybrid+rerank)": f"{metrics_b.mrr:.3f}"},
        {"metric": "precision@5",  "A (dense)": "n/a (no rerank)",               "B (hybrid+rerank)": f"{metrics_b.precision_at_5:.3f}"},
        {"metric": "p50 retrieve", "A (dense)": f"{metrics.retrieve_latency_p50:.1f}ms", "B (hybrid+rerank)": f"{metrics_b.retrieve_latency_p50:.1f}ms"},
        {"metric": "p50 rerank",   "A (dense)": "n/a",                           "B (hybrid+rerank)": f"{metrics_b.rerank_latency_p50:.1f}ms"},
    ]).set_index("metric")
    comparison
else:
    print("Skipped: offline artifacts not found.")"""
))

# ── Section 9 ─────────────────────────────────────────────────────────────────
cells.append(md(
"""## 9. Per-question deep-dive — find misses and inspect them

Which questions are recall misses at k=5 but hits at k=20?
What does the ranked list look like for those questions?"""
))

cells.append(code(
"""from atlasfin.eval.metrics import _is_hit

if retriever is not None:
    miss_at_5_hit_at_20 = [
        (g, res)
        for g, res in zip(gold_3m, results)
        if not any(_is_hit(c, g.evidence) for c in res.candidates[:5])
        and     any(_is_hit(c, g.evidence) for c in res.candidates[:20])
    ]

    print(f"Miss@5 but Hit@20: {len(miss_at_5_hit_at_20)} questions\\n")
    for g, res in miss_at_5_hit_at_20[:3]:
        gold_pages = [e.evidence_page_num for e in g.evidence]
        print(f"Q: {g.question[:100]}")
        print(f"   Gold pages: {gold_pages}")
        for rank, c in enumerate(res.candidates[:20], 1):
            hit    = _is_hit(c, g.evidence)
            marker = "  ← GOLD HIT" if hit else ""
            print(f"   [{rank:>2}] pages={c.pages} score={c.score:.4f}{marker}")
        print()
else:
    print("Skipped: retriever not available.")"""
))

# ── Write out ─────────────────────────────────────────────────────────────────
import random, string
for i, cell in enumerate(cells):
    cell["id"] = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13"},
    },
    "cells": cells,
}

out = Path(__file__).parent / "exploration.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written {len(cells)} cells to {out}")
