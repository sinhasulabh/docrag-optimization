# voyage-finance-2 is an older ("voyage-2"-generation) hosted model; whether it supports
# Matryoshka/output_dimension truncation was NOT verified against a live call (no working
# API key at the time this was written -- see embedding/voyage_impl.py's caveat), so it's
# treated conservatively as native-dimension-only until confirmed otherwise. Newer Voyage
# models (voyage-3-large and later) are commonly documented to support truncation to these
# points -- also not independently verified here, treat as a reasonable placeholder.
VOYAGE_FINANCE_2_DIMENSIONS = {None}
VOYAGE_OTHER_MRL_DIMENSIONS = {256, 512, 1024, 2048, None}


def validate_experiment_config(cfg) -> None:
    """Cross-field checks that can't be expressed as a single dataclass field constraint.
    Raises ValueError on the first violation found.
    """
    p, c, e, r, rr = cfg.parsing, cfg.chunking, cfg.embedding, cfg.retrieval, cfg.reranking

    if p.ocr not in ("off", "on", "auto"):
        raise ValueError(f"parsing.ocr must be off|on|auto, got {p.ocr!r}")
    if p.table_mode not in ("structured", "text"):
        raise ValueError(f"parsing.table_mode must be structured|text, got {p.table_mode!r}")

    if c.strategy not in ("structure_aware", "recursive", "fixed"):
        raise ValueError(
            f"chunking.strategy must be structure_aware|recursive|fixed, got {c.strategy!r}"
        )
    if c.max_tokens <= 0:
        raise ValueError(f"chunking.max_tokens must be positive, got {c.max_tokens}")
    if c.overlap_tokens < 0:
        raise ValueError(f"chunking.overlap_tokens must be >= 0, got {c.overlap_tokens}")
    if c.overlap_tokens >= c.max_tokens:
        raise ValueError(
            f"chunking.overlap_tokens ({c.overlap_tokens}) must be < max_tokens ({c.max_tokens})"
        )
    if c.contextual_prefix and not c.contextual_model:
        raise ValueError("chunking.contextual_prefix=True requires chunking.contextual_model")

    if e.model_id == "voyage-finance-2" and e.dimension not in VOYAGE_FINANCE_2_DIMENSIONS:
        raise ValueError(
            f"embedding.dimension {e.dimension!r} is not supported -- voyage-finance-2's "
            "output_dimension truncation support is unverified, so only the native "
            "dimension (dimension=None) is allowed for this model"
        )
    if e.model_id.startswith("voyage-") and e.model_id != "voyage-finance-2":
        if e.dimension not in VOYAGE_OTHER_MRL_DIMENSIONS:
            raise ValueError(
                f"embedding.dimension {e.dimension!r} is not one of {e.model_id}'s commonly "
                f"documented MRL truncation points {sorted(d for d in VOYAGE_OTHER_MRL_DIMENSIONS if d)} "
                "(unverified against live docs -- adjust if wrong)"
            )
    if e.batch_size <= 0:
        raise ValueError(f"embedding.batch_size must be positive, got {e.batch_size}")

    if r.mode not in ("dense", "hybrid"):
        raise ValueError(f"retrieval.mode must be dense|hybrid, got {r.mode!r}")
    if r.fusion not in ("rrf", "weighted"):
        raise ValueError(f"retrieval.fusion must be rrf|weighted, got {r.fusion!r}")
    if r.fusion == "weighted" and r.mode != "hybrid":
        raise ValueError("retrieval.fusion='weighted' only makes sense when mode='hybrid'")
    if not 0.0 <= r.bm25_weight <= 1.0:
        raise ValueError(f"retrieval.bm25_weight must be in [0,1], got {r.bm25_weight}")
    if r.top_k <= 0:
        raise ValueError(f"retrieval.top_k must be positive, got {r.top_k}")
    if r.query_transform not in ("none", "rewrite", "decompose", "hyde"):
        raise ValueError(
            f"retrieval.query_transform must be none|rewrite|decompose|hyde, got {r.query_transform!r}"
        )
    if r.query_transform != "none" and not r.query_model:
        raise ValueError(f"retrieval.query_transform={r.query_transform!r} requires query_model")
    if r.index_type not in ("hnsw", "ivf", "flat"):
        raise ValueError(f"retrieval.index_type must be hnsw|ivf|flat, got {r.index_type!r}")
    if r.ann_search_param <= 0:
        raise ValueError(f"retrieval.ann_search_param must be positive, got {r.ann_search_param}")

    if rr.enabled:
        if not rr.model_id:
            raise ValueError("reranking.enabled=True requires reranking.model_id")
        if rr.depth <= 0:
            raise ValueError(f"reranking.depth must be positive, got {rr.depth}")
        if rr.depth > r.top_k:
            raise ValueError(
                f"reranking.depth ({rr.depth}) must be <= retrieval.top_k ({r.top_k})"
            )
        if rr.max_pair_tokens <= 0:
            raise ValueError(
                f"reranking.max_pair_tokens must be positive, got {rr.max_pair_tokens}"
            )
