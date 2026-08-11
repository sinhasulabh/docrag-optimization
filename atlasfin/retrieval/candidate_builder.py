from atlasfin.config.schema import ChunkingConfig, RetrievalConfig
from atlasfin.contracts import Candidate
from atlasfin.index.chunk_store import ChunkStore


def hydrate(
    chunk_id: str,
    score: float,
    chunk_store: ChunkStore,
    retrieval_cfg: RetrievalConfig,
    chunking_cfg: ChunkingConfig,
) -> Candidate:
    chunk = chunk_store.get(chunk_id)
    payload_text, pages = chunk.text, chunk.pages

    if retrieval_cfg.parent_child:
        siblings = chunk_store.siblings(chunk_id)
        span_pages = sorted({p for sib in siblings for p in sib.pages})
        page_range = span_pages[-1] - span_pages[0] + 1  # not len(span_pages) -- two chunks
        # can share a heading yet land far apart (e.g. pages 1 and 124, a heading-detection
        # collision), which a distinct-page-count check would miss entirely.
        if page_range <= retrieval_cfg.parent_child_max_page_range:
            # widen: payload_text and pages must always describe the same content, or
            # eval's recall@k (which checks evidence_page_num against `pages`) would credit
            # content the payload never actually included -- see chunk_store.parent_text().
            payload_text = chunk_store.parent_text(chunk_id)
            pages = span_pages
        # else: oversized/scattered section -- fall back to child-only payload_text/pages
        # rather than truncating, since a truncated page list would again describe
        # different content than payload_text.

    # pages is always populated regardless of metadata_fields -- eval structurally needs it
    # for recall@k/MRR (see contracts/chunk.py's Chunk docstring for the same reasoning).
    available = {
        "section": chunk.section,
        "pages": chunk.pages,
        "doc_type": chunk.doc_type,
        "doc_period": chunk.doc_period,
        "company": chunk.company,
    }
    metadata = {field: available[field] for field in chunking_cfg.metadata_fields if field in available}

    return Candidate(
        chunk_id=chunk_id,
        score=score,
        text=chunk.text,
        payload_text=payload_text,
        pages=pages,
        metadata=metadata,
    )
