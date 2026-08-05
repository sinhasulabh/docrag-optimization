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
    payload_text = chunk_store.parent_text(chunk_id) if retrieval_cfg.parent_child else chunk.text

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
        pages=chunk.pages,
        metadata=metadata,
    )
