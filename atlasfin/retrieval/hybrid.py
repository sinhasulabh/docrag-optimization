from atlasfin.config.schema import ChunkingConfig, RetrievalConfig
from atlasfin.contracts import Candidate
from atlasfin.embedding.interface import Embedder
from atlasfin.index.bm25_store import BM25Store
from atlasfin.index.chunk_store import ChunkStore
from atlasfin.index.vector_store import VectorStore

from . import candidate_builder, fusion


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        chunk_store: ChunkStore,
        cfg: RetrievalConfig,
        chunking_cfg: ChunkingConfig,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._chunk_store = chunk_store
        self._cfg = cfg
        self._chunking_cfg = chunking_cfg

    def retrieve(self, query: str, k: int, filters: dict | None = None) -> list[Candidate]:
        query_vector = self._embedder.embed_query(query)
        dense_hits = self._vector_store.search(
            query_vector, k, filters if self._cfg.filters_enabled else None
        )
        sparse_hits = self._bm25_store.search(query, k)

        if self._cfg.fusion == "rrf":
            fused = fusion.rrf_fuse([dense_hits, sparse_hits])
        elif self._cfg.fusion == "weighted":
            fused = fusion.weighted_fuse(
                [dense_hits, sparse_hits], [1.0 - self._cfg.bm25_weight, self._cfg.bm25_weight]
            )
        else:
            raise ValueError(f"retrieval.fusion must be rrf|weighted, got {self._cfg.fusion!r}")

        top = fused[:k]
        return [
            candidate_builder.hydrate(cid, score, self._chunk_store, self._cfg, self._chunking_cfg)
            for cid, score in top
        ]
