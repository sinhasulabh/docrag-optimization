from atlasfin.config.schema import ChunkingConfig, RetrievalConfig
from atlasfin.contracts import Candidate
from atlasfin.embedding.interface import Embedder
from atlasfin.index.chunk_store import ChunkStore
from atlasfin.index.vector_store import VectorStore

from . import candidate_builder


class DenseRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunk_store: ChunkStore,
        cfg: RetrievalConfig,
        chunking_cfg: ChunkingConfig,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_store = chunk_store
        self._cfg = cfg
        self._chunking_cfg = chunking_cfg

    def retrieve(self, query: str, k: int, filters: dict | None = None) -> list[Candidate]:
        query_vector = self._embedder.embed_query(query)
        hits = self._vector_store.search(
            query_vector, k, filters if self._cfg.filters_enabled else None
        )
        return [
            candidate_builder.hydrate(cid, score, self._chunk_store, self._cfg, self._chunking_cfg)
            for cid, score in hits
        ]
