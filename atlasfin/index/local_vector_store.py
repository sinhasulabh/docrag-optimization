import json
from pathlib import Path

import numpy as np


class LocalBruteForceVectorStore:
    """In-memory numpy brute-force (exact) search. Smoke-test / local-dev only -- this is
    what lets the retrieval-side pipeline be verified end to end with zero Vertex AI spend;
    it is NOT what production experiments run against (see index/vertex_vector_store.py).
    Assumes vectors are already L2-normalized (VoyageEmbedder does this when
    EmbeddingConfig.normalize=True), so a plain dot product is cosine similarity.
    """

    def __init__(self):
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None  # [n, dim], float32

    def upsert(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        if not chunk_ids:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        if self._vectors is None:
            self._vectors = arr
            self._ids = list(chunk_ids)
        else:
            self._vectors = np.vstack([self._vectors, arr])
            self._ids.extend(chunk_ids)

    def search(
        self, query_vector: list[float], k: int, filters: dict | None = None
    ) -> list[tuple[str, float]]:
        if filters:
            raise NotImplementedError(
                "LocalBruteForceVectorStore does not support metadata filters"
            )
        if self._vectors is None or len(self._ids) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        scores = self._vectors @ q
        k = min(k, len(self._ids))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._ids[i], float(scores[i])) for i in top_idx]

    def save(self, embeddings_path: str | Path, chunk_ids_path: str | Path) -> None:
        if self._vectors is None:
            raise RuntimeError("LocalBruteForceVectorStore is empty -- nothing to save")
        np.save(embeddings_path, self._vectors)
        Path(chunk_ids_path).write_text(json.dumps(self._ids))

    @classmethod
    def load(
        cls, embeddings_path: str | Path, chunk_ids_path: str | Path
    ) -> "LocalBruteForceVectorStore":
        store = cls()
        store._vectors = np.load(embeddings_path)
        store._ids = json.loads(Path(chunk_ids_path).read_text())
        return store
