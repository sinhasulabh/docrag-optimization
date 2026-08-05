from pathlib import Path

import bm25s


class BM25Store:
    """Local sparse (BM25) store, persisted to disk via bm25s's own .save()/.load() --
    scipy-vectorized, no server, deps (numpy/scipy) already present transitively.
    """

    def __init__(self):
        self._bm25: bm25s.BM25 | None = None

    def build(self, chunk_id_to_text: dict[str, str]) -> None:
        chunk_ids = list(chunk_id_to_text.keys())
        texts = [chunk_id_to_text[cid] for cid in chunk_ids]
        tokenized = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        self._bm25 = bm25s.BM25(corpus=chunk_ids)
        self._bm25.index(tokenized, show_progress=False)

    def search(self, query_text: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("BM25Store not built/loaded -- call .build() or .load() first")
        corpus_size = len(self._bm25.corpus)
        if corpus_size == 0:
            return []
        k = min(k, corpus_size)  # bm25s raises if k exceeds corpus size
        query_tokens = bm25s.tokenize(query_text, stopwords="en", show_progress=False)
        results, scores = self._bm25.retrieve(query_tokens, k=k, show_progress=False)
        hits: list[tuple[str, float]] = []
        for item, score in zip(results[0], scores[0]):
            chunk_id = item["text"] if isinstance(item, dict) else item
            hits.append((str(chunk_id), float(score)))
        return hits

    def save(self, path: str | Path) -> None:
        if self._bm25 is None:
            raise RuntimeError("BM25Store not built -- call .build() first")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(path), corpus=self._bm25.corpus)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Store":
        store = cls()
        store._bm25 = bm25s.BM25.load(str(path), load_corpus=True)
        return store
