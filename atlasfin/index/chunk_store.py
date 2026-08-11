import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from atlasfin.contracts import Chunk


class ChunkStore:
    """chunk_id -> Chunk, backed by a chunks.jsonl file (one JSON object per line, row order
    == embeddings.npy row order when built alongside it in pipeline/offline.py).
    """

    def __init__(self, chunks: list[Chunk] | None = None):
        self._chunks: dict[str, Chunk] = {c.chunk_id: c for c in (chunks or [])}
        self._order: list[str] = [c.chunk_id for c in (chunks or [])]
        # parent_chunk_id -> every chunk_id sharing it (incl. itself) -- built once here so
        # RetrievalConfig.parent_child can widen a candidate's payload/pages to the whole
        # section without a per-query linear scan. See candidate_builder.hydrate().
        self._siblings_by_parent: dict[str, list[str]] = defaultdict(list)
        for c in chunks or []:
            if c.parent_chunk_id is not None:
                self._siblings_by_parent[c.parent_chunk_id].append(c.chunk_id)

    def __len__(self) -> int:
        return len(self._chunks)

    def get(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def get_or_none(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> list[Chunk]:
        return [self._chunks[cid] for cid in self._order]

    def parent_text(self, chunk_id: str) -> str:
        """The text to hand off for this chunk_id: its parent_text if it has one, else its
        own text (single-chunk sections have no parent -- see structure_aware.py).
        """
        chunk = self.get(chunk_id)
        return chunk.parent_text if chunk.parent_text is not None else chunk.text

    def siblings(self, chunk_id: str) -> list[Chunk]:
        """All chunks in the same section group as chunk_id, including itself -- i.e. the
        chunks whose concatenated text makes up chunk.parent_text. Returns just [chunk] for
        a single-chunk section (no parent_chunk_id).
        """
        chunk = self.get(chunk_id)
        if chunk.parent_chunk_id is None:
            return [chunk]
        return [self._chunks[cid] for cid in self._siblings_by_parent[chunk.parent_chunk_id]]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for chunk_id in self._order:
                f.write(json.dumps(asdict(self._chunks[chunk_id])) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ChunkStore":
        chunks: list[Chunk] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(Chunk(**json.loads(line)))
        return cls(chunks)
