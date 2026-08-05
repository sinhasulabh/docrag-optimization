from typing import Protocol


class VectorStore(Protocol):
    def upsert(self, chunk_ids: list[str], vectors: list[list[float]]) -> None: ...
    def search(
        self, query_vector: list[float], k: int, filters: dict | None
    ) -> list[tuple[str, float]]: ...
