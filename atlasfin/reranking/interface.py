from typing import Protocol

from atlasfin.contracts import Candidate


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Candidate], depth: int) -> list[Candidate]: ...
