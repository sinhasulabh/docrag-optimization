from typing import Protocol

from atlasfin.contracts import Candidate


class Retriever(Protocol):
    def retrieve(self, query: str, k: int, filters: dict | None) -> list[Candidate]: ...
