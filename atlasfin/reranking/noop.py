from atlasfin.contracts import Candidate


class NoRerank:
    def rerank(self, query: str, candidates: list[Candidate], depth: int) -> list[Candidate]:
        return candidates[:depth]
