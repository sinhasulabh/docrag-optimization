from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:  # a retrieved (and later reranked) unit
    chunk_id: str
    score: float  # retriever score, then overwritten by reranker score
    text: str  # the text that was matched
    payload_text: str  # what gets handed off (may be the parent, if parent-child)
    pages: list[int]
    metadata: dict
