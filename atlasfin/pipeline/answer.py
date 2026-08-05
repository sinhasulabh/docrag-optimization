import time
from dataclasses import dataclass

from atlasfin.config.schema import ExperimentConfig
from atlasfin.contracts import Candidate

from .online import OnlineComponents


@dataclass
class AnswerResult:
    question: str
    candidates: list[Candidate]  # final ranked list: post-rerank if it ran, else post-retrieve
    retrieve_ms: float
    rerank_ms: float | None  # None if reranking did not run


def answer(question: str, online: OnlineComponents, cfg: ExperimentConfig) -> AnswerResult:
    """Stops at "ranked context handed off" -- no generation call, per the spec's explicit
    non-goal. filters=None always: FinanceBench questions carry a grounding doc_name, but
    passing that as a retrieval filter would trivially inflate recall by restricting the
    search space to the answer's own document, defeating the point of evaluating retrieval.
    """
    started = time.perf_counter()
    candidates = online.retriever.retrieve(question, cfg.retrieval.top_k, filters=None)
    retrieve_ms = (time.perf_counter() - started) * 1000

    rerank_ms = None
    if online.reranker is not None:
        rerank_started = time.perf_counter()
        candidates = online.reranker.rerank(question, candidates, cfg.reranking.depth)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

    return AnswerResult(
        question=question, candidates=candidates, retrieve_ms=retrieve_ms, rerank_ms=rerank_ms
    )
