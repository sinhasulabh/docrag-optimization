from typing import Protocol

from atlasfin.contracts import Candidate

from .gold import GoldRecord
from .metrics import Metrics, latency_percentiles, mrr, precision_at_5, recall_at_k


class ScoredResult(Protocol):
    """Structural type, not imported from pipeline/ -- eval/ must not depend on pipeline/'s
    types (pipeline/ depends on eval/, not the other way around). pipeline/answer.py's
    AnswerResult satisfies this shape.
    """

    candidates: list[Candidate]  # final ranked list: post-rerank if it ran, else post-retrieve
    retrieve_ms: float
    rerank_ms: float | None  # None if reranking did not run for this question


def score(results: list[ScoredResult], gold_set: list[GoldRecord]) -> Metrics:
    if len(results) != len(gold_set):
        raise ValueError(
            f"results ({len(results)}) and gold_set ({len(gold_set)}) must be the same "
            "length and order-aligned (one AnswerResult per gold question, in order)"
        )

    per_query_candidates = [r.candidates for r in results]
    per_query_evidence = [g.evidence for g in gold_set]

    any_reranked = any(r.rerank_ms is not None for r in results)
    precision = precision_at_5(per_query_candidates, per_query_evidence) if any_reranked else None

    retrieve_p50, retrieve_p95 = latency_percentiles([r.retrieve_ms for r in results])
    rerank_values = [r.rerank_ms for r in results if r.rerank_ms is not None]
    rerank_p50, rerank_p95 = latency_percentiles(rerank_values) if rerank_values else (None, None)

    return Metrics(
        recall_at=recall_at_k(per_query_candidates, per_query_evidence),
        mrr=mrr(per_query_candidates, per_query_evidence),
        precision_at_5=precision,
        retrieve_latency_p50=retrieve_p50,
        retrieve_latency_p95=retrieve_p95,
        rerank_latency_p50=rerank_p50,
        rerank_latency_p95=rerank_p95,
        n_questions=len(gold_set),
        n_questions_scored=len(results),
    )
