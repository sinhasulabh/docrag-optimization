from dataclasses import dataclass

import numpy as np

from atlasfin.contracts import Candidate, parse_chunk_id

from .gold import GoldEvidence

RECALL_CUTOFFS = (5, 10, 20)
PRECISION_CUTOFF = 5


@dataclass(frozen=True)
class Metrics:
    recall_at: dict[int, float]
    mrr: float
    precision_at_5: float | None  # None if reranking didn't run for this scoring pass
    retrieve_latency_p50: float
    retrieve_latency_p95: float
    rerank_latency_p50: float | None
    rerank_latency_p95: float | None
    n_questions: int
    n_questions_scored: int


def _is_hit(candidate: Candidate, evidence: list[GoldEvidence]) -> bool:
    doc_name, _, _ = parse_chunk_id(candidate.chunk_id)
    return any(e.doc_name == doc_name and e.evidence_page_num in candidate.pages for e in evidence)


def recall_at_k(
    per_query_candidates: list[list[Candidate]],
    per_query_evidence: list[list[GoldEvidence]],
    cutoffs: tuple[int, ...] = RECALL_CUTOFFS,
) -> dict[int, float]:
    """A question 'hits' at cutoff k if ANY of its gold evidence pages appears among the
    top-k candidates' pages, for a candidate from the matching doc_name. "Any evidence item"
    (not "all") -- matches FinanceBench's typical single/multi-hop structure.
    """
    n = len(per_query_candidates)
    result: dict[int, float] = {}
    for k in cutoffs:
        hits = sum(
            1
            for cands, evidence in zip(per_query_candidates, per_query_evidence)
            if any(_is_hit(c, evidence) for c in cands[:k])
        )
        result[k] = hits / n if n else 0.0
    return result


def mrr(
    per_query_candidates: list[list[Candidate]], per_query_evidence: list[list[GoldEvidence]]
) -> float:
    n = len(per_query_candidates)
    if n == 0:
        return 0.0
    total = 0.0
    for cands, evidence in zip(per_query_candidates, per_query_evidence):
        for rank, c in enumerate(cands, start=1):
            if _is_hit(c, evidence):
                total += 1.0 / rank
                break
    return total / n


def precision_at_5(
    per_query_candidates: list[list[Candidate]], per_query_evidence: list[list[GoldEvidence]]
) -> float | None:
    n = len(per_query_candidates)
    if n == 0:
        return None
    total = 0.0
    for cands, evidence in zip(per_query_candidates, per_query_evidence):
        top = cands[:PRECISION_CUTOFF]
        if not top:
            continue
        total += sum(1 for c in top if _is_hit(c, evidence)) / len(top)
    return total / n


def latency_percentiles(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    p50, p95 = np.percentile(values, [50, 95])
    return float(p50), float(p95)
