from atlasfin.contracts import Candidate, make_chunk_id
from atlasfin.eval.gold import GoldEvidence
from atlasfin.eval.metrics import latency_percentiles, mrr, precision_at_5, recall_at_k


def _cand(doc_name: str, pages: list[int], score: float = 1.0) -> Candidate:
    return Candidate(
        chunk_id=make_chunk_id(doc_name, "s", f"{doc_name}-{pages}-{score}"),
        score=score,
        text="",
        payload_text="",
        pages=pages,
        metadata={},
    )


def test_recall_at_k_hit_and_miss():
    evidence = [GoldEvidence(doc_name="D", evidence_page_num=5, evidence_text="")]
    hit_candidates = [_cand("D", [1]), _cand("D", [5]), _cand("D", [9])]
    miss_candidates = [_cand("D", [1]), _cand("D", [2]), _cand("D", [3])]

    recall = recall_at_k([hit_candidates, miss_candidates], [evidence, evidence], cutoffs=(2, 5))
    assert recall[2] == 0.5  # hit is at rank 2 for the first query, no hit within top-2 for second
    assert recall[5] == 0.5  # miss_candidates never contains page 5 at all


def test_recall_requires_matching_doc_name():
    """A candidate with the right page but the WRONG doc must not count as a hit."""
    evidence = [GoldEvidence(doc_name="D", evidence_page_num=5, evidence_text="")]
    wrong_doc_candidates = [_cand("OTHER_DOC", [5])]
    recall = recall_at_k([wrong_doc_candidates], [evidence], cutoffs=(5,))
    assert recall[5] == 0.0


def test_mrr_reciprocal_of_first_hit_rank():
    evidence = [GoldEvidence(doc_name="D", evidence_page_num=5, evidence_text="")]
    candidates = [_cand("D", [1]), _cand("D", [2]), _cand("D", [5]), _cand("D", [5])]
    assert abs(mrr([candidates], [evidence]) - (1 / 3)) < 1e-9


def test_mrr_zero_when_no_hit():
    evidence = [GoldEvidence(doc_name="D", evidence_page_num=5, evidence_text="")]
    candidates = [_cand("D", [1]), _cand("D", [2])]
    assert mrr([candidates], [evidence]) == 0.0


def test_precision_at_5():
    evidence = [GoldEvidence(doc_name="D", evidence_page_num=5, evidence_text="")]
    candidates = [_cand("D", [5]), _cand("D", [5]), _cand("D", [1]), _cand("D", [1]), _cand("D", [1])]
    assert abs(precision_at_5([candidates], [evidence]) - (2 / 5)) < 1e-9


def test_precision_at_5_empty_returns_none():
    assert precision_at_5([], []) is None


def test_latency_percentiles_empty():
    assert latency_percentiles([]) == (0.0, 0.0)


def test_latency_percentiles_basic():
    p50, p95 = latency_percentiles([100.0, 200.0, 300.0, 400.0, 500.0])
    assert p50 == 300.0
    assert p95 > p50
