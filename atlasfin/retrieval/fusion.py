from collections import defaultdict

# The standard constant from Cormack et al. / used by Elasticsearch's own RRF -- not exposed
# as a config knob, just a sane fixed default.
RRF_K = 60


def rrf_fuse(ranked_lists: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over the union of one or more ranked (chunk_id, score) lists.
    Pure function, no I/O -- reused identically by hybrid.py (dense+sparse) and by
    query_transform.py's TransformingRetriever (fusing multiple decomposed sub-queries).
    """
    fused: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked, start=1):
            fused[chunk_id] += 1.0 / (RRF_K + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def weighted_fuse(
    ranked_lists: list[list[tuple[str, float]]], weights: list[float]
) -> list[tuple[str, float]]:
    """Min-max normalizes each list's raw scores to [0,1] independently first -- dense
    cosine/dot scores and BM25 scores are on incomparable scales, so blending raw scores
    without normalization would be meaningless -- then combines by `weights`.
    """
    if len(ranked_lists) != len(weights):
        raise ValueError("ranked_lists and weights must be the same length")

    combined: dict[str, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        if not ranked:
            continue
        values = [score for _, score in ranked]
        lo, hi = min(values), max(values)
        span = hi - lo
        for chunk_id, score in ranked:
            normalized = (score - lo) / span if span > 0 else 1.0
            combined[chunk_id] += weight * normalized
    return sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
