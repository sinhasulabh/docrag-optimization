from atlasfin.retrieval.fusion import rrf_fuse, weighted_fuse


def test_rrf_fuse_hand_computed_order():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 5.0), ("c", 4.0), ("d", 3.0)]
    fused = rrf_fuse([dense, sparse])
    # b: rank2(dense)+rank1(sparse); c: rank3(dense)+rank2(sparse); a: rank1(dense) only; d: rank3(sparse) only
    assert [cid for cid, _ in fused] == ["b", "c", "a", "d"]


def test_rrf_fuse_single_list_is_reciprocal_rank():
    fused = rrf_fuse([[("a", 1.0), ("b", 1.0)]])
    assert dict(fused)["a"] == 1.0 / 61
    assert dict(fused)["b"] == 1.0 / 62


def test_rrf_fuse_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


def test_weighted_fuse_hand_computed():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 5.0), ("c", 4.0), ("d", 3.0)]
    fused = dict(weighted_fuse([dense, sparse], [0.5, 0.5]))
    assert abs(fused["a"] - 0.5) < 1e-9  # dense-only, normalized to top of its own list
    assert abs(fused["b"] - 0.75) < 1e-9  # mid dense + top sparse
    assert abs(fused["c"] - 0.25) < 1e-9  # bottom dense + mid sparse
    assert abs(fused["d"] - 0.0) < 1e-9  # sparse-only, normalized to bottom of its own list


def test_weighted_fuse_mismatched_lengths_raises():
    import pytest

    with pytest.raises(ValueError):
        weighted_fuse([[("a", 1.0)]], [0.5, 0.5])
