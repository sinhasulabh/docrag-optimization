from atlasfin.config.schema import ChunkingConfig, ExperimentConfig, RerankingConfig, RetrievalConfig
from atlasfin.config.fingerprint import offline_fingerprint


def test_same_config_same_fingerprint():
    cfg1 = ExperimentConfig(name="a")
    cfg2 = ExperimentConfig(name="a")
    assert offline_fingerprint(cfg1) == offline_fingerprint(cfg2)


def test_online_only_change_keeps_fingerprint():
    """Acceptance criterion 3: changing retrieval/reranking must NOT change the fingerprint."""
    cfg1 = ExperimentConfig(name="baseline")
    cfg2 = ExperimentConfig(
        name="baseline_v2",
        retrieval=RetrievalConfig(mode="hybrid", top_k=50),
        reranking=RerankingConfig(enabled=True, model_id="x", depth=10),
    )
    assert offline_fingerprint(cfg1) == offline_fingerprint(cfg2)


def test_offline_change_changes_fingerprint():
    """Acceptance criterion 4: changing an offline knob MUST change the fingerprint."""
    cfg1 = ExperimentConfig(name="baseline")
    cfg2 = ExperimentConfig(name="baseline", chunking=ChunkingConfig(max_tokens=256))
    assert offline_fingerprint(cfg1) != offline_fingerprint(cfg2)


def test_fingerprint_ignores_name():
    cfg1 = ExperimentConfig(name="foo")
    cfg2 = ExperimentConfig(name="bar")
    assert offline_fingerprint(cfg1) == offline_fingerprint(cfg2)


def test_fingerprint_has_stable_prefix():
    cfg = ExperimentConfig(name="a")
    fp = offline_fingerprint(cfg)
    assert fp.startswith("v1-")
    assert len(fp) == len("v1-") + 16
