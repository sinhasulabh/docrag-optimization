from pathlib import Path

import pytest

from atlasfin.config.fingerprint import offline_fingerprint
from atlasfin.config.loader import load_experiment_config
from atlasfin.config.schema import ExperimentConfig, RerankingConfig
from atlasfin.config.validation import validate_experiment_config

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "atlasfin" / "experiments" / "configs"


def test_load_baseline_dense_3m():
    spec = load_experiment_config(CONFIGS_DIR / "baseline_dense_3m.yaml")
    assert spec.config.name == "baseline_dense_3m"
    assert len(spec.doc_names) == 8
    assert spec.config.retrieval.mode == "dense"
    assert spec.config.retrieval.index_type == "flat"


def test_load_hybrid_rerank_3m():
    spec = load_experiment_config(CONFIGS_DIR / "hybrid_rerank_3m.yaml")
    assert spec.config.retrieval.mode == "hybrid"
    assert spec.config.reranking.enabled is True


def test_the_two_example_configs_share_offline_fingerprint():
    baseline = load_experiment_config(CONFIGS_DIR / "baseline_dense_3m.yaml")
    hybrid = load_experiment_config(CONFIGS_DIR / "hybrid_rerank_3m.yaml")
    assert offline_fingerprint(baseline.config) == offline_fingerprint(hybrid.config)


def test_validation_rejects_depth_exceeding_top_k():
    cfg = ExperimentConfig(name="bad", reranking=RerankingConfig(enabled=True, model_id="x", depth=999))
    with pytest.raises(ValueError):
        validate_experiment_config(cfg)


def test_validation_accepts_defaults():
    validate_experiment_config(ExperimentConfig(name="ok"))
