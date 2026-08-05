from dataclasses import dataclass
from pathlib import Path

import yaml

from .schema import (
    ChunkingConfig,
    EmbeddingConfig,
    ExperimentConfig,
    ParsingConfig,
    RerankingConfig,
    RetrievalConfig,
)
from .validation import validate_experiment_config


@dataclass
class ExperimentRunSpec:
    """ExperimentConfig plus the run-context fields (which docs, which gold set) that are
    NOT part of the spec's ExperimentConfig itself -- offline_fingerprint() must only ever
    depend on parsing/chunking/embedding, so which documents to index can't live on that
    dataclass without corrupting the fingerprint. This wrapper is what an experiment YAML
    file actually describes end to end.
    """

    config: ExperimentConfig
    ingest_run_uri: str
    doc_names: list[str]
    gold_set_path: str


def _sub_config(cls, data: dict | None):
    return cls(**(data or {}))


def load_experiment_config(path: str | Path) -> ExperimentRunSpec:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = ExperimentConfig(
        name=raw["name"],
        parsing=_sub_config(ParsingConfig, raw.get("parsing")),
        chunking=_sub_config(ChunkingConfig, raw.get("chunking")),
        embedding=_sub_config(EmbeddingConfig, raw.get("embedding")),
        retrieval=_sub_config(RetrievalConfig, raw.get("retrieval")),
        reranking=_sub_config(RerankingConfig, raw.get("reranking")),
    )
    validate_experiment_config(cfg)

    return ExperimentRunSpec(
        config=cfg,
        ingest_run_uri=raw["ingest_run_uri"],
        doc_names=list(raw["doc_names"]),
        gold_set_path=raw["gold_set_path"],
    )
