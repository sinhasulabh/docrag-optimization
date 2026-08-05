import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from atlasfin.config.schema import ExperimentConfig
from atlasfin.eval import GoldRecord, Metrics, score

from . import cache
from .answer import AnswerResult, answer
from .offline import build_offline
from .online import build_online

logger = logging.getLogger("atlasfin.pipeline.runner")


def run_experiment(
    cfg: ExperimentConfig,
    gold_set: list[GoldRecord],
    doc_names: list[str],
    *,
    ingest_run_uri: str,
    gcp_project: str | None = None,
    gcp_location: str = "us-central1",
    parse_device: str = "auto",
) -> Metrics:
    index = build_offline(
        cfg,
        doc_names,
        ingest_run_uri=ingest_run_uri,
        gcp_project=gcp_project,
        gcp_location=gcp_location,
        parse_device=parse_device,
    )
    online = build_online(cfg, index, gcp_project=gcp_project, gcp_location=gcp_location)

    results: list[AnswerResult] = [answer(g.question, online, cfg) for g in gold_set]
    metrics = score(results, gold_set)

    _write_run_artifacts(cfg, gold_set, results, metrics)
    return metrics


def _write_run_artifacts(
    cfg: ExperimentConfig,
    gold_set: list[GoldRecord],
    results: list[AnswerResult],
    metrics: Metrics,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    run_dir = cache.runs_dir(cfg.name, ts)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2))
    with (run_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for gold, result in zip(gold_set, results):
            f.write(
                json.dumps(
                    {
                        "financebench_id": gold.financebench_id,
                        "question": gold.question,
                        "retrieve_ms": result.retrieve_ms,
                        "rerank_ms": result.rerank_ms,
                        "top_chunk_ids": [c.chunk_id for c in result.candidates[:5]],
                    }
                )
                + "\n"
            )
    logger.info("pipeline.runner: wrote run artifacts to %s", run_dir)
