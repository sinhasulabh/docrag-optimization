import argparse
import dataclasses
import json
import logging
import sys

from atlasfin.config.loader import load_experiment_config
from atlasfin.eval.gold import load_gold_set

from .runner import run_experiment


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AtlasFin retrieval experiment runner")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run an experiment config against its gold set")
    run_p.add_argument("--config", required=True, help="path to an experiment YAML file")
    run_p.add_argument(
        "--gcp-project",
        default=None,
        help="omit to skip Vertex AI Vector Search entirely (offline artifacts only, no "
        "dense retrieval online step without also deploying)",
    )
    run_p.add_argument("--gcp-location", default="us-central1")
    run_p.add_argument(
        "--parse-device", default="auto", choices=["auto", "cpu", "cuda", "mps", "xpu"]
    )

    return p


def main() -> None:
    _configure_logging()
    args = build_arg_parser().parse_args()
    logger = logging.getLogger("atlasfin.pipeline.cli")

    if args.command == "run":
        spec = load_experiment_config(args.config)
        gold_set = load_gold_set(spec.gold_set_path, doc_names=spec.doc_names)
        if not gold_set:
            logger.error(
                "no gold questions grounded in doc_names=%s, nothing to score", spec.doc_names
            )
            sys.exit(1)

        metrics = run_experiment(
            spec.config,
            gold_set,
            spec.doc_names,
            ingest_run_uri=spec.ingest_run_uri,
            gcp_project=args.gcp_project,
            gcp_location=args.gcp_location,
            parse_device=args.parse_device,
        )
        logger.info("pipeline.cli: %s scored %d/%d questions", spec.config.name, metrics.n_questions_scored, metrics.n_questions)
        print(json.dumps(dataclasses.asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
