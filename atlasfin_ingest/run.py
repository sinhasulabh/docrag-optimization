import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from .config import Config
from .gcs import build_run_id, ensure_bucket, get_client, object_prefix
from .manifest import ManifestWriter
from .models import Status
from .stage1_source import SourceFilter, read_source
from .stage2_download import download_all
from .stage3_parse import parse_all


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AtlasFin ingestion pipeline (Stages 1-3)")
    p.add_argument("--source-uri", required=True, help="Path/URI to the source JSONL")
    p.add_argument("--project-name", required=True, help="-> bucket name prefix, e.g. atlasfin")
    p.add_argument("--gcs-project", required=True, help="GCP project id")
    p.add_argument("--gcs-location", default="us-central1")
    p.add_argument("--bucket-strategy", choices=["per_run", "stable"], default="stable")
    p.add_argument("--env", default="dev")
    p.add_argument("--dl-timeout-s", type=int, default=30)
    p.add_argument("--dl-max-retries", type=int, default=4)
    p.add_argument("--dl-max-size-mb", type=int, default=100)
    p.add_argument("--dl-concurrency", type=int, default=12)
    p.add_argument("--dl-min-host-interval-s", type=float, default=0.5)
    p.add_argument("--user-agent", default="AtlasFin-ingest/0.1 (you@example.com)")
    p.add_argument("--ocr", action="store_true", help="enable OCR (default off for born-digital PDFs)")
    p.add_argument("--parse-concurrency", type=int, default=2)
    p.add_argument("--parse-timeout-s", type=int, default=300)
    p.add_argument(
        "--parse-device",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        default="auto",
        help="force a specific accelerator for docling model inference (default: auto)",
    )
    p.add_argument("--docling-version-pin", default=None)
    p.add_argument("--doc-types", nargs="*", default=None)
    p.add_argument("--companies", nargs="*", default=None)
    p.add_argument("--doc-names", nargs="*", default=None)
    p.add_argument("--local-scratch-dir", default=None)
    p.add_argument(
        "--run-id",
        default=None,
        help=(
            "resume an existing run instead of starting a new one -- reuses its "
            "bucket/prefix so already-downloaded/parsed docs are skipped and only "
            "missing ones are (re)attempted, e.g. 20260808t221108z"
        ),
    )
    return p


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        project_name=args.project_name,
        source_uri=args.source_uri,
        gcs_project=args.gcs_project,
        gcs_location=args.gcs_location,
        bucket_strategy=args.bucket_strategy,
        env=args.env,
        dl_timeout_s=args.dl_timeout_s,
        dl_max_retries=args.dl_max_retries,
        dl_max_size_mb=args.dl_max_size_mb,
        dl_concurrency=args.dl_concurrency,
        dl_min_host_interval_s=args.dl_min_host_interval_s,
        user_agent=args.user_agent,
        ocr=args.ocr,
        parse_concurrency=args.parse_concurrency,
        parse_timeout_s=args.parse_timeout_s,
        parse_device=args.parse_device,
        docling_version_pin=args.docling_version_pin,
        filter_doc_types=args.doc_types,
        filter_companies=args.companies,
        filter_doc_names=args.doc_names,
        local_scratch_dir=args.local_scratch_dir,
    )


def run(cfg: Config, run_id: str | None = None) -> int:
    logger = logging.getLogger("atlasfin_ingest.run")
    resuming = run_id is not None
    run_id = run_id or build_run_id()
    logger.info(
        "run: %s run_id=%s bucket_strategy=%s",
        "resuming" if resuming else "starting",
        run_id,
        cfg.bucket_strategy,
    )

    flt = SourceFilter(
        doc_types=cfg.filter_doc_types,
        companies=cfg.filter_companies,
        doc_names=cfg.filter_doc_names,
    )
    records = read_source(cfg.source_uri, flt)
    if not records:
        logger.info("run: 0 source records after filtering, nothing to do")
        return 0

    bucket = ensure_bucket(cfg, run_id)
    prefix = object_prefix(cfg, run_id)
    client = get_client(cfg)

    local_dir = Path(tempfile.mkdtemp(prefix=f"atlasfin-run-{run_id}-"))
    manifest = ManifestWriter(client, bucket, prefix, local_dir)
    manifest.write_source_snapshot(records)

    raws = download_all(records, bucket, prefix, cfg, manifest)
    manifest.flush()

    parsed = parse_all(raws, bucket, prefix, cfg, manifest)
    manifest.flush()

    summary = {
        "run_id": run_id,
        "bucket": bucket,
        "prefix": prefix,
        "attempted": len(records),
        "downloaded": sum(1 for r in raws if r.status == Status.DONE),
        "skipped": sum(1 for r in raws if r.status == Status.SKIPPED),
        "download_failed": sum(1 for r in raws if r.status == Status.FAILED),
        "parsed": sum(1 for p in parsed if p.status == Status.DONE),
        "parse_failed": sum(1 for p in parsed if p.status == Status.FAILED),
        "failures": (
            [
                {"doc_name": r.doc_name, "stage": "download", "error": r.error}
                for r in raws
                if r.status == Status.FAILED
            ]
            + [
                {"doc_name": p.doc_name, "stage": "parse", "error": p.error}
                for p in parsed
                if p.status == Status.FAILED
            ]
        ),
    }
    manifest.write_summary(summary)
    logger.info("run: summary %s", json.dumps(summary))

    any_failed = summary["download_failed"] > 0 or summary["parse_failed"] > 0
    return 1 if any_failed else 0


def main() -> None:
    _configure_logging()
    args = build_arg_parser().parse_args()
    cfg = build_config(args)
    sys.exit(run(cfg, run_id=args.run_id))


if __name__ == "__main__":
    main()
