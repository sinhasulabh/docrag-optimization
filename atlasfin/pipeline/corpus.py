import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

from atlasfin_ingest import gcs as ingest_gcs
from atlasfin.config.schema import ParsingConfig
from atlasfin.contracts import ParsedObject, RawObject, SourceRecord, Status
from atlasfin.parsing.build import build as build_parser

logger = logging.getLogger("atlasfin.pipeline.corpus")


def split_run_uri(ingest_run_uri: str) -> tuple[str, str]:
    if not ingest_run_uri.startswith("gs://"):
        raise ValueError(f"ingest_run_uri must be a gs:// URI, got {ingest_run_uri!r}")
    _, _, rest = ingest_run_uri.partition("gs://")
    bucket, _, prefix = rest.partition("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def _load_run_manifest(
    client: storage.Client, bucket: str, prefix: str, scratch_dir: Path
) -> list[dict]:
    blob_name = f"{prefix}_manifest/run_manifest.jsonl"
    if ingest_gcs.get_blob_metadata(client, bucket, blob_name) is None:
        return []
    local_path = scratch_dir / "run_manifest.jsonl"
    ingest_gcs.download_blob_to_file(client, bucket, blob_name, local_path)
    rows = []
    with local_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    local_path.unlink(missing_ok=True)
    return rows


def _parsed_object_from_row(row: dict) -> ParsedObject:
    return ParsedObject(
        doc_name=row["doc_name"],
        gcs_raw_uri=row["gcs_raw_uri"],
        gcs_parsed_uri=row["gcs_parsed_uri"],
        page_count=row["page_count"],
        docling_version=row["docling_version"],
        parse_ms=row["parse_ms"],
        parsed_at=datetime.fromisoformat(row["parsed_at"]),
        status=Status(row["status"]),
        error=row.get("error"),
    )


def _raw_object_from_row(row: dict) -> RawObject:
    return RawObject(
        doc_name=row["doc_name"],
        source_uri=row["source_uri"],
        gcs_raw_uri=row["gcs_raw_uri"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        content_type=row["content_type"],
        http_status=row["http_status"],
        fetched_at=datetime.fromisoformat(row["fetched_at"]),
        status=Status(row["status"]),
        error=row.get("error"),
    )


def resolve_parsed_docs(
    doc_names: list[str],
    *,
    ingest_run_uri: str,
    source_lookup: dict[str, SourceRecord],
    gcs_client: storage.Client,
    parsing_cfg: ParsingConfig,
    scratch_dir: Path,
    parse_device: str = "auto",
    parse_timeout_s: int = 300,
) -> list[ParsedObject]:
    """Fast path: doc already parsed under ingest_run_uri (reuses the real ingestion output,
    resolved via the existing run_manifest.jsonl -- no re-parse). Slow path: falls back to a
    live Docling parse via parsing.build() for any doc not already there.

    KNOWN LIMITATION: the fast path does not verify the existing GCS parse actually used the
    same ParsingConfig this experiment specifies (atlasfin_ingest doesn't snapshot its
    effective config today). Documented assumption: an experiment pointed at an existing
    ingest_run_uri must set ParsingConfig to match what that ingest run actually used.
    """
    bucket, prefix = split_run_uri(ingest_run_uri)
    manifest_rows = _load_run_manifest(gcs_client, bucket, prefix, scratch_dir)
    parsed_by_doc = {r["doc_name"]: r for r in manifest_rows if "docling_version" in r}
    raw_by_doc = {r["doc_name"]: r for r in manifest_rows if "sha256" in r}

    parser = None
    results: list[ParsedObject] = []
    for doc_name in doc_names:
        row = parsed_by_doc.get(doc_name)
        parsed_blob_name = f"{prefix}parsed/{doc_name}.docling.json"
        if row is not None and ingest_gcs.get_blob_metadata(gcs_client, bucket, parsed_blob_name):
            results.append(_parsed_object_from_row(row))
            continue

        logger.info("corpus: %s not pre-parsed under %s, parsing live", doc_name, ingest_run_uri)
        raw = resolve_raw_doc(
            doc_name,
            bucket=bucket,
            prefix=prefix,
            raw_by_doc=raw_by_doc,
            source_lookup=source_lookup,
            gcs_client=gcs_client,
        )
        if parser is None:
            parser = build_parser(
                parsing_cfg,
                gcs_client=gcs_client,
                bucket=bucket,
                prefix=prefix,
                scratch_dir=scratch_dir,
                parse_device=parse_device,
                parse_timeout_s=parse_timeout_s,
            )
        parsed = parser.parse(raw)
        if parsed.status != Status.DONE:
            raise RuntimeError(f"live parse failed for {doc_name}: {parsed.error}")
        results.append(parsed)
    return results


def resolve_raw_doc(
    doc_name: str,
    *,
    bucket: str,
    prefix: str,
    raw_by_doc: dict[str, dict],
    source_lookup: dict[str, SourceRecord],
    gcs_client: storage.Client,
) -> RawObject:
    row = raw_by_doc.get(doc_name)
    if row is not None:
        return _raw_object_from_row(row)

    source = source_lookup.get(doc_name)
    if source is None:
        raise KeyError(f"no SourceRecord for doc_name={doc_name!r} to resolve a raw path")
    raw_blob_name = f"{prefix}raw/{source.doc_type}/{doc_name}.pdf"
    if ingest_gcs.get_blob_metadata(gcs_client, bucket, raw_blob_name) is None:
        raise FileNotFoundError(
            f"no raw or parsed object found for {doc_name!r} under gs://{bucket}/{prefix}"
        )
    return RawObject(
        doc_name=doc_name,
        source_uri=source.doc_link,
        gcs_raw_uri=f"gs://{bucket}/{raw_blob_name}",
        sha256="",
        size_bytes=0,
        content_type="application/pdf",
        http_status=200,
        fetched_at=datetime.now(timezone.utc),
        status=Status.DONE,
    )
