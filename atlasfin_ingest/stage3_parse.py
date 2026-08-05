import json
import logging
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures import as_completed
from datetime import datetime, timezone
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from google.cloud import storage

from . import gcs
from .config import Config
from .manifest import ManifestWriter
from .models import ParsedObject, RawObject, Status

logger = logging.getLogger("atlasfin_ingest.stage3")

# PyTorch's MPS (and some CUDA/CPU model-loading paths) backend is not safe under concurrent
# inference from multiple threads in one process -- observed as a silent process crash (no
# Python exception) when parse_concurrency > 1 let two docling conversions run at once. Only
# the actual model-inference call is serialized here; downloading raw bytes and uploading
# results still overlap across parse_concurrency workers.
_INFERENCE_LOCK = threading.Lock()


def _parsed_blob_names(prefix: str, doc_name: str) -> tuple[str, str]:
    return f"{prefix}parsed/{doc_name}.docling.json", f"{prefix}parsed/{doc_name}.md"


def build_converter(cfg: Config) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = cfg.ocr
    pipeline_options.do_table_structure = True
    pipeline_options.accelerator_options = AcceleratorOptions(device=cfg.parse_device)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _resolve_docling_version(cfg: Config) -> str:
    if cfg.docling_version_pin:
        return cfg.docling_version_pin
    import docling

    return getattr(docling, "__version__", "unknown")


def _convert(converter: DocumentConverter, local_path: Path):
    with _INFERENCE_LOCK:
        return converter.convert(str(local_path)).document


def parse_one(
    raw: RawObject,
    bucket: str,
    prefix: str,
    cfg: Config,
    client: storage.Client,
    converter: DocumentConverter,
    scratch_dir: Path,
) -> ParsedObject:
    json_blob, md_blob = _parsed_blob_names(prefix, raw.doc_name)
    gcs_parsed_uri = f"gs://{bucket}/{json_blob}"
    docling_version = _resolve_docling_version(cfg)
    local_pdf = scratch_dir / f"{raw.doc_name}.raw.pdf"
    started = time.perf_counter()

    def _elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    def _failed(error: str) -> ParsedObject:
        return ParsedObject(
            doc_name=raw.doc_name,
            gcs_raw_uri=raw.gcs_raw_uri,
            gcs_parsed_uri=gcs_parsed_uri,
            page_count=0,
            docling_version=docling_version,
            parse_ms=_elapsed_ms(),
            parsed_at=datetime.now(timezone.utc),
            status=Status.FAILED,
            error=error,
        )

    try:
        _, _, blob_name = raw.gcs_raw_uri.partition(f"gs://{bucket}/")
        gcs.download_blob_to_file(client, bucket, blob_name, local_pdf)

        with ThreadPoolExecutor(max_workers=1) as one:
            fut = one.submit(_convert, converter, local_pdf)
            try:
                doc = fut.result(timeout=cfg.parse_timeout_s)
            except FutureTimeoutError:
                return _failed(f"parse timed out after {cfg.parse_timeout_s}s")

        page_count = len(getattr(doc, "pages", None) or [])

        local_json = scratch_dir / f"{raw.doc_name}.docling.json"
        try:
            local_json.write_text(json.dumps(doc.export_to_dict()))
            gcs.upload_file(client, bucket, json_blob, local_json, "application/json")
        finally:
            local_json.unlink(missing_ok=True)

        try:
            local_md = scratch_dir / f"{raw.doc_name}.md"
            local_md.write_text(doc.export_to_markdown())
            gcs.upload_file(client, bucket, md_blob, local_md, "text/markdown")
            local_md.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("stage3: %s markdown export failed (non-fatal): %s", raw.doc_name, exc)

        return ParsedObject(
            doc_name=raw.doc_name,
            gcs_raw_uri=raw.gcs_raw_uri,
            gcs_parsed_uri=gcs_parsed_uri,
            page_count=page_count,
            docling_version=docling_version,
            parse_ms=_elapsed_ms(),
            parsed_at=datetime.now(timezone.utc),
            status=Status.DONE,
        )
    except Exception as exc:
        logger.error("stage3: %s failed: %s", raw.doc_name, exc)
        return _failed(str(exc))
    finally:
        local_pdf.unlink(missing_ok=True)


def parse_all(
    raws: list[RawObject], bucket: str, prefix: str, cfg: Config, manifest: ManifestWriter
) -> list[ParsedObject]:
    client = gcs.get_client(cfg)
    converter = build_converter(cfg)
    scratch_dir = Path(cfg.local_scratch_dir or tempfile.gettempdir()) / "atlasfin-ingest"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    eligible = [r for r in raws if r.status in (Status.DONE, Status.SKIPPED)]
    results: list[ParsedObject | None] = [None] * len(eligible)

    with ThreadPoolExecutor(max_workers=cfg.parse_concurrency) as pool:
        futures = {
            pool.submit(parse_one, raw, bucket, prefix, cfg, client, converter, scratch_dir): i
            for i, raw in enumerate(eligible)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            parsed = fut.result()
            results[i] = parsed
            manifest.append(parsed)

    finished: list[ParsedObject] = [r for r in results if r is not None]
    n_done = sum(1 for r in finished if r.status == Status.DONE)
    n_failed = sum(1 for r in finished if r.status == Status.FAILED)
    logger.info(
        "stage3: done=%d failed=%d (skipped upstream=%d)",
        n_done,
        n_failed,
        len(raws) - len(eligible),
    )
    return finished
