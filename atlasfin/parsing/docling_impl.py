import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from google.cloud import storage

from atlasfin_ingest import gcs as ingest_gcs

# Reuse the EXACT lock object the ingestion package already derived from a real MPS-crash
# debugging session (see atlasfin_ingest/stage3_parse.py) rather than re-deriving a second
# one -- two independent locks would not actually serialize inference against each other if
# both packages ever ran docling conversions in the same process.
from atlasfin_ingest.stage3_parse import _INFERENCE_LOCK
from atlasfin.config.schema import ParsingConfig
from atlasfin.contracts import ParsedObject, RawObject, Status

logger = logging.getLogger("atlasfin.parsing.docling_impl")

DEFAULT_PARSE_TIMEOUT_S = 300
DownloadFn = Callable[[storage.Client, str, str, Path], None]


def _parsed_blob_names(prefix: str, doc_name: str) -> tuple[str, str]:
    return f"{prefix}parsed/{doc_name}.docling.json", f"{prefix}parsed/{doc_name}.md"


def _pipeline_options(cfg: ParsingConfig, parse_device: str) -> PdfPipelineOptions:
    opts = PdfPipelineOptions()
    if cfg.ocr == "off":
        opts.do_ocr = False
    elif cfg.ocr == "on":
        opts.do_ocr = True
        opts.ocr_options.force_full_page_ocr = True
    elif cfg.ocr == "auto":
        opts.do_ocr = True
        opts.ocr_options.force_full_page_ocr = False
    else:
        raise ValueError(f"parsing.ocr must be off|on|auto, got {cfg.ocr!r}")

    opts.do_table_structure = cfg.table_mode == "structured"
    opts.generate_page_images = cfg.keep_images
    opts.generate_picture_images = cfg.keep_images
    opts.accelerator_options = AcceleratorOptions(device=parse_device)
    return opts


def build_converter(cfg: ParsingConfig, parse_device: str = "auto") -> DocumentConverter:
    if cfg.backend != "docling":
        raise ValueError(f"parsing.backend={cfg.backend!r} not supported (only 'docling')")
    pipeline_options = _pipeline_options(cfg, parse_device)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _convert_locked(converter: DocumentConverter, local_path: Path):
    with _INFERENCE_LOCK:
        return converter.convert(str(local_path)).document


class DoclingParser:
    """Parser implementation wrapping Docling. Constructed with the GCS/scratch context a
    single Parser.parse(raw) call needs, since the Protocol itself takes only `raw`.
    """

    def __init__(
        self,
        cfg: ParsingConfig,
        *,
        gcs_client: storage.Client,
        bucket: str,
        prefix: str,
        scratch_dir: Path,
        download_fn: DownloadFn | None = None,
        parse_device: str = "auto",
        parse_timeout_s: int = DEFAULT_PARSE_TIMEOUT_S,
    ):
        self._cfg = cfg
        self._client = gcs_client
        self._bucket = bucket
        self._prefix = prefix
        self._scratch_dir = Path(scratch_dir)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self._download_fn: DownloadFn = download_fn or ingest_gcs.download_blob_to_file
        self._parse_timeout_s = parse_timeout_s
        self._converter = build_converter(cfg, parse_device)

    def parse(self, raw: RawObject) -> ParsedObject:
        json_blob, md_blob = _parsed_blob_names(self._prefix, raw.doc_name)
        gcs_parsed_uri = f"gs://{self._bucket}/{json_blob}"
        docling_version = _installed_docling_version()
        local_pdf = self._scratch_dir / f"{raw.doc_name}.raw.pdf"
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        def failed(error: str) -> ParsedObject:
            return ParsedObject(
                doc_name=raw.doc_name,
                gcs_raw_uri=raw.gcs_raw_uri,
                gcs_parsed_uri=gcs_parsed_uri,
                page_count=0,
                docling_version=docling_version,
                parse_ms=elapsed_ms(),
                parsed_at=datetime.now(timezone.utc),
                status=Status.FAILED,
                error=error,
            )

        try:
            _, _, blob_name = raw.gcs_raw_uri.partition(f"gs://{self._bucket}/")
            self._download_fn(self._client, self._bucket, blob_name, local_pdf)

            with ThreadPoolExecutor(max_workers=1) as one:
                fut = one.submit(_convert_locked, self._converter, local_pdf)
                try:
                    doc = fut.result(timeout=self._parse_timeout_s)
                except FutureTimeoutError:
                    return failed(f"parse timed out after {self._parse_timeout_s}s")

            page_count = len(getattr(doc, "pages", None) or [])

            local_json = self._scratch_dir / f"{raw.doc_name}.docling.json"
            try:
                local_json.write_text(json.dumps(doc.export_to_dict()))
                ingest_gcs.upload_file(
                    self._client, self._bucket, json_blob, local_json, "application/json"
                )
            finally:
                local_json.unlink(missing_ok=True)

            try:
                local_md = self._scratch_dir / f"{raw.doc_name}.md"
                local_md.write_text(doc.export_to_markdown())
                ingest_gcs.upload_file(
                    self._client, self._bucket, md_blob, local_md, "text/markdown"
                )
                local_md.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(
                    "parsing: %s markdown export failed (non-fatal): %s", raw.doc_name, exc
                )

            return ParsedObject(
                doc_name=raw.doc_name,
                gcs_raw_uri=raw.gcs_raw_uri,
                gcs_parsed_uri=gcs_parsed_uri,
                page_count=page_count,
                docling_version=docling_version,
                parse_ms=elapsed_ms(),
                parsed_at=datetime.now(timezone.utc),
                status=Status.DONE,
            )
        except Exception as exc:
            logger.error("parsing: %s failed: %s", raw.doc_name, exc)
            return failed(str(exc))
        finally:
            local_pdf.unlink(missing_ok=True)


def _installed_docling_version() -> str:
    import docling

    return getattr(docling, "__version__", "unknown")
