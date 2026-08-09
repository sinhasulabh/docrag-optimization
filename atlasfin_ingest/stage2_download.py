import hashlib
import logging
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from google.cloud import storage

from . import gcs
from .config import Config
from .manifest import ManifestWriter
from .models import RawObject, SourceRecord, Status

logger = logging.getLogger("atlasfin_ingest.stage2")

RETRY_STATUS = {429, 500, 502, 503, 504}
PDF_MAGIC = b"%PDF"


class HostRateLimiter:
    """Enforces a minimum gap between requests to the same host (SEC EDGAR fair-access etc.)."""

    def __init__(self, min_interval_s: float):
        self._min_interval = min_interval_s
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            last = self._last_call.get(host, 0.0)
            wait_s = self._min_interval - (now - last)
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_call[host] = time.monotonic()


def _raw_blob_name(prefix: str, rec: SourceRecord) -> str:
    return f"{prefix}raw/{rec.doc_type}/{rec.doc_name}.pdf"


def _stream_download(
    url: str, dest: Path, cfg: Config, limiter: HostRateLimiter
) -> tuple[int, str, str, int]:
    """Streams url to dest, hashing in a single pass. Returns (http_status, content_type, sha256, size_bytes)."""
    host = urlparse(url).netloc
    headers = {"User-Agent": cfg.user_agent}
    max_bytes = cfg.dl_max_size_mb * 1024 * 1024
    last_exc: Exception | None = None

    for attempt in range(cfg.dl_max_retries + 1):
        limiter.wait(host)
        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=cfg.dl_timeout_s, allow_redirects=True
            ) as resp:
                if resp.status_code in RETRY_STATUS and attempt < cfg.dl_max_retries:
                    backoff = (2**attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        "stage2: %s -> HTTP %d, retrying in %.1fs (attempt %d/%d)",
                        url,
                        resp.status_code,
                        backoff,
                        attempt + 1,
                        cfg.dl_max_retries,
                    )
                    time.sleep(backoff)
                    continue

                resp.raise_for_status()

                sha256 = hashlib.sha256()
                size = 0
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=256 * 1024):
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(
                                f"download exceeded dl_max_size_mb={cfg.dl_max_size_mb}"
                            )
                        sha256.update(chunk)
                        f.write(chunk)
                return resp.status_code, resp.headers.get("Content-Type", ""), sha256.hexdigest(), size
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < cfg.dl_max_retries:
                backoff = (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "stage2: %s -> %s, retrying in %.1fs (attempt %d/%d)",
                    url,
                    exc,
                    backoff,
                    attempt + 1,
                    cfg.dl_max_retries,
                )
                time.sleep(backoff)
                continue
            raise

    raise last_exc or RuntimeError("download failed after retries")


def download_one(
    rec: SourceRecord,
    bucket: str,
    prefix: str,
    cfg: Config,
    client: storage.Client,
    limiter: HostRateLimiter,
    scratch_dir: Path,
) -> RawObject:
    blob_name = _raw_blob_name(prefix, rec)
    gcs_raw_uri = f"gs://{bucket}/{blob_name}"

    existing = gcs.get_blob_metadata(client, bucket, blob_name)
    if existing and existing["metadata"].get("sha256"):
        logger.info("stage2: %s already present at %s, skipping", rec.doc_name, gcs_raw_uri)
        return RawObject(
            doc_name=rec.doc_name,
            source_uri=rec.doc_link,
            gcs_raw_uri=gcs_raw_uri,
            sha256=existing["metadata"]["sha256"],
            size_bytes=int(existing["size"] or 0),
            content_type=existing["content_type"] or "application/pdf",
            http_status=200,
            fetched_at=datetime.now(timezone.utc),
            status=Status.SKIPPED,
        )

    tmp_path = scratch_dir / f"{rec.doc_name}.download.tmp"
    http_status = 0
    try:
        http_status, content_type, sha256, size = _stream_download(rec.doc_link, tmp_path, cfg, limiter)

        if size == 0:
            raise ValueError("downloaded 0 bytes")

        with tmp_path.open("rb") as f:
            head = f.read(4)
        looks_pdf = head == PDF_MAGIC or "application/pdf" in content_type.lower()
        if not looks_pdf:
            raise ValueError(f"not a PDF (content-type={content_type!r}, magic={head!r})")

        gcs.upload_file(
            client,
            bucket,
            blob_name,
            tmp_path,
            "application/pdf",
            metadata={
                "sha256": sha256,
                "source_uri": rec.doc_link,
                "doc_name": rec.doc_name,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return RawObject(
            doc_name=rec.doc_name,
            source_uri=rec.doc_link,
            gcs_raw_uri=gcs_raw_uri,
            sha256=sha256,
            size_bytes=size,
            content_type="application/pdf",
            http_status=http_status,
            fetched_at=datetime.now(timezone.utc),
            status=Status.DONE,
        )
    except Exception as exc:
        status_from_exc = getattr(getattr(exc, "response", None), "status_code", None)
        logger.error("stage2: %s failed: %s", rec.doc_name, exc)
        return RawObject(
            doc_name=rec.doc_name,
            source_uri=rec.doc_link,
            gcs_raw_uri=gcs_raw_uri,
            sha256="",
            size_bytes=0,
            content_type="",
            http_status=status_from_exc or http_status,
            fetched_at=datetime.now(timezone.utc),
            status=Status.FAILED,
            error=str(exc),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def download_all(
    records: list[SourceRecord], bucket: str, prefix: str, cfg: Config, manifest: ManifestWriter
) -> list[RawObject]:
    client = gcs.get_client(cfg)
    limiter = HostRateLimiter(cfg.dl_min_host_interval_s)
    scratch_dir = Path(cfg.local_scratch_dir or tempfile.gettempdir()) / "atlasfin-ingest"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    results: list[RawObject | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=cfg.dl_concurrency) as pool:
        futures = {
            pool.submit(download_one, rec, bucket, prefix, cfg, client, limiter, scratch_dir): i
            for i, rec in enumerate(records)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            raw = fut.result()
            results[i] = raw
            manifest.append(raw)
            manifest.flush()  # incremental -- a crash mid-stage must not lose already-recorded progress

    finished: list[RawObject] = [r for r in results if r is not None]
    n_done = sum(1 for r in finished if r.status == Status.DONE)
    n_skipped = sum(1 for r in finished if r.status == Status.SKIPPED)
    n_failed = sum(1 for r in finished if r.status == Status.FAILED)
    logger.info("stage2: done=%d skipped=%d failed=%d", n_done, n_skipped, n_failed)
    return finished
