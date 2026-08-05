import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from google.api_core.exceptions import Conflict
from google.cloud import storage

from .config import Config

logger = logging.getLogger("atlasfin_ingest.gcs")

# lowercase letters, digits, hyphens, underscores, dots; must start/end with letter or digit
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-_.]{1,61}[a-z0-9]$")


def build_run_id() -> str:
    """Compact UTC timestamp, e.g. 20260731t203045z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")


def _sanitize(part: str) -> str:
    part = part.lower()
    part = re.sub(r"[^a-z0-9\-_.]", "-", part)
    return part.strip("-_.")


def validate_bucket_name(name: str) -> None:
    if not (3 <= len(name) <= 63):
        raise ValueError(f"bucket name {name!r} must be 3-63 chars, got {len(name)}")
    if not _NAME_RE.match(name):
        raise ValueError(
            f"bucket name {name!r} must use only lowercase letters, digits, '-', '_', '.' "
            "and start/end with a letter or digit"
        )
    lname = name.lower()
    if lname.startswith("goog"):
        raise ValueError(f"bucket name {name!r} must not start with 'goog'")
    if "google" in lname:
        raise ValueError(f"bucket name {name!r} must not contain 'google'")


def resolve_bucket_name(cfg: Config, run_id: str) -> str:
    project = _sanitize(cfg.project_name)
    if cfg.bucket_strategy == "per_run":
        name = f"{project}-{run_id}"
    else:
        name = f"{project}-raw-{_sanitize(cfg.env)}"
    validate_bucket_name(name)
    return name


def object_prefix(cfg: Config, run_id: str) -> str:
    """Empty for per_run (new bucket = isolation); run=<ts>/ for stable (shared bucket)."""
    if cfg.bucket_strategy == "per_run":
        return ""
    return f"run={run_id}/"


def get_client(cfg: Config) -> storage.Client:
    return storage.Client(project=cfg.gcs_project)


def ensure_bucket(cfg: Config, run_id: str, client: storage.Client | None = None) -> str:
    name = resolve_bucket_name(cfg, run_id)
    client = client or get_client(cfg)
    bucket = client.bucket(name)
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True
    try:
        client.create_bucket(bucket, location=cfg.gcs_location)
        logger.info("gcs: created bucket %s in %s", name, cfg.gcs_location)
    except Conflict:
        logger.info("gcs: bucket %s already exists, reusing", name)
    return name


def upload_file(
    client: storage.Client,
    bucket_name: str,
    blob_name: str,
    local_path: Path,
    content_type: str,
    metadata: dict | None = None,
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if metadata:
        blob.metadata = metadata
    blob.upload_from_filename(str(local_path), content_type=content_type)


def download_blob_to_file(
    client: storage.Client, bucket_name: str, blob_name: str, local_path: Path
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(local_path))


def get_blob_metadata(client: storage.Client, bucket_name: str, blob_name: str) -> dict | None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return None
    blob.reload()
    return {
        "metadata": blob.metadata or {},
        "size": blob.size,
        "content_type": blob.content_type,
    }


def iter_blob_lines(gs_uri: str):
    """Stream text lines from a gs:// URI (used when source_uri itself lives in GCS)."""
    assert gs_uri.startswith("gs://")
    _, _, rest = gs_uri.partition("gs://")
    bucket_name, _, blob_name = rest.partition("/")
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    yield from blob.download_as_text().splitlines()
