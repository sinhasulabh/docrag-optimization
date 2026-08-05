import json
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from google.cloud import storage

from . import gcs


def _json_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "value"):  # Enum members (Status)
        return o.value
    raise TypeError(f"not JSON serializable: {type(o)!r}")


def _to_jsonable(record) -> dict:
    return asdict(record) if is_dataclass(record) else record


class ManifestWriter:
    """Append-only per-doc status log for one run, mirrored to GCS under _manifest/."""

    def __init__(self, client: storage.Client, bucket: str, prefix: str, local_dir: Path):
        self._client = client
        self._bucket = bucket
        self._prefix = prefix
        self._local_dir = local_dir
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = local_dir / "run_manifest.jsonl"
        self._manifest_path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record) -> None:
        line = json.dumps(_to_jsonable(record), default=_json_default)
        with self._lock:
            with self._manifest_path.open("a") as f:
                f.write(line + "\n")

    def flush(self) -> None:
        """Push the accumulated local manifest to GCS (object is overwritten, not appended)."""
        blob_name = f"{self._prefix}_manifest/run_manifest.jsonl"
        gcs.upload_file(
            self._client, self._bucket, blob_name, self._manifest_path, "application/x-ndjson"
        )

    def write_source_snapshot(self, records: list) -> None:
        blob_name = f"{self._prefix}_manifest/source.jsonl"
        local_path = self._local_dir / "source.jsonl"
        with local_path.open("w") as f:
            for rec in records:
                f.write(json.dumps(_to_jsonable(rec), default=_json_default) + "\n")
        gcs.upload_file(self._client, self._bucket, blob_name, local_path, "application/x-ndjson")

    def write_summary(self, summary: dict) -> None:
        blob_name = f"{self._prefix}_manifest/run_summary.json"
        local_path = self._local_dir / "run_summary.json"
        local_path.write_text(json.dumps(summary, indent=2, default=_json_default))
        gcs.upload_file(self._client, self._bucket, blob_name, local_path, "application/json")
