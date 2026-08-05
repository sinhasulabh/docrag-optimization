import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import ExperimentConfig

FP_SCHEMA_VERSION = 1


def offline_fingerprint(cfg: "ExperimentConfig") -> str:
    """Hashes the OFFLINE config subset (parsing/chunking/embedding) + the library versions
    that can silently change parse/chunk output for byte-identical config. Deliberately
    excludes name/retrieval/reranking -- that's what makes an online-knob-only config change
    produce the same fingerprint (and therefore reuse the cached offline artifact).
    """
    import importlib.metadata

    import docling

    payload = {
        "fp_schema_version": FP_SCHEMA_VERSION,
        "parsing": asdict(cfg.parsing),
        "chunking": asdict(cfg.chunking),
        "embedding": asdict(cfg.embedding),
        "docling_version": docling.__version__,
        "docling_core_version": importlib.metadata.version("docling-core"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"v{FP_SCHEMA_VERSION}-{digest}"
