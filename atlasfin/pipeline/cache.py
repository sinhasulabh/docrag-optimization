import os
from pathlib import Path


def cache_root() -> Path:
    return Path(os.environ.get("ATLASFIN_CACHE_DIR", ".atlasfin_cache"))


def offline_dir(fingerprint: str) -> Path:
    return cache_root() / "offline" / fingerprint


def runs_dir(experiment_name: str, timestamp: str) -> Path:
    return cache_root() / "runs" / f"{experiment_name}__{timestamp}"
