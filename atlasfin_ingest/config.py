from dataclasses import dataclass


@dataclass
class Config:
    project_name: str  # -> bucket prefix, e.g. "atlasfin"
    source_uri: str
    gcs_project: str
    gcs_location: str = "us-central1"
    bucket_strategy: str = "stable"  # "per_run" | "stable"
    env: str = "dev"
    # download
    dl_timeout_s: int = 30
    dl_max_retries: int = 4
    dl_max_size_mb: int = 100
    dl_concurrency: int = 12
    dl_min_host_interval_s: float = 0.5  # per-host politeness (e.g. SEC EDGAR fair-access)
    user_agent: str = "AtlasFin-ingest/0.1 (you@example.com)"
    # parse
    ocr: bool = False
    parse_concurrency: int = 2
    parse_timeout_s: int = 300
    parse_device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps" | "xpu"
    docling_version_pin: str | None = None  # None -> use installed docling.__version__
    # filter (optional)
    filter_doc_types: list[str] | None = None
    filter_companies: list[str] | None = None
    filter_doc_names: list[str] | None = None
    # local scratch space for streaming temp files
    local_scratch_dir: str | None = None

    def __post_init__(self) -> None:
        if self.bucket_strategy not in ("per_run", "stable"):
            raise ValueError(
                f"bucket_strategy must be 'per_run' or 'stable', got {self.bucket_strategy!r}"
            )
        if self.parse_device not in ("auto", "cpu", "cuda", "mps", "xpu"):
            raise ValueError(
                f"parse_device must be one of auto/cpu/cuda/mps/xpu, got {self.parse_device!r}"
            )
