from pathlib import Path

from google.cloud import storage

from atlasfin.config.schema import ParsingConfig

from .docling_impl import DEFAULT_PARSE_TIMEOUT_S, DoclingParser
from .interface import Parser


def build(
    cfg: ParsingConfig,
    *,
    gcs_client: storage.Client,
    bucket: str,
    prefix: str,
    scratch_dir: Path,
    parse_device: str = "auto",
    parse_timeout_s: int = DEFAULT_PARSE_TIMEOUT_S,
) -> Parser:
    if cfg.backend != "docling":
        raise ValueError(f"parsing.backend={cfg.backend!r} not supported (only 'docling')")
    return DoclingParser(
        cfg,
        gcs_client=gcs_client,
        bucket=bucket,
        prefix=prefix,
        scratch_dir=scratch_dir,
        parse_device=parse_device,
        parse_timeout_s=parse_timeout_s,
    )
