import os
from pathlib import Path

from google.cloud import storage

from atlasfin.config.schema import ChunkingConfig
from atlasfin.contracts import SourceRecord

from .fixed import FixedChunker
from .interface import Chunker
from .recursive import RecursiveChunker
from .structure_aware import StructureAwareChunker


def _default_contextual_cache_path(model_id: str) -> Path:
    root = Path(os.environ.get("ATLASFIN_CACHE_DIR", ".atlasfin_cache"))
    safe_model = model_id.replace("/", "__")
    return root / "contextual_prefix" / f"{safe_model}.jsonl"


def build(
    cfg: ChunkingConfig,
    *,
    source_lookup: dict[str, SourceRecord],
    gcs_client: storage.Client,
    scratch_dir: Path,
) -> Chunker:
    contextual_prefixer = None
    if cfg.contextual_prefix:
        if cfg.strategy != "structure_aware":
            raise ValueError(
                f"chunking.contextual_prefix=True is only implemented for strategy="
                f"'structure_aware' (recursive/fixed don't track per-chunk document "
                f"position needed to situate a blurb), got strategy={cfg.strategy!r}"
            )
        from .contextual import GeminiContextualPrefixer

        contextual_prefixer = GeminiContextualPrefixer(
            model_id=cfg.contextual_model,
            cache_path=_default_contextual_cache_path(cfg.contextual_model),
        )

    if cfg.strategy == "structure_aware":
        return StructureAwareChunker(
            cfg,
            source_lookup=source_lookup,
            gcs_client=gcs_client,
            scratch_dir=scratch_dir,
            contextual_prefixer=contextual_prefixer,
        )
    if cfg.strategy == "recursive":
        return RecursiveChunker(
            cfg, source_lookup=source_lookup, gcs_client=gcs_client, scratch_dir=scratch_dir
        )
    if cfg.strategy == "fixed":
        return FixedChunker(
            cfg, source_lookup=source_lookup, gcs_client=gcs_client, scratch_dir=scratch_dir
        )
    raise ValueError(f"chunking.strategy must be structure_aware|recursive|fixed, got {cfg.strategy!r}")
