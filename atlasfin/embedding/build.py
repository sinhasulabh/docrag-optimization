from atlasfin.config.schema import EmbeddingConfig

from .interface import Embedder
from .voyage_impl import VoyageEmbedder


def build(cfg: EmbeddingConfig) -> Embedder:
    if not cfg.model_id.startswith("voyage-"):
        raise ValueError(
            f"embedding.model_id={cfg.model_id!r} not supported (only Voyage AI models -- "
            "e.g. 'voyage-finance-2')"
        )
    return VoyageEmbedder(cfg)
