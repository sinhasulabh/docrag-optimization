from atlasfin.config.schema import RerankingConfig

from .cross_encoder import CrossEncoderReranker
from .interface import Reranker
from .noop import NoRerank


def build(cfg: RerankingConfig) -> Reranker | None:
    if not cfg.enabled:
        return None
    return CrossEncoderReranker(cfg)
