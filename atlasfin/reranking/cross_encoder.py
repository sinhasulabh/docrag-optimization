from dataclasses import replace

from sentence_transformers import CrossEncoder

from atlasfin.config.schema import RerankingConfig
from atlasfin.contracts import Candidate

DEFAULT_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, cfg: RerankingConfig):
        self._cfg = cfg
        # max_length truncation happens via the model's own tokenizer at construction time,
        # not per predict() call.
        self._model = CrossEncoder(cfg.model_id or DEFAULT_MODEL_ID, max_length=cfg.max_pair_tokens)

    def rerank(self, query: str, candidates: list[Candidate], depth: int) -> list[Candidate]:
        window = candidates[:depth]
        if not window:
            return []
        pairs = [(query, c.text) for c in window]
        scores = self._model.predict(pairs, show_progress_bar=False)
        rescored = [replace(c, score=float(s)) for c, s in zip(window, scores)]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored
