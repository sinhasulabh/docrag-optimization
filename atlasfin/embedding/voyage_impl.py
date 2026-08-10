import math
import os

import voyageai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from atlasfin.config.schema import EmbeddingConfig

DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT_S = 60.0


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class VoyageEmbedder:
    """Voyage AI embeddings -- voyage-finance-2 is domain-tuned for financial filings,
    which is why it's the default here rather than a general-purpose model. Uses
    VOYAGE_API_KEY, a standalone Voyage credential -- separate from GEMINI_API_KEY, which is
    still used for the generative LLM pieces (retrieval/query_transform.py,
    chunking/contextual.py). Retries are handled by the voyageai SDK itself (max_retries=) for
    RateLimitError/ServiceUnavailableError/Timeout -- but NOT for a bare dropped connection
    (observed in practice as APIConnectionError over a long sequential embedding run), so
    _embed_batch adds its own retry layer for that specific case.
    """

    def __init__(self, cfg: EmbeddingConfig, api_key: str | None = None):
        self._cfg = cfg
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set (from dashboard.voyageai.com -- not a Gemini "
                "credential)"
            )
        self._client = voyageai.Client(
            api_key=key, max_retries=DEFAULT_MAX_RETRIES, timeout=DEFAULT_TIMEOUT_S
        )

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return self._embed_all(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed_all([text], input_type="query")[0]

    def _embed_all(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._cfg.batch_size):
            batch = texts[i : i + self._cfg.batch_size]
            vectors.extend(self._embed_batch(batch, input_type))
        return vectors

    @retry(
        retry=retry_if_exception_type(voyageai.error.APIConnectionError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _embed_batch(self, batch: list[str], input_type: str) -> list[list[float]]:
        kwargs: dict = dict(texts=batch, model=self._cfg.model_id, input_type=input_type)
        # NOT verified offline whether voyage-finance-2 (an older, "voyage-2"-generation
        # hosted model) supports output_dimension (Matryoshka) truncation the way newer
        # Voyage models do -- only sent through if explicitly configured, so an unsupported
        # request surfaces as a clear InvalidRequestError rather than silently misbehaving.
        # See config/validation.py's dimension check for the same caveat.
        if self._cfg.dimension is not None:
            kwargs["output_dimension"] = self._cfg.dimension

        result = self._client.embed(**kwargs)
        vectors = result.embeddings
        if self._cfg.normalize:
            vectors = [_l2_normalize(v) for v in vectors]
        return vectors
