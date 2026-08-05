import hashlib
import json
import logging
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from atlasfin.contracts import Candidate

from . import fusion
from .interface import Retriever

logger = logging.getLogger("atlasfin.retrieval.query_transform")

_MAX_RETRIES = 4

_REWRITE_PROMPT = """Rewrite the following search query to be clearer and more specific, for \
retrieving relevant passages from company financial filings (10-K/10-Q). Answer with ONLY \
the rewritten query and nothing else.

Query: {query}"""

_DECOMPOSE_PROMPT = """Break the following question down into 2-4 simpler standalone \
sub-questions that together would help answer it. If the question is already simple, just \
return it as-is. Answer with ONLY the sub-questions, one per line, no numbering.

Question: {query}"""

_HYDE_PROMPT = """Write a short hypothetical passage from a company's financial filing \
(10-K/10-Q) that would answer the following question, for the purpose of improving semantic \
search retrieval. Answer with ONLY the passage.

Question: {query}"""


class QueryTransformer(Protocol):
    def transform(self, query: str) -> list[str]: ...  # 1+ query variants to retrieve+fuse


class NoopTransform:
    def transform(self, query: str) -> list[str]:
        return [query]


class _GeminiTransformBase:
    """Shared plumbing for the three real Gemini-backed transforms: temp=0 + a local
    jsonlines cache keyed by (mode, model, query) -- this is what makes run_experiment's
    determinism requirement (two runs of the same config produce identical metrics) actually
    achievable when an LLM is in the retrieval loop, not just "probably stable at temp=0".
    """

    mode: str
    prompt_template: str

    def __init__(self, model_id: str, cache_path: Path, api_key: str | None = None):
        self._model_id = model_id
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (needed for retrieval.query_transform)")
        self._client = genai.Client(api_key=key)
        self._cache_path = Path(cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = _load_cache(self._cache_path)

    def _call(self, query: str) -> str:
        key = _cache_key(self.mode, self._model_id, query)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        prompt = self.prompt_template.format(query=query)
        config = types.GenerateContentConfig(temperature=0)
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self._model_id, contents=prompt, config=config
                )
                text = (resp.text or "").strip()
                self._cache[key] = text
                _append_cache(self._cache_path, key, text)
                return text
            except genai_errors.ServerError as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES:
                    raise
                backoff = (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "query_transform: %s call failed (%s), retrying in %.1fs", self.mode, exc, backoff
                )
                time.sleep(backoff)
        raise last_exc or RuntimeError("unreachable")


class RewriteTransform(_GeminiTransformBase):
    mode = "rewrite"
    prompt_template = _REWRITE_PROMPT

    def transform(self, query: str) -> list[str]:
        return [self._call(query)]


class DecomposeTransform(_GeminiTransformBase):
    mode = "decompose"
    prompt_template = _DECOMPOSE_PROMPT

    def transform(self, query: str) -> list[str]:
        raw = self._call(query)
        sub_queries = [line.strip() for line in raw.splitlines() if line.strip()]
        return sub_queries or [query]


class HyDETransform(_GeminiTransformBase):
    mode = "hyde"
    prompt_template = _HYDE_PROMPT

    def transform(self, query: str) -> list[str]:
        return [self._call(query)]


class TransformingRetriever:
    """Wraps a base Retriever with a QueryTransformer. A single-variant transform (rewrite,
    hyde) just retrieves with the transformed text. A multi-variant transform (decompose)
    retrieves separately per sub-query and fuses via the same rrf_fuse() used for dense+sparse
    fusion in hybrid.py -- reused, not reimplemented.
    """

    def __init__(self, inner: Retriever, transformer: QueryTransformer):
        self._inner = inner
        self._transformer = transformer

    def retrieve(self, query: str, k: int, filters: dict | None = None) -> list[Candidate]:
        variants = self._transformer.transform(query)
        if len(variants) == 1:
            return self._inner.retrieve(variants[0], k, filters)

        candidates_by_id: dict[str, Candidate] = {}
        ranked_lists: list[list[tuple[str, float]]] = []
        for variant in variants:
            candidates = self._inner.retrieve(variant, k, filters)
            candidates_by_id.update({c.chunk_id: c for c in candidates})
            ranked_lists.append([(c.chunk_id, c.score) for c in candidates])

        fused = fusion.rrf_fuse(ranked_lists)[:k]
        return [replace(candidates_by_id[cid], score=score) for cid, score in fused]


def _cache_key(mode: str, model_id: str, query: str) -> str:
    blob = f"{mode}\x00{model_id}\x00{query}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    cache: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["key"]] = row["value"]
    return cache


def _append_cache(path: Path, key: str, value: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")
