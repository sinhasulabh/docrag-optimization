import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

logger = logging.getLogger("atlasfin.chunking.contextual")

_PROMPT_TEMPLATE = """<document>
{full_doc_text}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Give a short, succinct context (1-2 sentences) to situate this chunk within the overall \
document, for the purpose of improving search retrieval of the chunk. Answer only with the \
succinct context and nothing else."""

# Gemini's explicit context caching has a minimum-token floor for some models; documents
# shorter than this just get inlined directly instead of cached (still correct, just less
# economical -- only matters for very short filings).
_MAX_INLINE_DOC_CHARS = 400_000
_CACHE_TTL_S = "3600s"
_MAX_RETRIES = 4


class GeminiContextualPrefixer:
    """Builds a short LLM-generated context blurb per chunk (Anthropic's "contextual
    retrieval" pattern) and prepends it to the chunk text for embedding. Uses Gemini context
    caching so the (large, repeated) full-document text is only billed once per document, not
    once per chunk -- naively inlining the full doc on every one of ~500 per-chunk calls for
    a single 10-K would be cost-prohibitive.

    Two layers of caching:
    - Gemini-side context cache (per doc_name, in-process only, TTL-bound): avoids re-paying
      for document tokens on every chunk within a run.
    - Local on-disk jsonlines cache (persists across runs): avoids re-calling Gemini at all
      for a chunk whose blurb was already generated -- this is also what makes
      run_experiment's determinism requirement achievable with an LLM in the loop.
    """

    def __init__(self, model_id: str, cache_path: Path, api_key: str | None = None):
        self._model_id = model_id
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self._cache_path = Path(cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._local_cache = _load_local_cache(self._cache_path)
        self._doc_cache_names: dict[str, str | None] = {}

    def build_embedding_text(self, *, doc_name: str, full_doc_text: str, chunk_text: str) -> str:
        key = _cache_key(self._model_id, doc_name, chunk_text)
        blurb = self._local_cache.get(key)
        if blurb is None:
            blurb = self._generate_blurb(doc_name, full_doc_text, chunk_text)
            self._local_cache[key] = blurb
            _append_local_cache(self._cache_path, key, blurb)
        return f"{blurb}\n\n{chunk_text}"

    def _generate_blurb(self, doc_name: str, full_doc_text: str, chunk_text: str) -> str:
        cached_content = self._get_or_create_doc_cache(doc_name, full_doc_text)
        if cached_content is not None:
            prompt = _PROMPT_TEMPLATE.format(full_doc_text="[see cached document]", chunk_text=chunk_text)
            config = types.GenerateContentConfig(temperature=0, cached_content=cached_content)
        else:
            prompt = _PROMPT_TEMPLATE.format(
                full_doc_text=full_doc_text[:_MAX_INLINE_DOC_CHARS], chunk_text=chunk_text
            )
            config = types.GenerateContentConfig(temperature=0)

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self._model_id, contents=prompt, config=config
                )
                return (resp.text or "").strip()
            except genai_errors.ServerError as exc:
                if attempt >= _MAX_RETRIES:
                    raise
                backoff = (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "contextual: %s blurb call failed (%s), retrying in %.1fs", doc_name, exc, backoff
                )
                time.sleep(backoff)
        raise RuntimeError("unreachable")

    def _get_or_create_doc_cache(self, doc_name: str, full_doc_text: str) -> str | None:
        if doc_name in self._doc_cache_names:
            return self._doc_cache_names[doc_name]
        cache_name: str | None = None
        try:
            cached = self._client.caches.create(
                model=self._model_id,
                config=types.CreateCachedContentConfig(
                    contents=[full_doc_text], ttl=_CACHE_TTL_S, display_name=f"atlasfin-{doc_name}"
                ),
            )
            cache_name = cached.name
        except genai_errors.ClientError as exc:
            logger.info(
                "contextual: %s not eligible for context caching (%s), inlining instead",
                doc_name,
                exc,
            )
        self._doc_cache_names[doc_name] = cache_name
        return cache_name


def _cache_key(model_id: str, doc_name: str, chunk_text: str) -> str:
    blob = f"{model_id}\x00{doc_name}\x00{chunk_text}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_local_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    cache: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[row["key"]] = row["blurb"]
    return cache


def _append_local_cache(path: Path, key: str, blurb: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "blurb": blurb}) + "\n")
