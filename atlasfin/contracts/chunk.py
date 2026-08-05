import hashlib
import re
from dataclasses import dataclass

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_section(section: str) -> str:
    """Lowercase, alnum + hyphens only. Empty/heading-less input -> 'root'."""
    slug = _SLUG_RE.sub("-", section.lower()).strip("-")
    return slug or "root"


def make_chunk_id(doc_name: str, section_slug: str, text: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{doc_name}#{section_slug}#{text_hash}"


def parse_chunk_id(chunk_id: str) -> tuple[str, str, str]:
    """Returns (doc_name, section_slug, text_hash). Raises ValueError on malformed input."""
    parts = chunk_id.split("#")
    if len(parts) != 3:
        raise ValueError(f"malformed chunk_id (expected doc_name#section_slug#hash): {chunk_id!r}")
    return parts[0], parts[1], parts[2]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{doc_name}#{section_slug}#{sha256(text)[:12]}"
    doc_name: str
    chunk_index: int  # 0-based position within this doc's chunk sequence
    text: str  # verbatim chunk text; chunk_id's hash is sha256(text)
    embedding_text: str  # text actually sent to the embedder; == text unless contextual_prefix is on
    parent_text: str | None  # full parent-section text, for RetrievalConfig.parent_child
    section: str  # " > "-joined heading path
    section_slug: str
    pages: list[int]  # sorted unique 1-indexed pages
    company: str
    doc_type: str
    doc_period: int
    parent_chunk_id: str | None
    token_count: int  # local proxy-tokenizer count, not a Gemini-exact count
    metadata: dict
