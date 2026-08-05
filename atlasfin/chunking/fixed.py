from pathlib import Path

from google.cloud import storage

from atlasfin.config.schema import ChunkingConfig
from atlasfin.contracts import Chunk, ParsedObject, SourceRecord, make_chunk_id

from ._docling_io import load_markdown, md_uri_for
from ._tokenizer import build_tokenizer, token_windows

_NO_HEADING_SECTION = ""
_NO_HEADING_SLUG = "root"


class FixedChunker:
    """Hard non-overlapping token windows over the plain-text .md export. overlap_tokens is
    deliberately ignored (ChunkingConfig documents it as "used by recursive/fixed" but a
    fixed-window strategy is exactly the one place overlap doesn't apply by definition).
    Same page/section tradeoff as RecursiveChunker -- no DoclingDocument structure walk.
    """

    def __init__(
        self,
        cfg: ChunkingConfig,
        *,
        source_lookup: dict[str, SourceRecord],
        gcs_client: storage.Client,
        scratch_dir: Path,
    ):
        self._cfg = cfg
        self._source_lookup = source_lookup
        self._gcs_client = gcs_client
        self._scratch_dir = Path(scratch_dir)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = build_tokenizer(cfg.max_tokens)

    def chunk(self, parsed: ParsedObject) -> list[Chunk]:
        source = self._source_lookup.get(parsed.doc_name)
        if source is None:
            raise KeyError(f"no SourceRecord for doc_name={parsed.doc_name!r} in source_lookup")

        text = load_markdown(
            md_uri_for(parsed.gcs_parsed_uri), self._gcs_client, self._scratch_dir
        )
        windows = token_windows(text, self._cfg.max_tokens, overlap_tokens=0, tokenizer=self._tokenizer)

        chunks: list[Chunk] = []
        for i, window_text in enumerate(windows):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(parsed.doc_name, _NO_HEADING_SLUG, window_text),
                    doc_name=parsed.doc_name,
                    chunk_index=i,
                    text=window_text,
                    embedding_text=window_text,
                    parent_text=None,
                    section=_NO_HEADING_SECTION,
                    section_slug=_NO_HEADING_SLUG,
                    pages=[],
                    company=source.company,
                    doc_type=source.doc_type,
                    doc_period=source.doc_period,
                    parent_chunk_id=None,
                    token_count=self._tokenizer.count_tokens(window_text),
                    metadata={
                        "chunker_strategy": "fixed",
                        "docling_version": parsed.docling_version,
                        "pages_available": False,
                    },
                )
            )
        return chunks
