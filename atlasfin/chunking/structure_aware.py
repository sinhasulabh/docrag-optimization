from collections import defaultdict
from pathlib import Path
from typing import Protocol

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from google.cloud import storage

from atlasfin.config.schema import ChunkingConfig
from atlasfin.contracts import Chunk, ParsedObject, SourceRecord, make_chunk_id, slugify_section

from ._docling_io import load_docling_document, load_markdown, md_uri_for
from ._tokenizer import build_tokenizer


class ContextualPrefixer(Protocol):
    def build_embedding_text(self, *, doc_name: str, full_doc_text: str, chunk_text: str) -> str: ...


class StructureAwareChunker:
    """Wraps docling_core's HybridChunker. The "parent" tier needed for
    RetrievalConfig.parent_child (an ONLINE knob -- the offline artifact must always carry
    parent text so toggling it later is free) is built by grouping HybridChunker's own child
    chunks by identical heading path, NOT via a separate HierarchicalChunker pass: verified
    against real parsed docs that HierarchicalChunker's un-merged output is MORE granular
    than HybridChunker's (one chunk per structural element, not one per section), so it
    cannot serve as a coarser parent layer. Grouping HybridChunker's own children by heading
    path is simpler and structurally guaranteed consistent with them.
    """

    def __init__(
        self,
        cfg: ChunkingConfig,
        *,
        source_lookup: dict[str, SourceRecord],
        gcs_client: storage.Client,
        scratch_dir: Path,
        contextual_prefixer: ContextualPrefixer | None = None,
    ):
        self._cfg = cfg
        self._source_lookup = source_lookup
        self._gcs_client = gcs_client
        self._scratch_dir = Path(scratch_dir)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self._contextual_prefixer = contextual_prefixer
        if cfg.contextual_prefix and contextual_prefixer is None:
            raise ValueError("chunking.contextual_prefix=True requires a contextual_prefixer")

        self._tokenizer = build_tokenizer(cfg.max_tokens)
        self._chunker = HybridChunker(
            tokenizer=self._tokenizer, max_tokens=cfg.max_tokens, merge_peers=True
        )

    def chunk(self, parsed: ParsedObject) -> list[Chunk]:
        source = self._source_lookup.get(parsed.doc_name)
        if source is None:
            raise KeyError(f"no SourceRecord for doc_name={parsed.doc_name!r} in source_lookup")

        doc = load_docling_document(parsed.gcs_parsed_uri, self._gcs_client, self._scratch_dir)
        doc_chunks = list(self._chunker.chunk(dl_doc=doc))

        groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i, dc in enumerate(doc_chunks):
            groups[tuple(dc.meta.headings or [])].append(i)

        parent_text_by_group: dict[tuple[str, ...], str] = {}
        parent_id_by_group: dict[tuple[str, ...], str] = {}
        for headings, idxs in groups.items():
            if len(idxs) < 2:
                continue  # single-chunk section: parent == child, no benefit to indirection
            parent_text = "\n\n".join(doc_chunks[i].text for i in idxs)
            section_slug = slugify_section(headings[-1] if headings else "")
            parent_text_by_group[headings] = parent_text
            parent_id_by_group[headings] = make_chunk_id(parsed.doc_name, section_slug, parent_text)

        full_doc_text = None
        if self._cfg.contextual_prefix:
            full_doc_text = load_markdown(
                md_uri_for(parsed.gcs_parsed_uri), self._gcs_client, self._scratch_dir
            )

        chunks: list[Chunk] = []
        for i, dc in enumerate(doc_chunks):
            text = dc.text
            if not text.strip():
                # observed in practice: a heading-only element (e.g. a dropped table or
                # page-break artifact) with no extractable body text -- useless for retrieval,
                # and Voyage's embed API rejects empty strings outright.
                continue

            headings = tuple(dc.meta.headings or [])
            pages = sorted({prov.page_no for item in dc.meta.doc_items for prov in item.prov})
            section_slug = slugify_section(headings[-1] if headings else "")

            embedding_text = text
            if self._contextual_prefixer is not None and full_doc_text is not None:
                embedding_text = self._contextual_prefixer.build_embedding_text(
                    doc_name=parsed.doc_name, full_doc_text=full_doc_text, chunk_text=text
                )

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(parsed.doc_name, section_slug, text),
                    doc_name=parsed.doc_name,
                    chunk_index=i,
                    text=text,
                    embedding_text=embedding_text,
                    parent_text=parent_text_by_group.get(headings),
                    section=" > ".join(headings),
                    section_slug=section_slug,
                    pages=pages,
                    company=source.company,
                    doc_type=source.doc_type,
                    doc_period=source.doc_period,
                    parent_chunk_id=parent_id_by_group.get(headings),
                    token_count=self._tokenizer.count_tokens(text),
                    metadata={
                        "chunker_strategy": "structure_aware",
                        "docling_version": parsed.docling_version,
                    },
                )
            )
        return chunks
