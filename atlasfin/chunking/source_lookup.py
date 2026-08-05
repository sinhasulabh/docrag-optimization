from atlasfin_ingest.stage1_source import read_source
from atlasfin.contracts import SourceRecord


def load_source_lookup(source_jsonl_path_or_uri: str) -> dict[str, SourceRecord]:
    """doc_name -> SourceRecord, loaded once and injected into a Chunker's constructor so
    Chunker.chunk(parsed) can stay a single-argument call per the Protocol, while still
    filling in company/doc_type/doc_period on each Chunk.
    """
    records = read_source(source_jsonl_path_or_uri)
    return {r.doc_name: r for r in records}
