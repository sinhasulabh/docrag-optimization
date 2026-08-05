from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Status(str, Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"  # already present (resume)


@dataclass(frozen=True)
class SourceRecord:  # Stage 1 output
    doc_name: str  # primary key
    company: str
    gics_sector: str
    doc_type: str
    doc_period: int
    doc_link: str


@dataclass(frozen=True)
class RawObject:  # Stage 2 output
    doc_name: str
    source_uri: str
    gcs_raw_uri: str  # gs://.../raw/{doc_type}/{doc_name}.pdf
    sha256: str  # forward hook for reconciliation/chunk IDs
    size_bytes: int
    content_type: str
    http_status: int
    fetched_at: datetime
    status: Status
    error: str | None = None


@dataclass(frozen=True)
class ParsedObject:  # Stage 3 output
    doc_name: str
    gcs_raw_uri: str
    gcs_parsed_uri: str  # gs://.../parsed/{doc_name}.docling.json
    page_count: int
    docling_version: str  # pinned; recorded for reproducibility
    parse_ms: int
    parsed_at: datetime
    status: Status
    error: str | None = None
