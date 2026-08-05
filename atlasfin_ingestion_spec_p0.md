# AtlasFin — Ingestion Spec (P0, Stages 1–3)

**Scope:** read the document source → download each file and store raw bytes on GCS → parse each document.
**Out of scope (explicit non-goals):** chunking, embedding, vector store, cross-run delta/reconciliation, dedup, chunk IDs, retrieval. Those are later phases. This spec deliberately stops at a parsed, persisted `DoclingDocument`.

The one thing we *do* carry forward for later phases is a per-document **content hash** (sha256) recorded at download time. We record it now; we do **not** act on it here.

---

## 0. Document source (grounded schema)

Source file: `financebench_document_information.jsonl` — one JSON object per line, 361 records, every record has a fetchable `doc_link`.

| field         | type   | role                                             | example                                   |
|---------------|--------|--------------------------------------------------|-------------------------------------------|
| `doc_name`    | str    | **stable business key / primary id**             | `3M_2015_10K`                             |
| `company`     | str    | metadata / partition                             | `3M`                                     |
| `gics_sector` | str    | metadata                                         | `Industrials`                            |
| `doc_type`    | str    | partition (`10k`,`10q`,`8k`,`Earnings`,…)        | `10k`                                    |
| `doc_period`  | int    | metadata                                         | `2015`                                   |
| `doc_link`    | str    | **fetch URL** (PDF)                              | `https://investors.3m.com/.../....pdf`   |

`doc_name` is the identity that must remain stable across every stage and every future run. Never key anything off list position.

---

## 1. Data contracts (the boundaries between stages)

Three immutable records flow through the pipeline. Each stage consumes the previous record and emits the next. Define them as typed objects.

```python
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class Status(str, Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"          # already present (resume)

@dataclass(frozen=True)
class SourceRecord:            # Stage 1 output
    doc_name: str             # primary key
    company: str
    gics_sector: str
    doc_type: str
    doc_period: int
    doc_link: str

@dataclass(frozen=True)
class RawObject:              # Stage 2 output
    doc_name: str
    source_uri: str
    gcs_raw_uri: str          # gs://.../raw/{doc_type}/{doc_name}.pdf
    sha256: str               # forward hook for reconciliation/chunk IDs
    size_bytes: int
    content_type: str
    http_status: int
    fetched_at: datetime
    status: Status
    error: str | None = None

@dataclass(frozen=True)
class ParsedObject:           # Stage 3 output
    doc_name: str
    gcs_raw_uri: str
    gcs_parsed_uri: str       # gs://.../parsed/{doc_name}.docling.json
    page_count: int
    docling_version: str      # pinned; recorded for reproducibility
    parse_ms: int
    parsed_at: datetime
    status: Status
    error: str | None = None
```

---

## 2. GCS layout & bucket strategy

### Bucket name
Requested format: `projectname-datetimestamp`, e.g. `atlasfin-20260731t203045z`.

**Hard constraints to enforce before creation** (GCS rules): 3–63 chars; lowercase letters, digits, hyphens, underscores, dots only; must start and end with a letter or digit; globally unique; must not start with `goog` or contain `google`. Build the timestamp as compact UTC (`YYYYMMDDtHHMMSSz`) so it stays inside the allowed charset and stays short.

> **Design note — read before you commit to this.** A *new bucket per run* gives you clean, immutable per-run snapshots, but it has two production costs: (a) it **breaks cross-run delta** — every run lands in a fresh bucket, so you can never diff against a stable location, which is exactly the reconciliation we designed earlier; and (b) bucket creation is a heavier, rate-limited, per-project-capped operation, so this pattern does not scale to frequent runs.
>
> The production-standard alternative is **one stable bucket per project+env** (`atlasfin-raw-dev`) with the run timestamp in the **object path prefix** (`raw/run=<ts>/...`). Same snapshot isolation, but a stable location you can reconcile against later.
>
> This spec keeps `bucket_strategy` as a config knob (`per_run` | `stable`) so the choice is one line, not a rewrite. Default it to whichever you decide — I'd default `stable` for anything you intend to grow.

### Object layout (inside the chosen bucket)
```
{prefix}/raw/{doc_type}/{doc_name}.pdf              # Stage 2: raw bytes
{prefix}/parsed/{doc_name}.docling.json             # Stage 3: lossless DoclingDocument
{prefix}/parsed/{doc_name}.md                        # Stage 3: optional, human-eyeball only
{prefix}/_manifest/source.jsonl                      # Stage 1 snapshot (desired set)
{prefix}/_manifest/run_manifest.jsonl                # append-only per-doc status log
{prefix}/_manifest/run_summary.json                  # final counts
```
(`{prefix}` is empty for `per_run`, or `run=<ts>` for `stable`.)

Every raw object also carries **object metadata**: `sha256`, `source_uri`, `fetched_at`, `doc_name`.

---

## 3. Stage specs

### Stage 1 — Read source
**Input:** path/URI to the source JSONL; optional filter.
**Output:** `list[SourceRecord]` + snapshot written to `_manifest/source.jsonl`.

Behavior:
- Parse JSONL line-by-line; validate each row against `SourceRecord`.
- Dedupe on `doc_name` (last-wins, and log the collision).
- Validate `doc_link` is a well-formed `http(s)` URL. Invalid/missing → drop the record with a logged warning; **do not abort the run**.
- Support an optional filter (`doc_types`, `companies`, or an explicit `doc_names` allow-list — e.g. to ingest only the ~84 filings referenced by the Q&A set instead of all 361).

Edge cases: empty source → exit cleanly with a summary of 0. Malformed line → skip + log, continue.

```python
def read_source(source_uri: str, flt: "SourceFilter | None" = None) -> list[SourceRecord]: ...
```

### Stage 2 — Download & store
**Input:** `list[SourceRecord]`, resolved target bucket (created if absent).
**Output:** `list[RawObject]`; each appended to `run_manifest.jsonl`.

Bucket setup:
- Resolve bucket name per `bucket_strategy`; **validate against GCS rules** before the API call.
- Create if not exists in configured `location`; enable uniform bucket-level access. (Bucket creation is idempotent-ish — treat "already exists / owned by you" as success.)

Per-document download:
- HTTP GET with **streaming** (never load a full PDF into memory), a **timeout**, and **retry with exponential backoff** on `429`/`5xx`/connection errors.
- Send a **descriptive `User-Agent`** with contact info. Several `doc_link`s resolve to SEC-hosted content, and SEC EDGAR fair-access returns `403` for requests without a declared UA and throttles above its rate limit — so set the UA and keep per-host request rate polite.
- Follow redirects; cap at `max_size_mb`.
- **Validate content**: non-empty, and PDF magic bytes (`%PDF`) or `application/pdf` content-type. Anything else → `FAILED` (likely an HTML error/login page, not a PDF).
- Compute `sha256` **while streaming** (single pass).
- Upload to `raw/{doc_type}/{doc_name}.pdf` with the object metadata above.

Within-run idempotency (resume): if the target object already exists **and** its stored `sha256` metadata matches a freshly-computed hash (or you trust an existing object), mark `SKIPPED` and don't re-upload. (Cross-run delta is explicitly *not* here.)

Concurrency: network-bound → a bounded pool is fine (`download.concurrency`, e.g. 8–16), with per-host politeness.

Failure isolation: any single doc failing (`404/403/timeout/non-PDF/zero-byte`) → `RawObject(status=FAILED, error=...)`, log, **continue the loop**. One bad link never aborts the run.

```python
def ensure_bucket(cfg: Config) -> str: ...                       # returns bucket name
def download_one(rec: SourceRecord, bucket: str, cfg: Config) -> RawObject: ...
def download_all(records: list[SourceRecord], bucket: str, cfg: Config) -> list[RawObject]: ...
```

### Stage 3 — Parse
**Input:** `RawObject`s with `status in {DONE, SKIPPED}`.
**Output:** `list[ParsedObject]`; each appended to `run_manifest.jsonl`.

Per-document parse:
- Fetch raw bytes (stream to a temp file or in-memory buffer).
- `doc = DocumentConverter().convert(path_or_stream).document`.
- Export **lossless JSON** (`doc.export_to_dict()`); upload to `parsed/{doc_name}.docling.json`.
- Optionally `doc.export_to_markdown()` → `parsed/{doc_name}.md` for eyeballing only (not the source of truth).
- Record `page_count`, `parse_ms`, and the **pinned `docling_version`** (parser output changes across versions — this is your reproducibility anchor, same lesson as chunker config).

Config: `ocr` (default off for born-digital PDFs; on for scanned — 10-Ks are usually born-digital, so default off and flag exceptions), table mode on.

Concurrency: this stage runs vision/layout models and is **memory-bound**, so it needs its **own, lower** concurrency (`parse.concurrency`, e.g. 1–4) — do **not** reuse the download pool size. This is a batch job; never call `DocumentConverter` in a request path.

Failure isolation: corrupt/encrypted PDF, conversion error, per-doc timeout, OOM → `ParsedObject(status=FAILED, error=...)`, log, continue.

```python
def parse_one(raw: RawObject, bucket: str, cfg: Config) -> ParsedObject: ...
def parse_all(raws: list[RawObject], bucket: str, cfg: Config) -> list[ParsedObject]: ...
```

---

## 4. Config

```python
@dataclass
class Config:
    project_name: str                       # -> bucket prefix, e.g. "atlasfin"
    source_uri: str
    gcs_project: str
    gcs_location: str = "us-central1"
    bucket_strategy: str = "stable"         # "per_run" | "stable"
    env: str = "dev"
    # download
    dl_timeout_s: int = 30
    dl_max_retries: int = 4
    dl_max_size_mb: int = 100
    dl_concurrency: int = 12
    user_agent: str = "AtlasFin-ingest/0.1 (you@example.com)"
    # parse
    ocr: bool = False
    parse_concurrency: int = 2
    parse_timeout_s: int = 300
    docling_version_pin: str = "<pin exact version>"
    # filter (optional)
    filter_doc_types: list[str] | None = None
    filter_companies: list[str] | None = None
    filter_doc_names: list[str] | None = None
```

---

## 5. Run model, idempotency, observability

- One `run_id` (the UTC timestamp) per invocation.
- Per-doc lifecycle: `PENDING → (DONE | FAILED)` at Stage 2, then `→ (DONE | FAILED)` at Stage 3. Terminal status for **every** source record ends up in `run_manifest.jsonl`.
- **Resumable:** re-running against the same location skips objects already present (`SKIPPED`); a partial/crashed run heals on the next run.
- **Never aborts on a single-doc failure.** Process exit code is non-zero iff any doc ended `FAILED`, so CI/schedulers can detect partial failure.
- Structured (JSON) log line per doc with `doc_name`, stage, status, timing, error. Emit `run_summary.json`: totals for attempted / downloaded / skipped / parsed / failed, plus failure list.

---

## 6. Acceptance criteria (definition of done)

1. Given the source (all 361 or a filtered subset of N), a run creates/uses the configured bucket and lands **N raw PDFs** under `raw/` and **N parsed JSON** under `parsed/` (minus enumerated failures).
2. Every source record has a terminal status row in `run_manifest.jsonl`; nothing is silently dropped.
3. Every stored raw object has a recorded `sha256` (object metadata **and** manifest).
4. Re-running against the same location re-uploads nothing already present and matching (idempotent within a location) — proven by a second run reporting all `SKIPPED`.
5. Each parsed JSON loads back into a valid `DoclingDocument` (round-trip check).
6. Every parsed object records the exact `docling_version`.
7. A deliberately broken `doc_link` produces one `FAILED` row and does not stop the run.

---

## 7. Forward hooks (recorded now, used later — do NOT build here)

- `sha256` per doc → the change-detection key for the future reconcile loop.
- `doc_name` as stable business key → the namespace for future deterministic chunk IDs (`{doc_name}#{section_slug}#{sha(chunk_text)[:12]}`).
- Docling's page provenance in the parsed JSON → later maps a retrieved chunk to its source page, i.e. straight to the FinanceBench gold label `evidence_page_num`.

These are the only three seams this spec leaves open on purpose.
