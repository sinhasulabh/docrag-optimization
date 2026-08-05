# AtlasFin — Pipeline Component Architecture & Experiment-Config Spec

**Goal:** structure the RAG pipeline as five independently-swappable components (Parsing, Chunking, Embedding, Retrieval, Reranking), where **every tuning knob is a config parameter**, so an experiment is a config change — not a code change — and its impact is measured by the eval harness.

**Two design constraints from the conversation, baked in:**
1. **Separability** — each stage is a folder behind an interface, so any one can later be lifted into its own service by promoting its interface to a network API. Nothing reaches across a boundary except the shared data contracts.
2. **Config-driven experimentation** — the pipeline is assembled *from config* by a factory. Changing a knob and re-running is the entire experiment loop.

**Non-goal:** generation. The generative model and prompting are owned by another team. This pipeline ends at "a ranked, context-complete set of chunks handed off." Everything here is retrieval-side.

---

## 1. Folder layout

```
atlasfin/
  contracts/        # shared data types ONLY. No logic. The stable interface between stages.
  config/           # config schema + experiment config files
  parsing/          # ParsingComponent: interface + Docling impl
  chunking/         # ChunkingComponent: interface + strategy impls
  embedding/        # EmbeddingComponent: interface + model impls
  retrieval/        # RetrievalComponent: interface + dense/hybrid/filter/query-transform impls
  reranking/        # RerankingComponent: interface + none/cross-encoder impls
  index/            # storage adapters (vector store, BM25 store) — the plumbing, not a "stage"
  pipeline/         # orchestrator: builds components from config, runs offline + online paths
  eval/             # harness: runs a config against the gold set, emits metrics
  experiments/      # experiment configs + their results
```

**The separability mechanism (how a folder becomes a service later):** every component folder exposes (a) one **interface** (Protocol/ABC), (b) one or more **implementations**, and (c) a **factory** `build(cfg) -> Interface`. Callers depend only on the interface and the `contracts/` types. To extract a stage into a microservice, you keep the interface as the wire contract and swap the in-process impl for an HTTP/gRPC client that satisfies the same interface. No caller changes. That is the whole point of the interface boundary.

---

## 2. Component interfaces

One method each, typed against `contracts/`. (Signatures, not implementations.)

```python
# contracts/  — reused from prior specs: RawObject, ParsedObject, Chunk
@dataclass(frozen=True)
class Candidate:                 # a retrieved (and later reranked) unit
    chunk_id: str
    score: float                 # retriever score, then overwritten by reranker score
    text: str                    # the text that was matched
    payload_text: str            # what gets handed off (may be the parent, if parent-child)
    pages: list[int]
    metadata: dict

class Parser(Protocol):
    def parse(self, raw: "RawObject") -> "ParsedObject": ...

class Chunker(Protocol):
    def chunk(self, parsed: "ParsedObject") -> list["Chunk"]: ...

class Embedder(Protocol):
    def embed_docs(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...      # may differ (task prefix)

class Retriever(Protocol):
    def retrieve(self, query: str, k: int, filters: dict | None) -> list["Candidate"]: ...

class Reranker(Protocol):
    def rerank(self, query: str, candidates: list["Candidate"], depth: int) -> list["Candidate"]: ...
```

---

## 3. The offline/online split (the cost structure that makes experiments fast)

Knobs fall into two classes, and the orchestrator treats them differently:

- **Offline knobs** (Parsing, Chunking, Embedding): changing one **invalidates the built index** and forces a rebuild of everything downstream of it. Expensive. Cache the artifacts keyed by a fingerprint of the *offline* config subset, so you only rebuild when an offline knob actually changed.
- **Online knobs** (Retrieval, Reranking): changing one costs nothing but the next query. Free to sweep. No rebuild.

**Invalidation cascade** (what a change forces you to redo):

| Change a knob in… | Must redo |
|---|---|
| Parsing | parse → chunk → embed → index (everything) |
| Chunking | chunk → embed → index |
| Embedding | embed → index |
| Retrieval | nothing offline — next query only |
| Reranking | nothing offline — next query only |

Practical consequence for experimentation: **sweep online knobs freely and often; batch offline knobs**, because each offline change pays a full (partial) re-index. The orchestrator enforces this via fingerprint-keyed caches so you never accidentally re-embed when you only changed `top_k`.

---

## 4. Config schema (every knob, grouped by stage)

Each field is annotated `[offline]` or `[online]`. Defaults are a sane starting baseline, not a recommendation — the point is to sweep them.

```python
@dataclass
class ParsingConfig:                       # all [offline]
    backend: str = "docling"
    ocr: str = "auto"                      # off | on | auto (auto: OCR only image-only pages)
    table_mode: str = "structured"         # structured (TableFormer) | text
    keep_images: bool = False

@dataclass
class ChunkingConfig:                       # all [offline]
    strategy: str = "structure_aware"      # structure_aware | recursive | fixed
    max_tokens: int = 512
    overlap_tokens: int = 64               # used by recursive/fixed
    contextual_prefix: bool = False        # per-chunk LLM context blurb at ingestion
    contextual_model: str | None = None
    metadata_fields: tuple = ("section", "pages", "doc_type", "doc_period", "company")

@dataclass
class EmbeddingConfig:                       # all [offline]
    model_id: str = "<decide — drives chunk tokenizer too>"
    dimension: int | None = None           # MRL truncation; None = model native
    normalize: bool = True
    batch_size: int = 64

@dataclass
class RetrievalConfig:                       # all [online]
    mode: str = "dense"                    # dense | hybrid
    fusion: str = "rrf"                    # rrf | weighted   (used when mode=hybrid)
    bm25_weight: float = 0.5               # used when fusion=weighted
    top_k: int = 20                        # candidate pool size (recall ceiling)
    filters_enabled: bool = False          # metadata pre-filtering
    parent_child: bool = False             # retrieve child, hand off parent
    query_transform: str = "none"          # none | rewrite | decompose | hyde
    query_model: str | None = None         # LLM for the transform
    # ANN index params
    index_type: str = "hnsw"               # hnsw | ivf | flat
    ann_search_param: int = 64             # ef_search (hnsw) / nprobe (ivf)

@dataclass
class RerankingConfig:                       # all [online]
    enabled: bool = False
    model_id: str | None = None            # cross-encoder
    depth: int = 20                        # how many candidates to rescore (<= top_k)
    max_pair_tokens: int = 512             # (query+chunk) budget; guard against truncation

@dataclass
class ExperimentConfig:
    name: str
    parsing: ParsingConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    reranking: RerankingConfig

    def offline_fingerprint(self) -> str:  # hash(parsing, chunking, embedding) -> cache key
        ...
```

---

## 5. Orchestrator & experiment runner

```python
def build_offline(cfg: ExperimentConfig) -> IndexHandle:
    # parse -> chunk -> embed -> index. Cached by cfg.offline_fingerprint().
    # Rebuilds ONLY if the offline subset changed.
    ...

def build_online(cfg: ExperimentConfig, index: IndexHandle) -> Retriever | Reranker:
    # wires retriever (+ optional reranker) from config against the built index.
    ...

def run_experiment(cfg: ExperimentConfig, gold_set) -> Metrics:
    index   = build_offline(cfg)                     # cached across online sweeps
    retr    = build_online(cfg, index)
    results = [answer(q, retr, cfg) for q in gold_set]   # retrieve (+ rerank), hand off context
    return eval_harness.score(results, gold_set)     # P1 metrics
```

`answer()` stops at "ranked context handed off" — it does **not** call a generation model (out of scope). The eval harness therefore scores the **retrieval-side** metrics only.

---

## 6. Eval hook (what "see the impact" measures)

The runner reports, per config, against the FinanceBench gold set:
- **recall@k** and **MRR** vs `evidence_page_num` (did the right page make the candidate pool / rank near top).
- **precision@k after rerank** (did reranking move the gold chunk up).
- **latency** (p50/p95) split into retrieve vs rerank, so a quality gain is always read against its latency cost.
- (answer-correctness / faithfulness / abstention live in the generation team's eval, downstream — noted, not owned here.)

An experiment = one `ExperimentConfig` → one row of these metrics. Sweeping a knob = a set of rows you compare. That is the entire "change it and see the impact" loop.

---

## 7. Acceptance criteria

1. Each of the five stages is importable and testable in isolation, depending only on its interface + `contracts/`.
2. Swapping an implementation (e.g. `dense` → `hybrid` retriever) is a config change, zero caller changes.
3. Changing an **online** knob triggers **no** re-index (proven: offline fingerprint unchanged → cached artifacts reused).
4. Changing an **offline** knob rebuilds exactly the invalidated suffix of the pipeline and nothing more.
5. `run_experiment(cfg)` emits a metrics row for any valid config, and two runs of the same config produce identical metrics (determinism).
6. A stage can be extracted to a service by replacing its impl with a client satisfying the same interface — no other folder changes.
```
