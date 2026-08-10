from dataclasses import dataclass, field


@dataclass
class ParsingConfig:  # all [offline]
    backend: str = "docling"
    ocr: str = "auto"  # off | on | auto (auto: OCR only image-only pages)
    table_mode: str = "structured"  # structured (TableFormer) | text
    keep_images: bool = False


@dataclass
class ChunkingConfig:  # all [offline]
    strategy: str = "structure_aware"  # structure_aware | recursive | fixed
    max_tokens: int = 512
    overlap_tokens: int = 64  # used by recursive/fixed
    contextual_prefix: bool = False  # per-chunk LLM context blurb at ingestion
    contextual_model: str | None = None
    metadata_fields: tuple = ("section", "pages", "doc_type", "doc_period", "company")


@dataclass
class EmbeddingConfig:  # all [offline]
    model_id: str = "voyage-4"  # general-purpose; voyage-finance-2 is the finance-tuned alternative
    dimension: int | None = None  # MRL truncation; None = model native
    normalize: bool = True
    batch_size: int = 64


@dataclass
class RetrievalConfig:  # all [online]
    mode: str = "dense"  # dense | hybrid
    fusion: str = "rrf"  # rrf | weighted   (used when mode=hybrid)
    bm25_weight: float = 0.5  # used when fusion=weighted
    top_k: int = 20  # candidate pool size (recall ceiling)
    filters_enabled: bool = False  # metadata pre-filtering
    parent_child: bool = False  # retrieve child, hand off parent
    query_transform: str = "none"  # none | rewrite | decompose | hyde
    query_model: str | None = None  # LLM for the transform
    # ANN index params
    index_type: str = "hnsw"  # hnsw | ivf | flat
    ann_search_param: int = 64  # ef_search (hnsw) / nprobe (ivf); Vertex: leaf-search percent


@dataclass
class RerankingConfig:  # all [online]
    enabled: bool = False
    model_id: str | None = None  # cross-encoder
    depth: int = 20  # how many candidates to rescore (<= top_k)
    max_pair_tokens: int = 512  # (query+chunk) budget; guard against truncation


@dataclass
class ExperimentConfig:
    name: str
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)

    def offline_fingerprint(self) -> str:
        from .fingerprint import offline_fingerprint

        return offline_fingerprint(self)
