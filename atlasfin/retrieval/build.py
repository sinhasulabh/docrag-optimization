import os
from pathlib import Path

from google.cloud import aiplatform

from atlasfin.config.schema import ChunkingConfig, EmbeddingConfig, RetrievalConfig
from atlasfin.embedding.build import build as build_embedder
from atlasfin.index.bm25_store import BM25Store
from atlasfin.index.chunk_store import ChunkStore
from atlasfin.index.local_vector_store import LocalBruteForceVectorStore
from atlasfin.index.vertex_vector_store import VertexVectorStore

from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .interface import Retriever
from .query_transform import DecomposeTransform, HyDETransform, RewriteTransform


class VertexIndexNotDeployedError(RuntimeError):
    pass


def _default_query_transform_cache_path(query_model: str) -> Path:
    root = Path(os.environ.get("ATLASFIN_CACHE_DIR", ".atlasfin_cache"))
    safe_model = query_model.replace("/", "__")
    return root / "query_transform" / f"{safe_model}.jsonl"


def _build_transformer(cfg: RetrievalConfig):
    cache_path = _default_query_transform_cache_path(cfg.query_model)
    if cfg.query_transform == "rewrite":
        return RewriteTransform(cfg.query_model, cache_path)
    if cfg.query_transform == "decompose":
        return DecomposeTransform(cfg.query_model, cache_path)
    if cfg.query_transform == "hyde":
        return HyDETransform(cfg.query_model, cache_path)
    raise ValueError(f"retrieval.query_transform={cfg.query_transform!r} not supported")


def build(
    cfg: RetrievalConfig,
    *,
    embedding_cfg: EmbeddingConfig,
    chunking_cfg: ChunkingConfig,
    chunks_path: str | Path,
    bm25_path: str | Path | None,
    is_vertex_deployed: bool,
    index_endpoint_resource_name: str | None = None,
    deployed_index_id: str | None = None,
    gcp_project: str | None = None,
    gcp_location: str = "us-central1",
) -> Retriever:
    """Deliberately takes explicit unpacked fields rather than a pipeline.offline.IndexHandle
    object -- retrieval/ must not depend on pipeline/'s types (pipeline/ depends on
    retrieval/, not the other way around), per the separability constraint each component
    folder is supposed to honor.
    """
    if not is_vertex_deployed:
        raise VertexIndexNotDeployedError(
            "dense retrieval needs a deployed Vertex AI Vector Search endpoint, but none is "
            "deployed for this offline artifact. Run: python -m atlasfin.pipeline."
            "deploy_vertex_index --fingerprint <fp> --confirm (this is a real-money, "
            "explicitly-gated step -- see index/vertex_admin.py)"
        )

    embedder = build_embedder(embedding_cfg)
    chunk_store = ChunkStore.load(chunks_path)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=index_endpoint_resource_name, project=gcp_project, location=gcp_location
    )
    vector_store = VertexVectorStore(
        endpoint,
        deployed_index_id=deployed_index_id,
        index_type=cfg.index_type,
        ann_search_param=cfg.ann_search_param,
    )

    if cfg.mode == "dense":
        base: Retriever = DenseRetriever(embedder, vector_store, chunk_store, cfg, chunking_cfg)
    elif cfg.mode == "hybrid":
        if bm25_path is None:
            raise ValueError("retrieval.mode='hybrid' requires a built BM25 index (bm25_path)")
        bm25_store = BM25Store.load(bm25_path)
        base = HybridRetriever(embedder, vector_store, bm25_store, chunk_store, cfg, chunking_cfg)
    else:
        raise ValueError(f"retrieval.mode must be dense|hybrid, got {cfg.mode!r}")

    if cfg.query_transform == "none":
        return base

    from .query_transform import TransformingRetriever

    return TransformingRetriever(base, _build_transformer(cfg))


def build_local_smoke(
    cfg: RetrievalConfig,
    *,
    embedding_cfg: EmbeddingConfig,
    chunking_cfg: ChunkingConfig,
    chunk_store: ChunkStore,
    vector_store: LocalBruteForceVectorStore,
    bm25_store: BM25Store | None = None,
) -> Retriever:
    """The zero-Vertex-spend path: wires a DenseRetriever/HybridRetriever directly against
    an already-populated LocalBruteForceVectorStore, bypassing build()'s Vertex-only factory
    entirely. This is what the layer-3 end-to-end smoke test (real GCS + real Voyage
    embeddings, real gold questions, no deployed Vertex endpoint) uses.
    """
    embedder = build_embedder(embedding_cfg)
    if cfg.mode == "dense":
        return DenseRetriever(embedder, vector_store, chunk_store, cfg, chunking_cfg)
    if cfg.mode == "hybrid":
        if bm25_store is None:
            raise ValueError("retrieval.mode='hybrid' requires a bm25_store")
        return HybridRetriever(embedder, vector_store, bm25_store, chunk_store, cfg, chunking_cfg)
    raise ValueError(f"retrieval.mode must be dense|hybrid, got {cfg.mode!r}")
