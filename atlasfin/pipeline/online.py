from dataclasses import dataclass

from atlasfin.config.schema import ExperimentConfig
from atlasfin.reranking import build as build_reranker
from atlasfin.reranking.interface import Reranker
from atlasfin.retrieval import build as build_retriever
from atlasfin.retrieval.interface import Retriever

from .offline import IndexHandle


@dataclass
class OnlineComponents:
    retriever: Retriever
    reranker: Reranker | None  # None when reranking.enabled=False


def build_online(
    cfg: ExperimentConfig,
    index: IndexHandle,
    *,
    gcp_project: str | None = None,
    gcp_location: str = "us-central1",
) -> OnlineComponents:
    """Deliberately a small named dataclass, not the spec pseudocode's literal
    `Retriever | Reranker` union return type -- a caller having to isinstance-check a union
    is worse ergonomics than one dataclass with two named fields.
    """
    retriever = build_retriever(
        cfg.retrieval,
        embedding_cfg=cfg.embedding,
        chunking_cfg=cfg.chunking,
        chunks_path=index.chunks_path,
        bm25_path=index.bm25_path if cfg.retrieval.mode == "hybrid" else None,
        is_vertex_deployed=index.is_vertex_deployed,
        index_endpoint_resource_name=index.index_endpoint_resource_name,
        deployed_index_id=index.deployed_index_id,
        gcp_project=gcp_project,
        gcp_location=gcp_location,
    )
    reranker = build_reranker(cfg.reranking)
    return OnlineComponents(retriever=retriever, reranker=reranker)
