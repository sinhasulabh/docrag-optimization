from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace


class VertexVectorStore:
    """VectorStore backed by an already-deployed Vertex AI Vector Search Index Endpoint.

    Upserting is NOT done through this class -- that happens against the Index resource
    directly via index/vertex_admin.py::upsert_all() at build_offline time (automatic,
    storage-cost-only). This class only wraps the deployed, queryable endpoint, which only
    exists after the explicit, costed, manual deploy_vertex_index.py step.

    CAVEAT, stated rather than hidden: the exact sign/ordering convention of the `distance`
    Vertex returns for DOT_PRODUCT_DISTANCE has not been verified against a real deployed
    endpoint (deploying one costs money and was deliberately not done as part of writing this
    code -- see pipeline/deploy_vertex_index.py). The raw `distance` is passed through as
    `score` unmodified; confirm empirically (a query against a few known-similar/dissimilar
    vectors) whether higher-distance-is-more-similar or the reverse before trusting fused
    hybrid rankings that mix this score with BM25 scores.
    """

    def __init__(
        self,
        endpoint: aiplatform.MatchingEngineIndexEndpoint,
        *,
        deployed_index_id: str,
        index_type: str,
        ann_search_param: int,
    ):
        self._endpoint = endpoint
        self._deployed_index_id = deployed_index_id
        self._index_type = index_type
        self._ann_search_param = ann_search_param

    def upsert(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        raise NotImplementedError(
            "upsert against a deployed VertexVectorStore is not supported -- use "
            "index/vertex_admin.py::upsert_all() against the Index resource at "
            "build_offline time instead"
        )

    def search(
        self, query_vector: list[float], k: int, filters: dict | None = None
    ) -> list[tuple[str, float]]:
        kwargs: dict = dict(
            deployed_index_id=self._deployed_index_id, queries=[query_vector], num_neighbors=k
        )
        if self._index_type != "flat":
            # ann_search_param is percent-of-leaves-scanned (1-100) per the spec's
            # RetrievalConfig comment; Vertex's real knob is a 0-1 fraction.
            kwargs["fraction_leaf_nodes_to_search_override"] = max(
                0.01, min(1.0, self._ann_search_param / 100.0)
            )
        if filters:
            kwargs["filter"] = [Namespace(name=k_, allow_tokens=[v_]) for k_, v_ in filters.items()]

        results = self._endpoint.find_neighbors(**kwargs)
        neighbors = results[0] if results else []
        return [(n.id, float(n.distance)) for n in neighbors]
