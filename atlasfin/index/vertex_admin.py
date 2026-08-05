import logging

from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_config import (
    DistanceMeasureType,
)
from google.cloud.aiplatform_v1.types import index as index_types

logger = logging.getLogger("atlasfin.index.vertex_admin")

UPSERT_BATCH_SIZE = 100
# Google's own ScaNN/Tree-AH default-ish starting points; deliberately conservative for a
# small (thousands-of-chunks) corpus rather than tuned for scale.
DEFAULT_LEAF_NODE_EMBEDDING_COUNT = 500
DEFAULT_LEAF_NODES_TO_SEARCH_PERCENT = 10


def distance_measure_for(normalize: bool) -> DistanceMeasureType:
    return (
        DistanceMeasureType.DOT_PRODUCT_DISTANCE
        if normalize
        else DistanceMeasureType.COSINE_DISTANCE
    )


def find_index_by_display_name(
    display_name: str, *, project: str, location: str
) -> aiplatform.MatchingEngineIndex | None:
    matches = aiplatform.MatchingEngineIndex.list(
        filter=f'display_name="{display_name}"', project=project, location=location
    )
    return matches[0] if matches else None


def create_index(
    display_name: str,
    *,
    dimensions: int,
    index_type: str,
    normalize: bool,
    project: str,
    location: str,
) -> aiplatform.MatchingEngineIndex:
    """index_type='flat' -> exact Brute-Force index. 'hnsw'/'ivf' both collapse onto
    Vertex's one approximate algorithm (Tree-AH/ScaNN) -- it does not expose either as a
    literal separate option. index_update_method='STREAM_UPDATE' is required to allow
    incremental upsert_datapoints() afterward (the alternative, BATCH_UPDATE, only supports
    re-importing the entire corpus from a GCS contents_delta_uri).
    """
    distance = distance_measure_for(normalize)
    if index_type == "flat":
        return aiplatform.MatchingEngineIndex.create_brute_force_index(
            display_name=display_name,
            dimensions=dimensions,
            distance_measure_type=distance,
            index_update_method="STREAM_UPDATE",
            project=project,
            location=location,
            sync=True,
        )
    if index_type in ("hnsw", "ivf"):
        return aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=display_name,
            dimensions=dimensions,
            distance_measure_type=distance,
            leaf_node_embedding_count=DEFAULT_LEAF_NODE_EMBEDDING_COUNT,
            leaf_nodes_to_search_percent=DEFAULT_LEAF_NODES_TO_SEARCH_PERCENT,
            index_update_method="STREAM_UPDATE",
            project=project,
            location=location,
            sync=True,
        )
    raise ValueError(f"retrieval.index_type must be hnsw|ivf|flat, got {index_type!r}")


def create_or_get_index(
    display_name: str,
    *,
    dimensions: int,
    index_type: str,
    normalize: bool,
    project: str,
    location: str,
) -> aiplatform.MatchingEngineIndex:
    existing = find_index_by_display_name(display_name, project=project, location=location)
    if existing is not None:
        logger.info("vertex_admin: reusing existing index %s", display_name)
        return existing
    logger.info("vertex_admin: creating index %s (this costs storage only, no deploy)", display_name)
    return create_index(
        display_name,
        dimensions=dimensions,
        index_type=index_type,
        normalize=normalize,
        project=project,
        location=location,
    )


def upsert_all(
    index: aiplatform.MatchingEngineIndex,
    chunk_ids: list[str],
    vectors: list[list[float]],
    *,
    metadata: dict[str, dict[str, str]] | None = None,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """metadata (optional): chunk_id -> {restrict_name: value}, written as Vertex 'restricts'
    for RetrievalConfig.filters_enabled-style metadata pre-filtering.
    """
    metadata = metadata or {}
    for i in range(0, len(chunk_ids), batch_size):
        batch_ids = chunk_ids[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]
        datapoints = []
        for cid, vec in zip(batch_ids, batch_vecs):
            restricts = [
                index_types.IndexDatapoint.Restriction(namespace=k, allow_list=[v])
                for k, v in metadata.get(cid, {}).items()
            ]
            datapoints.append(
                index_types.IndexDatapoint(
                    datapoint_id=cid, feature_vector=vec, restricts=restricts
                )
            )
        index.upsert_datapoints(datapoints=datapoints)
    logger.info("vertex_admin: upserted %d datapoints into %s", len(chunk_ids), index.display_name)


def find_endpoint_by_display_name(
    display_name: str, *, project: str, location: str
) -> aiplatform.MatchingEngineIndexEndpoint | None:
    matches = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f'display_name="{display_name}"', project=project, location=location
    )
    return matches[0] if matches else None


def deploy_index(
    index: aiplatform.MatchingEngineIndex,
    *,
    display_name: str,
    deployed_index_id: str,
    project: str,
    location: str,
    machine_type: str = "e2-standard-2",
    min_replica_count: int = 1,
    max_replica_count: int = 1,
) -> aiplatform.MatchingEngineIndexEndpoint:
    """THE costed step -- deploys a persistently-billed, always-on Index Endpoint. Only ever
    called from pipeline/deploy_vertex_index.py's interactive, double-confirmed CLI. Never
    call this from build_offline/build_online.
    """
    endpoint = find_endpoint_by_display_name(display_name, project=project, location=location)
    if endpoint is None:
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=display_name,
            public_endpoint_enabled=True,
            project=project,
            location=location,
            sync=True,
        )
    endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_index_id,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        sync=True,
    )
    return endpoint


def undeploy_index(
    endpoint: aiplatform.MatchingEngineIndexEndpoint, *, deployed_index_id: str
) -> None:
    endpoint.undeploy_index(deployed_index_id=deployed_index_id)
