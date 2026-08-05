import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from google.cloud import storage

from atlasfin import chunking
from atlasfin import embedding as embedding_pkg
from atlasfin.chunking.source_lookup import load_source_lookup
from atlasfin.config.schema import ExperimentConfig
from atlasfin.contracts import Chunk
from atlasfin.index import vertex_admin
from atlasfin.index.bm25_store import BM25Store
from atlasfin.index.chunk_store import ChunkStore

from . import cache
from .corpus import resolve_parsed_docs, split_run_uri

logger = logging.getLogger("atlasfin.pipeline.offline")


@dataclass
class IndexHandle:
    offline_fingerprint: str
    offline_dir: Path
    chunks_path: Path
    embeddings_path: Path
    chunk_ids_path: Path
    bm25_path: Path
    n_chunks: int
    vertex_index_resource_name: str | None
    vertex_index_display_name: str | None
    is_vertex_deployed: bool
    deployed_index_id: str | None
    index_endpoint_resource_name: str | None


def _vertex_algo(index_type: str) -> str:
    # "hnsw" and "ivf" both collapse onto Vertex's one approximate algorithm (Tree-AH) --
    # see index/vertex_admin.py::create_index for the full mapping.
    return "flat" if index_type == "flat" else "tree_ah"


def _vertex_display_name(fp: str, cfg: ExperimentConfig) -> str:
    """Deliberately a SEPARATE key from the offline fingerprint: retrieval.index_type lives
    on RetrievalConfig (an ONLINE, non-fingerprinted field per the spec), but which physical
    Vertex index algorithm gets built is inescapably a structural choice. Two configs sharing
    the same offline fingerprint (same parsing/chunking/embedding) but different
    retrieval.index_type reuse the same local chunks/embeddings/BM25 artifacts, but need --
    and get -- different Vertex index resources.
    """
    algo = _vertex_algo(cfg.retrieval.index_type)
    distance = "dot" if cfg.embedding.normalize else "cosine"
    return f"atlasfin-{fp}-{algo}-{distance}"


def _cache_is_complete(offdir: Path) -> bool:
    return all(
        (offdir / name).exists()
        for name in ("manifest.json", "chunks.jsonl", "embeddings.npy", "chunk_ids.json")
    )


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_offline(
    cfg: ExperimentConfig,
    doc_names: list[str],
    *,
    ingest_run_uri: str,
    gcp_project: str | None = None,
    gcp_location: str = "us-central1",
    parse_device: str = "auto",
    parse_timeout_s: int = 300,
    force_rebuild: bool = False,
) -> IndexHandle:
    """parse -> chunk -> embed -> local BM25 -> (if gcp_project given) Vertex index
    create+upsert. Cached by cfg.offline_fingerprint(); rebuilds ONLY if that fingerprint (or
    the requested doc_names) changed. Note this extends the spec's literal build_offline(cfg)
    signature with doc_names/ingest_run_uri -- ExperimentConfig itself has no notion of which
    documents to index, so there's no way to call it as literally written.

    gcp_project=None deliberately skips Vertex index creation entirely (no GCP Vertex calls
    at all) -- this is what lets local dev/smoke-testing run the full parse/chunk/embed/BM25
    pipeline for free, verifying it with LocalBruteForceVectorStore instead.
    """
    fp = cfg.offline_fingerprint()
    offdir = cache.offline_dir(fp)
    offdir.mkdir(parents=True, exist_ok=True)
    chunks_path = offdir / "chunks.jsonl"
    embeddings_path = offdir / "embeddings.npy"
    chunk_ids_path = offdir / "chunk_ids.json"
    manifest_path = offdir / "manifest.json"
    bm25_dir = offdir / "bm25"
    vertex_state_path = offdir / "vertex" / "index_state.json"
    endpoint_state_path = offdir / "vertex" / "endpoint_state.json"
    scratch_dir = offdir / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    hit = (not force_rebuild) and _cache_is_complete(offdir)
    if hit:
        manifest = _load_json(manifest_path) or {}
        if set(manifest.get("doc_names", [])) != set(doc_names):
            logger.info(
                "pipeline.offline: fp=%s cache exists but doc_names changed, rebuilding", fp
            )
            hit = False

    if hit:
        logger.info("pipeline.offline: fingerprint=%s HIT, reusing cached artifacts", fp)
        n_chunks = len(json.loads(chunk_ids_path.read_text()))
    else:
        logger.info("pipeline.offline: fingerprint=%s MISS, building", fp)
        n_chunks = _build_offline_artifacts(
            cfg,
            doc_names,
            ingest_run_uri=ingest_run_uri,
            scratch_dir=scratch_dir,
            parse_device=parse_device,
            parse_timeout_s=parse_timeout_s,
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            chunk_ids_path=chunk_ids_path,
            bm25_dir=bm25_dir,
            manifest_path=manifest_path,
            fp=fp,
        )

    vertex_state = _load_json(vertex_state_path)
    endpoint_state = _load_json(endpoint_state_path)

    if gcp_project is not None:
        vertex_state = _ensure_vertex_index(
            cfg,
            fp,
            n_chunks=n_chunks,
            embeddings_path=embeddings_path,
            chunk_ids_path=chunk_ids_path,
            gcp_project=gcp_project,
            gcp_location=gcp_location,
            vertex_state=vertex_state,
            vertex_state_path=vertex_state_path,
        )
    else:
        logger.warning(
            "pipeline.offline: gcp_project not given, skipping Vertex index create/upsert -- "
            "dense retrieval needs a LocalBruteForceVectorStore instead"
        )

    is_deployed = bool(
        endpoint_state
        and endpoint_state.get("deployed")
        and vertex_state
        and endpoint_state.get("vertex_display_name") == vertex_state.get("display_name")
    )

    return IndexHandle(
        offline_fingerprint=fp,
        offline_dir=offdir,
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        chunk_ids_path=chunk_ids_path,
        bm25_path=bm25_dir,
        n_chunks=n_chunks,
        vertex_index_resource_name=(vertex_state or {}).get("resource_name"),
        vertex_index_display_name=(vertex_state or {}).get("display_name"),
        is_vertex_deployed=is_deployed,
        deployed_index_id=(endpoint_state or {}).get("deployed_index_id") if is_deployed else None,
        index_endpoint_resource_name=(
            (endpoint_state or {}).get("endpoint_resource_name") if is_deployed else None
        ),
    )


def _build_offline_artifacts(
    cfg: ExperimentConfig,
    doc_names: list[str],
    *,
    ingest_run_uri: str,
    scratch_dir: Path,
    parse_device: str,
    parse_timeout_s: int,
    chunks_path: Path,
    embeddings_path: Path,
    chunk_ids_path: Path,
    bm25_dir: Path,
    manifest_path: Path,
    fp: str,
) -> int:
    client = storage.Client()
    source_lookup = load_source_lookup(f"{ingest_run_uri}_manifest/source.jsonl")

    parsed_docs = resolve_parsed_docs(
        doc_names,
        ingest_run_uri=ingest_run_uri,
        source_lookup=source_lookup,
        gcs_client=client,
        parsing_cfg=cfg.parsing,
        scratch_dir=scratch_dir,
        parse_device=parse_device,
        parse_timeout_s=parse_timeout_s,
    )

    chunker = chunking.build(
        cfg.chunking, source_lookup=source_lookup, gcs_client=client, scratch_dir=scratch_dir
    )
    all_chunks: list[Chunk] = []
    for parsed in parsed_docs:
        all_chunks.extend(chunker.chunk(parsed))
    logger.info(
        "pipeline.offline: chunked %d docs into %d chunks", len(parsed_docs), len(all_chunks)
    )

    chunk_store = ChunkStore(all_chunks)
    chunk_store.save(chunks_path)

    embedder = embedding_pkg.build(cfg.embedding)
    vectors = embedder.embed_docs([c.embedding_text for c in all_chunks])
    np.save(embeddings_path, np.asarray(vectors, dtype=np.float32))
    chunk_ids_path.write_text(json.dumps([c.chunk_id for c in all_chunks]))

    bm25 = BM25Store()
    bm25.build({c.chunk_id: c.text for c in all_chunks})
    bm25.save(bm25_dir)

    manifest_path.write_text(
        json.dumps(
            {
                "offline_fingerprint": fp,
                "doc_names": doc_names,
                "n_chunks": len(all_chunks),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    return len(all_chunks)


def _ensure_vertex_index(
    cfg: ExperimentConfig,
    fp: str,
    *,
    n_chunks: int,
    embeddings_path: Path,
    chunk_ids_path: Path,
    gcp_project: str,
    gcp_location: str,
    vertex_state: dict | None,
    vertex_state_path: Path,
) -> dict:
    display_name = _vertex_display_name(fp, cfg)
    if (
        vertex_state
        and vertex_state.get("display_name") == display_name
        and vertex_state.get("upserted")
        and vertex_state.get("num_vectors") == n_chunks
    ):
        return vertex_state

    vectors = np.load(embeddings_path)
    dimensions = int(vectors.shape[1])
    index = vertex_admin.create_or_get_index(
        display_name,
        dimensions=dimensions,
        index_type=cfg.retrieval.index_type,
        normalize=cfg.embedding.normalize,
        project=gcp_project,
        location=gcp_location,
    )

    chunk_ids = json.loads(chunk_ids_path.read_text())
    vertex_admin.upsert_all(index, chunk_ids, vectors.tolist())

    state = {
        "display_name": display_name,
        "resource_name": index.resource_name,
        "upserted": True,
        "num_vectors": len(chunk_ids),
    }
    vertex_state_path.parent.mkdir(parents=True, exist_ok=True)
    vertex_state_path.write_text(json.dumps(state, indent=2))
    return state
