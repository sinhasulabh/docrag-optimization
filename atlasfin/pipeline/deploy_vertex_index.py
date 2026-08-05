import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from google.cloud import aiplatform

from atlasfin.index import vertex_admin

from . import cache

logger = logging.getLogger("atlasfin.pipeline.deploy_vertex_index")

COST_WARNING = """
================================================================================
WARNING: this deploys a Vertex AI Vector Search Index Endpoint. That endpoint
is a CONTINUOUSLY-BILLED resource (a running compute node) -- it costs money
every hour it stays deployed, whether or not you're querying it. Check current
Vertex AI Vector Search pricing at https://cloud.google.com/vertex-ai/pricing
before proceeding (a hardcoded number here would go stale). Run this script's
`undeploy` subcommand promptly once you're done testing.
================================================================================
"""


def _vertex_state_path(fingerprint: str):
    return cache.offline_dir(fingerprint) / "vertex" / "index_state.json"


def _endpoint_state_path(fingerprint: str):
    return cache.offline_dir(fingerprint) / "vertex" / "endpoint_state.json"


def _load_vertex_state(fingerprint: str) -> dict:
    path = _vertex_state_path(fingerprint)
    if not path.exists():
        raise FileNotFoundError(
            f"no Vertex index state found for fingerprint {fingerprint!r} at {path} -- run "
            "build_offline with a gcp_project first to create+upsert the index"
        )
    return json.loads(path.read_text())


def cmd_deploy(args: argparse.Namespace) -> None:
    if not args.confirm:
        print("Refusing to deploy without --confirm (this is a real-money action).")
        sys.exit(1)
    print(COST_WARNING)
    answer = input(
        f"Deploy a Vertex AI Vector Search endpoint for fingerprint {args.fingerprint!r} "
        f"in project {args.gcp_project!r}? [y/N] "
    )
    if answer.strip().lower() != "y":
        print("Aborted.")
        sys.exit(1)

    state = _load_vertex_state(args.fingerprint)
    index = aiplatform.MatchingEngineIndex(
        index_name=state["resource_name"], project=args.gcp_project, location=args.gcp_location
    )

    deployed_index_id = args.deployed_index_id or f"atlasfin_{args.fingerprint.replace('-', '_')}"
    endpoint = vertex_admin.deploy_index(
        index,
        display_name=f"{state['display_name']}-endpoint",
        deployed_index_id=deployed_index_id,
        project=args.gcp_project,
        location=args.gcp_location,
        machine_type=args.machine_type,
        min_replica_count=args.min_replica_count,
        max_replica_count=args.max_replica_count,
    )

    endpoint_state = {
        "deployed": True,
        "vertex_display_name": state["display_name"],
        "endpoint_resource_name": endpoint.resource_name,
        "deployed_index_id": deployed_index_id,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _endpoint_state_path(args.fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(endpoint_state, indent=2))
    print(f"Deployed. Endpoint: {endpoint.resource_name}, deployed_index_id: {deployed_index_id}")
    print("Remember to run this script's `undeploy` subcommand when done to stop billing.")


def cmd_undeploy(args: argparse.Namespace) -> None:
    path = _endpoint_state_path(args.fingerprint)
    if not path.exists():
        print(f"No deployed endpoint recorded for fingerprint {args.fingerprint!r}.")
        sys.exit(1)
    endpoint_state = json.loads(path.read_text())
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=endpoint_state["endpoint_resource_name"],
        project=args.gcp_project,
        location=args.gcp_location,
    )
    vertex_admin.undeploy_index(endpoint, deployed_index_id=endpoint_state["deployed_index_id"])
    endpoint_state["deployed"] = False
    path.write_text(json.dumps(endpoint_state, indent=2))
    print(f"Undeployed {endpoint_state['deployed_index_id']} from {endpoint_state['endpoint_resource_name']}.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Deploy/undeploy a Vertex AI Vector Search index endpoint "
        "(COSTS MONEY continuously while deployed -- see index/vertex_admin.py)"
    )
    p.add_argument("--gcp-project", required=True)
    p.add_argument("--gcp-location", default="us-central1")
    sub = p.add_subparsers(dest="command", required=True)

    deploy_p = sub.add_parser("deploy")
    deploy_p.add_argument("--fingerprint", required=True)
    deploy_p.add_argument("--confirm", action="store_true")
    deploy_p.add_argument("--deployed-index-id", default=None)
    deploy_p.add_argument("--machine-type", default="e2-standard-2")
    deploy_p.add_argument("--min-replica-count", type=int, default=1)
    deploy_p.add_argument("--max-replica-count", type=int, default=1)
    deploy_p.set_defaults(func=cmd_deploy)

    undeploy_p = sub.add_parser("undeploy")
    undeploy_p.add_argument("--fingerprint", required=True)
    undeploy_p.set_defaults(func=cmd_undeploy)

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
