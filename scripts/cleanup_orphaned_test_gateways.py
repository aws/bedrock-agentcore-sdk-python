#!/usr/bin/env python3
"""Clean up orphaned AgentCore Gateway resources left behind by integration tests.

The gateway integration tests run in a shared AWS account. Gateways and their
targets are normally torn down, but interrupted or failed runs leave orphans
behind (a target that fails to reach READY can leave the parent gateway
undeleted). These accumulate over time and clutter the account.

This script lists every gateway in the account and deletes the ones that look
like orphaned test fixtures, deleting their targets first so the gateway delete
succeeds.

Safety:
  * Dry-run by default. Pass ``--apply`` to actually delete.
  * Only deletes gateways whose name starts with a known test prefix AND that are
    older than ``--min-age-days`` (default 1 day), so it never races a live run.

Usage:
    python scripts/cleanup_orphaned_test_gateways.py                 # dry run
    python scripts/cleanup_orphaned_test_gateways.py --apply         # delete
    python scripts/cleanup_orphaned_test_gateways.py --region us-west-2 --min-age-days 2 --apply
"""

import sys
import time

import boto3
from _cleanup_utils import build_parser, paginate, run_cleanup

# Gateway name prefix created by the integration test suites. Only gateways whose
# name starts with this are eligible for deletion.
TEST_PREFIX = "sdk-integ-"


def delete_gateway(client, gateway_id, timeout_s=120, poll_s=5):
    """Delete a gateway's targets, wait for the async deletes to settle, then delete it.

    The parent gateway cannot be deleted until its targets are fully removed.
    """
    def target_ids():
        return [t["targetId"] for t in paginate(client, "list_gateway_targets", "items",
                                                 gatewayIdentifier=gateway_id, maxResults=100)]

    for tid in target_ids():
        client.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=tid)

    deadline = time.monotonic() + timeout_s
    while target_ids():
        if time.monotonic() > deadline:
            raise TimeoutError(f"targets for gateway {gateway_id} not deleted within {timeout_s}s")
        time.sleep(poll_s)

    client.delete_gateway(gatewayIdentifier=gateway_id)


def main() -> int:
    args = build_parser(__doc__, "gateways").parse_args()
    client = boto3.client("bedrock-agentcore-control", region_name=args.region)

    return run_cleanup(
        "gateways",
        list(paginate(client, "list_gateways", "items", maxResults=100)),
        label_of=lambda g: f"{g.get('name', '')} ({g.get('gatewayId')})",
        created_of=lambda g: g.get("createdAt"),
        is_test=lambda g: g.get("name", "").startswith(TEST_PREFIX),
        delete_one=lambda g: delete_gateway(client, g.get("gatewayId")),
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())
