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

import argparse
import datetime
import sys
import time

import boto3

# Gateway name prefixes created by the integration test suites. Only gateways
# whose name starts with one of these are eligible for deletion.
TEST_PREFIXES = (
    "sdk-integ-kb-tgt-",
    "sdk-integ-",
    "test-gateway",
    "integ-test",
)


def is_test_gateway(name: str) -> bool:
    return name.startswith(TEST_PREFIXES)


def list_all_gateways(client):
    gateways = []
    token = None
    while True:
        kwargs = {"maxResults": 100}
        if token:
            kwargs["nextToken"] = token
        resp = client.list_gateways(**kwargs)
        gateways.extend(resp.get("items", []))
        token = resp.get("nextToken")
        if not token:
            break
    return gateways


def _list_target_ids(client, gateway_id):
    ids = []
    token = None
    while True:
        kwargs = {"gatewayIdentifier": gateway_id, "maxResults": 100}
        if token:
            kwargs["nextToken"] = token
        resp = client.list_gateway_targets(**kwargs)
        ids.extend(t["targetId"] for t in resp.get("items", []) if t.get("targetId"))
        token = resp.get("nextToken")
        if not token:
            break
    return ids


def delete_gateway_targets(client, gateway_id, timeout_s=120, poll_s=5):
    """Delete all targets of a gateway and wait until they are gone.

    Target deletion is asynchronous; the parent gateway cannot be deleted until
    its targets are fully removed, so poll until the target list is empty.
    """
    for tid in _list_target_ids(client, gateway_id):
        client.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=tid)

    deadline = time.monotonic() + timeout_s
    while _list_target_ids(client, gateway_id):
        if time.monotonic() > deadline:
            raise TimeoutError(f"targets for gateway {gateway_id} not deleted within {timeout_s}s")
        time.sleep(poll_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-west-2", help="AWS region (default: us-west-2)")
    parser.add_argument(
        "--min-age-days",
        type=float,
        default=1.0,
        help="Only delete gateways older than this many days (default: 1)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete (default is dry-run)")
    args = parser.parse_args()

    client = boto3.client("bedrock-agentcore-control", region_name=args.region)

    gateways = list_all_gateways(client)
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=args.min_age_days)

    to_delete, skipped_recent, skipped_nonmatch = [], [], []
    for g in gateways:
        name = g.get("name", "")
        gid = g.get("gatewayId")
        created = g.get("createdAt")
        if not is_test_gateway(name):
            skipped_nonmatch.append(name)
        elif created and created > cutoff:
            skipped_recent.append(name)
        else:
            to_delete.append((gid, name))

    print(f"Region: {args.region}")
    print(f"Total gateways: {len(gateways)}")
    print(f"  non-test (kept):         {len(skipped_nonmatch)}")
    print(f"  too recent (kept):       {len(skipped_recent)} (younger than {args.min_age_days}d)")
    print(f"  orphaned test (delete):  {len(to_delete)}")
    print()

    if not to_delete:
        print("Nothing to delete.")
        return 0

    mode = "DELETING" if args.apply else "DRY RUN (would delete)"
    print(f"=== {mode} {len(to_delete)} gateways ===")
    failures = 0
    for gid, name in to_delete:
        if args.apply:
            try:
                delete_gateway_targets(client, gid)
                client.delete_gateway(gatewayIdentifier=gid)
                print(f"  deleted: {name} ({gid})")
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                failures += 1
                print(f"  FAILED:  {name} ({gid}) ({e})")
        else:
            print(f"  would delete: {name} ({gid})")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete.")
    elif failures:
        print(f"\nCompleted with {failures} failures.")
        return 1
    else:
        print(f"\nDeleted {len(to_delete)} orphaned gateways.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
