#!/usr/bin/env python3
"""Clean up orphaned AgentCore Memory resources left behind by integration tests.

The integration tests run in a shared AWS account. Memories created by tests are
normally torn down, but interrupted or failed runs leave orphans behind. Once the
account accumulates more than ~100 memories, ``test_list_memories`` (and any other
test that lists with the default page cap) can flake.

This script lists every memory in the account and deletes the ones that look like
orphaned test fixtures, while protecting long-lived memories that tests depend on
(e.g. the pre-populated memory referenced by the ``MEMORY_PREPOPULATED_ID`` secret).

Safety:
  * Dry-run by default. Pass ``--apply`` to actually delete.
  * Only deletes memories whose id starts with a known test prefix AND that are
    older than ``--min-age-days`` (default 1 day), so it never races a live run.
  * Never deletes protected memories (``--protect`` substrings).

Usage:
    python scripts/cleanup_orphaned_test_memories.py                 # dry run
    python scripts/cleanup_orphaned_test_memories.py --apply         # delete
    python scripts/cleanup_orphaned_test_memories.py --region us-west-2 --min-age-days 2 --apply
"""

import argparse
import datetime
import sys

import boto3

# Memory id prefixes created by the integration test suites. Only memories whose
# id starts with one of these are eligible for deletion.
TEST_PREFIXES = (
    "test_cp_",
    "mc_2026",
    "memory_",
    "test_memory",
    "sdk_integ",
    "integ_test",
)

# Substrings of memory ids that must never be deleted (referenced by CI secrets).
PROTECTED_SUBSTRINGS = (
    "prepopulated",
    "observability",
)


def is_protected(memory_id: str) -> bool:
    return any(s in memory_id for s in PROTECTED_SUBSTRINGS)


def is_test_memory(memory_id: str) -> bool:
    return memory_id.startswith(TEST_PREFIXES)


def list_all_memories(client):
    memories = []
    token = None
    while True:
        kwargs = {"maxResults": 100}
        if token:
            kwargs["nextToken"] = token
        resp = client.list_memories(**kwargs)
        memories.extend(resp.get("memories", []))
        token = resp.get("nextToken")
        if not token:
            break
    return memories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="us-west-2", help="AWS region (default: us-west-2)")
    parser.add_argument(
        "--min-age-days",
        type=float,
        default=1.0,
        help="Only delete memories older than this many days (default: 1)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete (default is dry-run)")
    args = parser.parse_args()

    client = boto3.client("bedrock-agentcore-control", region_name=args.region)

    memories = list_all_memories(client)
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=args.min_age_days)

    to_delete, skipped_protected, skipped_recent, skipped_nonmatch = [], [], [], []
    for m in memories:
        mid = m.get("id") or m.get("memoryId") or ""
        created = m.get("createdAt")
        if is_protected(mid):
            skipped_protected.append(mid)
        elif not is_test_memory(mid):
            skipped_nonmatch.append(mid)
        elif created and created > cutoff:
            skipped_recent.append(mid)
        else:
            to_delete.append(mid)

    print(f"Region: {args.region}")
    print(f"Total memories: {len(memories)}")
    print(f"  protected (kept):        {len(skipped_protected)}")
    print(f"  non-test (kept):         {len(skipped_nonmatch)}")
    print(f"  too recent (kept):       {len(skipped_recent)} (younger than {args.min_age_days}d)")
    print(f"  orphaned test (delete):  {len(to_delete)}")
    print()

    if not to_delete:
        print("Nothing to delete.")
        return 0

    mode = "DELETING" if args.apply else "DRY RUN (would delete)"
    print(f"=== {mode} {len(to_delete)} memories ===")
    failures = 0
    for mid in to_delete:
        if args.apply:
            try:
                client.delete_memory(memoryId=mid)
                print(f"  deleted: {mid}")
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                failures += 1
                print(f"  FAILED:  {mid} ({e})")
        else:
            print(f"  would delete: {mid}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete.")
    elif failures:
        print(f"\nCompleted with {failures} failures.")
        return 1
    else:
        print(f"\nDeleted {len(to_delete)} orphaned memories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
