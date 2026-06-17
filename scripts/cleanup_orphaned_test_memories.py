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
  * Never deletes protected memories (see PROTECTED_SUBSTRINGS).

Usage:
    python scripts/cleanup_orphaned_test_memories.py                 # dry run
    python scripts/cleanup_orphaned_test_memories.py --apply         # delete
    python scripts/cleanup_orphaned_test_memories.py --region us-west-2 --min-age-days 2 --apply
"""

import sys

import boto3
from _cleanup_utils import build_parser, paginate, run_cleanup

# Memory id prefixes created by the integration test suites. Only memories whose
# id starts with one of these are eligible for deletion.
TEST_PREFIXES = ("test_cp_", "mc_2026", "memory_")

# Substrings of memory ids that must never be deleted (referenced by CI secrets).
PROTECTED_SUBSTRINGS = ("prepopulated", "observability")


def _id(memory):
    return memory.get("id") or memory.get("memoryId") or ""


def main() -> int:
    args = build_parser(__doc__, "memories").parse_args()
    client = boto3.client("bedrock-agentcore-control", region_name=args.region)

    return run_cleanup(
        "memories",
        list(paginate(client, "list_memories", "memories", maxResults=100)),
        label_of=_id,
        created_of=lambda m: m.get("createdAt"),
        is_test=lambda m: _id(m).startswith(TEST_PREFIXES),
        is_protected=lambda m: any(s in _id(m) for s in PROTECTED_SUBSTRINGS),
        delete_one=lambda m: client.delete_memory(memoryId=_id(m)),
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())
