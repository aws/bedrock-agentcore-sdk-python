"""Shared helpers for the orphaned integration-test resource cleanup scripts.

Both cleanup scripts (memories, gateways) share the same shape: list everything
in a shared test account, keep what's protected/non-test/too-recent, and delete
the rest — dry-run by default. This module holds that common scaffolding so each
script only declares its resource-specific bits (prefixes, list/delete calls).
"""

import argparse
import datetime


def build_parser(description, noun):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--region", default="us-west-2", help="AWS region (default: us-west-2)")
    parser.add_argument(
        "--min-age-days",
        type=float,
        default=1.0,
        help=f"Only delete {noun} older than this many days (default: 1)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete (default is dry-run)")
    return parser


def paginate(client, operation, items_key, **kwargs):
    """Yield all items across pages of a list_* operation that uses nextToken."""
    token = None
    while True:
        page = getattr(client, operation)(**kwargs, **({"nextToken": token} if token else {}))
        yield from page.get(items_key, [])
        token = page.get("nextToken")
        if not token:
            return


def run_cleanup(noun, items, *, label_of, created_of, is_test, delete_one, args, is_protected=None):
    """Bucket items into kept/deleted, print a report, and delete (unless dry-run)."""
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.min_age_days)

    to_delete, protected, recent, nonmatch = [], [], [], []
    for item in items:
        if is_protected and is_protected(item):
            protected.append(item)
        elif not is_test(item):
            nonmatch.append(item)
        elif created_of(item) and created_of(item) > cutoff:
            recent.append(item)
        else:
            to_delete.append(item)

    print(f"Region: {args.region}")
    print(f"Total {noun}: {len(items)}")
    if is_protected:
        print(f"  protected (kept):        {len(protected)}")
    print(f"  non-test (kept):         {len(nonmatch)}")
    print(f"  too recent (kept):       {len(recent)} (younger than {args.min_age_days}d)")
    print(f"  orphaned test (delete):  {len(to_delete)}")
    print()

    if not to_delete:
        print("Nothing to delete.")
        return 0

    print(f"=== {'DELETING' if args.apply else 'DRY RUN (would delete)'} {len(to_delete)} {noun} ===")
    failures = 0
    for item in to_delete:
        label = label_of(item)
        if not args.apply:
            print(f"  would delete: {label}")
            continue
        try:
            delete_one(item)
            print(f"  deleted: {label}")
        except Exception as e:  # noqa: BLE001 - best-effort cleanup
            failures += 1
            print(f"  FAILED:  {label} ({e})")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete.")
    elif failures:
        print(f"\nCompleted with {failures} failures.")
        return 1
    else:
        print(f"\nDeleted {len(to_delete)} orphaned {noun}.")
    return 0
