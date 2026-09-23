"""Shared pytest configuration."""

import sys


def pytest_runtest_teardown(item):
    """Stop ragas' background telemetry thread once ragas is imported.

    The thread calls ``time.sleep(1)`` in a loop. Patching ``time.sleep`` in a
    test patches it for every thread, so the ragas thread's calls land on the
    mock and break call-count assertions.
    """
    analytics = sys.modules.get("ragas._analytics")
    if analytics is not None:
        analytics._analytics_batcher._running = False
