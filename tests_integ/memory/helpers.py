import time

import pytest


def poll_until(fn, predicate, max_wait=180, poll_interval=10, fail_if=None):
    """Poll fn() until predicate(result) is True or max_wait exceeded."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        result = fn()
        if fail_if and fail_if(result):
            pytest.fail(f"poll_until hit terminal state: {result}")
        if predicate(result):
            return result
        time.sleep(poll_interval)
    pytest.fail(f"poll_until timed out after {max_wait}s")
