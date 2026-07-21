"""Mypy regression tests for the public native-memory integration types."""

import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).with_name("typing_fixtures")


def _run_mypy(fixture: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mypy", "--no-incremental", "--show-error-codes", str(FIXTURES / fixture)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_package_root_types_compose_with_memory_manager() -> None:
    """Documented root imports retain precise types and factory lists remain composable."""
    result = _run_mypy("valid.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Success: no issues found" in result.stdout


def test_absent_optional_methods_are_not_statically_callable() -> None:
    """Protocol conformance must not advertise unsupported runtime capabilities."""
    result = _run_mypy("reject_absent_methods.py")
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert output.count('"Never" not callable  [misc]') == 3, output
