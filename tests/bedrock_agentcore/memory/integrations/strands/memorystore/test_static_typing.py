"""Mypy regression tests for the public native-memory integration types."""

import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).with_name("typing_fixtures")


def _run_mypy(fixture: str) -> subprocess.CompletedProcess[str]:
    # Installed dependency stubs target this interpreter; Python 3.10 is
    # checked by its own CI matrix entry.
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--python-version",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            "--no-incremental",
            "--show-error-codes",
            str(FIXTURES / fixture),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_memorystore_package_types_compose_with_memory_manager() -> None:
    """Documented canonical imports retain precise types and factory lists remain composable."""
    result = _run_mypy("valid.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Success: no issues found" in result.stdout


def test_absent_optional_methods_are_not_statically_callable() -> None:
    """Protocol conformance must not advertise unsupported runtime capabilities."""
    result = _run_mypy("reject_absent_methods.py")
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert output.count('"Never" not callable  [misc]') == 3, output
