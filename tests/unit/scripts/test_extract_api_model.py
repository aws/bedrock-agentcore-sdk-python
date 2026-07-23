"""Tests for the Python API doc-model extractor."""

import importlib.util
from pathlib import Path

_EXTRACT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "extract_api_model.py"
_spec = importlib.util.spec_from_file_location("extract_api_model", _EXTRACT_PATH)
extract_api_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_api_model)


def test_multiline_fields_and_named_examples_are_preserved():
    doc = """Run an evaluation.

    Args:
        wait_config: Optional WaitConfig for polling behavior.
        **kwargs: Arguments forwarded to the API.

    Returns:
        A list of spans.

    Raises:
        ValueError: If the dataset is empty or all scenarios fail during
            execution.

    Example (Runtime agent):
        >>> run("runtime")

    Example (Custom agent):
        >>> run("custom")
    """

    parsed = extract_api_model.parse_google_docstring(doc)

    assert [param["name"] for param in parsed["params"]] == ["wait_config", "**kwargs"]
    assert parsed["returns"]["description"] == "A list of spans."
    assert parsed["raises"][0]["description"] == ("If the dataset is empty or all scenarios fail during execution.")
    assert [example["code"] for example in parsed["examples"]] == [
        '>>> run("runtime")',
        '>>> run("custom")',
    ]
