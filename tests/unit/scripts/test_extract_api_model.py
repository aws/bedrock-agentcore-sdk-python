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
        *args: Positional arguments forwarded to the API.
        **kwargs: Arguments forwarded to the API.

    Returns:
        A list of spans.

    Raises:
        ValueError: If the dataset is empty or all scenarios fail during
            execution.

    Example (Runtime agent):
        >>> run("runtime")
        >>> run("runtime-again")

    Example (Custom agent):
        >>> run("custom")
    """

    parsed = extract_api_model.parse_google_docstring(doc)

    assert [param["name"] for param in parsed["params"]] == ["wait_config", "*args", "**kwargs"]
    assert [param["required"] for param in parsed["params"]] == [True, False, False]
    assert parsed["returns"]["description"] == "A list of spans."
    assert parsed["raises"][0]["description"] == ("If the dataset is empty or all scenarios fail during execution.")
    assert [example["code"] for example in parsed["examples"]] == [
        '>>> run("runtime")\n>>> run("runtime-again")',
        '>>> run("custom")',
    ]


def test_public_class_summaries_use_action_verbs():
    class Actor:
        """Represents an actor within a session."""

    Actor.__module__ = "bedrock_agentcore.memory"

    entry = extract_api_model.entry_from_object("Actor", Actor)

    assert entry["summary"].startswith("Provides a handle")


def test_internal_batch_size_is_removed_from_public_method_description():
    def delete_all_long_term_memories_in_namespace():
        """Delete all records.

        This method processes records in chunks of 100.
        """

    entry = extract_api_model.entry_from_object(
        "delete_all_long_term_memories_in_namespace",
        delete_all_long_term_memories_in_namespace,
    )

    assert entry["summary"] == "Deletes all long-term memory records in the specified namespace."
    assert entry["description"] == "Retrieves all records and deletes them in batches."
