"""Tests for the shared AsciiDoc renderer (scripts/render_adoc.py).

These lock in the rendering properties that were hard-won during validation:
no leftover markdown code fences, balanced ``----`` source-block delimiters, no
``self``/``cls`` noise, and no duplicated summary/description text.
"""

import importlib.util
from pathlib import Path

import pytest

# scripts/ is not a package; load render_adoc.py directly by path.
_RENDER_PATH = Path(__file__).resolve().parents[3] / "scripts" / "render_adoc.py"
_spec = importlib.util.spec_from_file_location("render_adoc", _RENDER_PATH)
render_adoc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_adoc)


def _render(entry, level=2):
    out = []
    render_adoc.render_entry(entry, level, out)
    return "\n".join(out)


def _entry(**overrides):
    base = {
        "kind": "class",
        "name": "Widget",
        "signature": "Widget(name: str)",
        "summary": "A widget.",
        "description": "",
        "params": [],
        "returns": None,
        "raises": [],
        "examples": [],
        "members": [],
    }
    base.update(overrides)
    return base


class TestRenderProse:
    def test_fenced_block_becomes_source_block(self):
        text = "Intro line.\n\n```python\nx = 1\n```"
        out = "\n".join(render_adoc.render_prose(text))
        assert "```" not in out
        assert "[source,python]" in out
        assert "x = 1" in out

    def test_indented_fence_is_handled(self):
        text = "Example:\n\n    ```python\n    y = 2\n    ```"
        out = "\n".join(render_adoc.render_prose(text))
        assert "```" not in out
        # dedented inside the source block
        assert "\ny = 2" in "\n" + out

    def test_plain_prose_passthrough(self):
        assert render_adoc.render_prose("just text") == ["just text"]


class TestRenderEntry:
    def test_no_fence_leaks_in_description(self):
        adoc = _render(_entry(description="See:\n\n```python\nfoo()\n```"))
        assert "```" not in adoc
        assert "[source,python]" in adoc

    def test_balanced_source_delimiters(self):
        adoc = _render(_entry(examples=[{"lang": "python", "code": "a = 1"}]))
        assert adoc.count("----") % 2 == 0

    def test_summary_and_description_not_duplicated(self):
        adoc = _render(_entry(summary="A widget.", description="More detail."))
        assert adoc.count("A widget.") == 1
        assert adoc.count("More detail.") == 1

    def test_params_render_as_definition_list(self):
        adoc = _render(_entry(params=[{"name": "name", "type": "str", "required": True, "description": "The name."}]))
        assert "`name`" in adoc
        assert "The name." in adoc

    def test_example_stray_fence_stripped(self):
        # A closing fence plus trailing prose swept into the example must not leak.
        code = "a = 1\n```\nNotes: not code."
        adoc = _render(_entry(examples=[{"lang": "python", "code": code}]))
        assert "```" not in adoc
        assert "Notes: not code." not in adoc


class TestCleanExampleCode:
    def test_strips_fences_and_trailing_prose(self):
        cleaned = render_adoc.clean_example_code("x = 1\n```\nTrailing prose.")
        assert "```" not in cleaned
        assert "Trailing prose." not in cleaned
        assert "x = 1" in cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
