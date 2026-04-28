"""Prompt template rendering utilities for the evaluation runner.

Templates are stored as Jinja2 ``.j2`` files in this directory.
Use :func:`render_template_file` for built-in templates and
:func:`render_template_string` for caller-supplied template strings.
"""

from pathlib import Path
from typing import Any

_PROMPTS_DIR = Path(__file__).parent
_environment = None


def _get_environment():
    global _environment
    if _environment is None:
        try:
            from jinja2 import Environment, FileSystemLoader
        except ImportError as e:
            raise ImportError(
                "jinja2 is required for SimulatedScenario execution. "
                "Install it with: pip install 'bedrock-agentcore[simulation]'"
            ) from e
        _environment = Environment(
            loader=FileSystemLoader(_PROMPTS_DIR),
            autoescape=False,
            keep_trailing_newline=True,
        )
    return _environment


def render_template_file(name: str, **kwargs: Any) -> str:
    """Render a built-in ``.j2`` template file by name.

    Args:
        name: Filename relative to the prompts directory (e.g. ``"structured_user_simulator.j2"``).
        **kwargs: Variables substituted into the template.

    Returns:
        The rendered template string.
    """
    return _get_environment().get_template(name).render(**kwargs)


def render_template_string(template_str: str, **kwargs: Any) -> str:
    """Render a caller-supplied Jinja2 template string.

    Args:
        template_str: A Jinja2 template string (use ``{{ variable }}`` syntax).
        **kwargs: Variables substituted into the template.

    Returns:
        The rendered template string.
    """
    return _get_environment().from_string(template_str).render(**kwargs)
