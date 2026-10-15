"""Prompt template rendering utilities for the evaluation runner.

Templates are stored as Jinja2 ``.j2`` files in this directory.
Use :func:`render_template_file` for built-in templates and
:func:`render_template_string` for caller-supplied template strings.
"""

import re
from pathlib import Path
from typing import Any

_PROMPTS_DIR = Path(__file__).parent
_environment = None
_string_environment = None

_JINJA2_IMPORT_ERROR = (
    "jinja2 is required for SimulatedScenario execution. Install it with: pip install 'bedrock-agentcore[simulation]'"
)
_MIN_STRING_TEMPLATE_JINJA2_VERSION = (3, 1, 6)


def _get_environment():
    """Return the environment for built-in, trusted ``.j2`` files shipped with the SDK."""
    global _environment
    if _environment is None:
        try:
            from jinja2 import Environment, FileSystemLoader
        except ImportError as e:
            raise ImportError(_JINJA2_IMPORT_ERROR) from e
        _environment = Environment(  # nosec B701 - templates render plain text/JSON, not HTML
            loader=FileSystemLoader(_PROMPTS_DIR),
            autoescape=False,
            keep_trailing_newline=True,
        )
    return _environment


def _get_string_environment():
    """Return an isolated environment for caller-supplied template strings.

    ``ImmutableSandboxedEnvironment`` keeps input values unchanged, while omitting a
    loader keeps template strings independent from the bundled template files.
    """
    global _string_environment
    if _string_environment is None:
        try:
            import jinja2
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError as e:
            raise ImportError(_JINJA2_IMPORT_ERROR) from e
        version = tuple(int(part) for part in re.findall(r"\d+", jinja2.__version__)[:3])
        if version < _MIN_STRING_TEMPLATE_JINJA2_VERSION:
            raise ImportError(
                f"jinja2>=3.1.6 is required to render system_prompt_template "
                f"(found {jinja2.__version__}). Install it with: pip install 'bedrock-agentcore[simulation]'"
            )
        _string_environment = ImmutableSandboxedEnvironment(  # nosec B701 - templates render plain text/JSON
            autoescape=False,
            keep_trailing_newline=True,
        )
    return _string_environment


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
    """Render a caller-supplied Jinja2 template string in a sandbox.

    Args:
        template_str: A Jinja2 template string (use ``{{ variable }}`` syntax).
        **kwargs: Variables substituted into the template.

    Returns:
        The rendered template string.

    Raises:
        jinja2.exceptions.SecurityError: If the template attempts an unsafe operation.
    """
    return _get_string_environment().from_string(template_str).render(**kwargs)
