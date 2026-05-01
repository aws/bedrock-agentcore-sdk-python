"""Namespace utilities for data plane and control plane API calls."""

import warnings
from typing import Dict, List, Optional


def build_namespace_params(namespace: Optional[str] = None, namespace_path: Optional[str] = None) -> Dict[str, str]:
    """Build the namespace kwargs for a data plane API call.

    Exactly one of ``namespace`` (exact match) or ``namespace_path``
    (hierarchical path prefix) must be provided. Wildcards (``*``) are not
    supported in either field.

    Raises:
        ValueError: if both arguments are provided, neither is provided, or
            the provided value contains a wildcard.
    """
    if namespace is not None and namespace_path is not None:
        raise ValueError("'namespace' and 'namespace_path' are mutually exclusive.")
    if namespace is None and namespace_path is None:
        raise ValueError("At least one of 'namespace' or 'namespace_path' must be provided.")

    value = namespace if namespace is not None else namespace_path
    if "*" in value:
        raise ValueError("Wildcards (*) are not supported in namespaces.")

    if namespace is not None:
        return {"namespace": namespace}
    return {"namespacePath": namespace_path}


def resolve_namespace_templates(
    namespaces: Optional[List[str]] = None,
    namespace_templates: Optional[List[str]] = None,
    param_name: str = "namespaces",
    new_param_name: Optional[str] = None,
) -> Optional[List[str]]:
    """Resolve the deprecated ``namespaces`` kwarg and the new ``namespace_templates`` kwarg.

    Used by control-plane strategy methods. Exactly one (or neither) may be provided.
    If the deprecated form is used, a ``DeprecationWarning`` is emitted. Returns the
    resolved list, or ``None`` if neither was provided.

    Args:
        namespaces: The deprecated parameter value, if any.
        namespace_templates: The new parameter value, if any.
        param_name: Base name used in error/warning messages for the deprecated form
            (e.g. "namespaces" or "reflection_namespaces").
        new_param_name: Name of the replacement form to reference in messages. Defaults
            to ``param_name`` with ``"namespaces"`` replaced by ``"namespace_templates"``.
            Override when the replacement identifier doesn't follow that pattern (for
            example, a dict key like ``reflection_config['namespaceTemplates']``).

    Raises:
        ValueError: if both arguments are provided.
    """
    if new_param_name is None:
        new_param_name = param_name.replace("namespaces", "namespace_templates")

    if namespaces is not None and namespace_templates is not None:
        raise ValueError(
            f"'{param_name}' and '{new_param_name}' are mutually exclusive. "
            f"Prefer '{new_param_name}' ('{param_name}' is deprecated)."
        )

    if namespaces is not None:
        warnings.warn(
            f"The '{param_name}' parameter is deprecated and will be removed in a future release. "
            f"Use '{new_param_name}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return namespaces

    return namespace_templates
