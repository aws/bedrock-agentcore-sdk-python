"""Namespace utilities for data plane API calls."""

from typing import Dict, Optional


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
