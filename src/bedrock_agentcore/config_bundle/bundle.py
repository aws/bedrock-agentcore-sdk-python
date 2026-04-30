"""Configuration bundle reference model."""

from dataclasses import dataclass
from typing import Any, Dict

# ComponentConfigurationMap value: {componentId: {"configuration": <Document>}}
ConfigBundleComponents = Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class ConfigBundleRef:
    """Lightweight reference to a configuration bundle version, parsed from OTEL baggage.

    .. warning::
        This feature is in preview and may change in future releases.
    """

    bundle_arn: str
    bundle_version: str

    def __post_init__(self) -> None:
        """Validate bundle ARN and version."""
        if not self.bundle_arn:
            raise ValueError("bundle_arn must not be empty")
        if not self.bundle_version:
            raise ValueError("bundle_version must not be empty")
        parts = self.bundle_arn.rsplit("/", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError(f"bundle_arn does not contain a valid bundle ID segment: {self.bundle_arn!r}")

    @property
    def bundle_id(self) -> str:
        """Extract bundle ID from ARN (last path segment after '/')."""
        return self.bundle_arn.rsplit("/", 1)[-1]
