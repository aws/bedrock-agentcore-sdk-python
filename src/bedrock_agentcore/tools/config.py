"""Configuration helpers for Bedrock AgentCore Tools.

This module provides dataclasses and helper functions to simplify working with
browser and code interpreter configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VpcConfig:
    """VPC configuration for browsers and code interpreters.

    Attributes:
        security_groups: List of security group IDs
        subnets: List of subnet IDs
    """

    security_groups: List[str]
    subnets: List[str]

    def to_dict(self) -> Dict:
        """Convert to API-compatible dictionary."""
        return {"securityGroups": self.security_groups, "subnets": self.subnets}


@dataclass
class NetworkConfiguration:
    """Network configuration for browsers and code interpreters.

    Attributes:
        network_mode: Either "PUBLIC" or "VPC"
        vpc_config: VPC configuration (required if network_mode is VPC)
    """

    network_mode: str = "PUBLIC"
    vpc_config: Optional[VpcConfig] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.network_mode not in ["PUBLIC", "VPC"]:
            raise ValueError(f"network_mode must be 'PUBLIC' or 'VPC', got '{self.network_mode}'")

        if self.network_mode == "VPC" and not self.vpc_config:
            raise ValueError("vpc_config is required when network_mode is 'VPC'")

    def to_dict(self) -> Dict:
        """Convert to API-compatible dictionary."""
        config = {"networkMode": self.network_mode}
        if self.vpc_config:
            config["vpcConfig"] = self.vpc_config.to_dict()
        return config

    @classmethod
    def public(cls) -> "NetworkConfiguration":
        """Create a PUBLIC network configuration."""
        return cls(network_mode="PUBLIC")

    @classmethod
    def vpc(cls, security_groups: List[str], subnets: List[str]) -> "NetworkConfiguration":
        """Create a VPC network configuration.

        Args:
            security_groups: List of security group IDs
            subnets: List of subnet IDs

        Returns:
            NetworkConfiguration with VPC settings
        """
        return cls(network_mode="VPC", vpc_config=VpcConfig(security_groups, subnets))


@dataclass
class S3Location:
    """S3 location for recording storage.

    Attributes:
        bucket: S3 bucket name
        key_prefix: Optional S3 key prefix
    """

    bucket: str
    key_prefix: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to API-compatible dictionary."""
        location = {"bucket": self.bucket}
        if self.key_prefix:
            location["keyPrefix"] = self.key_prefix
        return location


@dataclass
class RecordingConfiguration:
    """Recording configuration for browsers.

    Attributes:
        enabled: Whether recording is enabled
        s3_location: S3 location for storing recordings
    """

    enabled: bool = True
    s3_location: Optional[S3Location] = None

    def to_dict(self) -> Dict:
        """Convert to API-compatible dictionary."""
        config = {"enabled": self.enabled}
        if self.s3_location:
            config["s3Location"] = self.s3_location.to_dict()
        return config

    @classmethod
    def disabled(cls) -> "RecordingConfiguration":
        """Create a disabled recording configuration."""
        return cls(enabled=False)

    @classmethod
    def enabled_with_location(cls, bucket: str, key_prefix: Optional[str] = None) -> "RecordingConfiguration":
        """Create an enabled recording configuration with S3 location.

        Args:
            bucket: S3 bucket name
            key_prefix: Optional S3 key prefix

        Returns:
            RecordingConfiguration with S3 location
        """
        return cls(enabled=True, s3_location=S3Location(bucket, key_prefix))


@dataclass
class BrowserSigningConfiguration:
    """Web Bot Auth (Browser Signing) configuration.

    This enables cryptographic identity for browsers to reduce CAPTCHA friction.

    Attributes:
        enabled: Whether browser signing (Web Bot Auth) is enabled
    """

    enabled: bool = True

    def to_dict(self) -> Dict:
        """Convert to API-compatible dictionary."""
        return {"enabled": self.enabled}

    @classmethod
    def enabled_config(cls) -> "BrowserSigningConfiguration":
        """Create an enabled browser signing configuration."""
        return cls(enabled=True)

    @classmethod
    def disabled_config(cls) -> "BrowserSigningConfiguration":
        """Create a disabled browser signing configuration."""
        return cls(enabled=False)


@dataclass
class ViewportConfiguration:
    """Browser viewport configuration.

    Attributes:
        width: Viewport width in pixels
        height: Viewport height in pixels
    """

    width: int
    height: int

    def to_dict(self) -> Dict:
        """Convert to API-compatible dictionary."""
        return {"width": self.width, "height": self.height}

    @classmethod
    def desktop_hd(cls) -> "ViewportConfiguration":
        """Standard HD desktop viewport (1920x1080)."""
        return cls(width=1920, height=1080)

    @classmethod
    def desktop_4k(cls) -> "ViewportConfiguration":
        """4K desktop viewport (3840x2160)."""
        return cls(width=3840, height=2160)

    @classmethod
    def laptop(cls) -> "ViewportConfiguration":
        """Standard laptop viewport (1366x768)."""
        return cls(width=1366, height=768)

    @classmethod
    def tablet(cls) -> "ViewportConfiguration":
        """Tablet viewport (768x1024)."""
        return cls(width=768, height=1024)

    @classmethod
    def mobile(cls) -> "ViewportConfiguration":
        """Mobile viewport (375x667)."""
        return cls(width=375, height=667)


@dataclass
class BrowserConfiguration:
    """Complete browser configuration for create_browser.

    This is a convenience class that bundles all browser creation parameters.

    Attributes:
        name: Browser name
        execution_role_arn: IAM role ARN
        network_configuration: Network settings
        description: Optional description
        recording: Optional recording configuration
        browser_signing: Optional Web Bot Auth configuration
        tags: Optional tags
    """

    name: str
    execution_role_arn: str
    network_configuration: NetworkConfiguration
    description: Optional[str] = None
    recording: Optional[RecordingConfiguration] = None
    browser_signing: Optional[BrowserSigningConfiguration] = None
    tags: Optional[Dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to API-compatible dictionary for create_browser."""
        config = {
            "name": self.name,
            "executionRoleArn": self.execution_role_arn,
            "networkConfiguration": self.network_configuration.to_dict(),
        }

        if self.description:
            config["description"] = self.description

        if self.recording:
            config["recording"] = self.recording.to_dict()

        if self.browser_signing:
            config["browserSigning"] = self.browser_signing.to_dict()

        if self.tags:
            config["tags"] = self.tags

        return config


@dataclass
class CodeInterpreterConfiguration:
    """Complete code interpreter configuration for create_code_interpreter.

    Attributes:
        name: Code interpreter name
        execution_role_arn: IAM role ARN
        network_configuration: Network settings
        description: Optional description
        tags: Optional tags
    """

    name: str
    execution_role_arn: str
    network_configuration: NetworkConfiguration
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to API-compatible dictionary for create_code_interpreter."""
        config = {
            "name": self.name,
            "executionRoleArn": self.execution_role_arn,
            "networkConfiguration": self.network_configuration.to_dict(),
        }

        if self.description:
            config["description"] = self.description

        if self.tags:
            config["tags"] = self.tags

        return config


def create_browser_config(
    name: str,
    execution_role_arn: str,
    enable_web_bot_auth: bool = False,
    enable_recording: bool = False,
    recording_bucket: Optional[str] = None,
    recording_prefix: Optional[str] = None,
    use_vpc: bool = False,
    security_groups: Optional[List[str]] = None,
    subnets: Optional[List[str]] = None,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> BrowserConfiguration:
    """Create a browser configuration with common options.

    Args:
        name: Browser name
        execution_role_arn: IAM role ARN
        enable_web_bot_auth: Enable Web Bot Auth for CAPTCHA reduction
        enable_recording: Enable session recording
        recording_bucket: S3 bucket for recordings (required if enable_recording=True)
        recording_prefix: S3 key prefix for recordings
        use_vpc: Use VPC network configuration
        security_groups: Security group IDs (required if use_vpc=True)
        subnets: Subnet IDs (required if use_vpc=True)
        description: Browser description
        tags: Resource tags

    Returns:
        BrowserConfiguration ready for create_browser

    Example:
        >>> # Create browser with Web Bot Auth and recording
        >>> config = create_browser_config(
        ...     name="my_signed_browser",
        ...     execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
        ...     enable_web_bot_auth=True,
        ...     enable_recording=True,
        ...     recording_bucket="my-recordings-bucket",
        ...     recording_prefix="competitive-intel/"
        ... )
        >>> browser = client.create_browser(**config.to_dict())
    """
    # Network configuration
    if use_vpc:
        if not security_groups or not subnets:
            raise ValueError("security_groups and subnets are required when use_vpc=True")
        network_config = NetworkConfiguration.vpc(security_groups, subnets)
    else:
        network_config = NetworkConfiguration.public()

    # Recording configuration
    recording_config = None
    if enable_recording:
        if not recording_bucket:
            raise ValueError("recording_bucket is required when enable_recording=True")
        recording_config = RecordingConfiguration.enabled_with_location(recording_bucket, recording_prefix)

    # Browser signing configuration
    signing_config = None
    if enable_web_bot_auth:
        signing_config = BrowserSigningConfiguration.enabled_config()

    return BrowserConfiguration(
        name=name,
        execution_role_arn=execution_role_arn,
        network_configuration=network_config,
        description=description,
        recording=recording_config,
        browser_signing=signing_config,
        tags=tags or {},
    )
