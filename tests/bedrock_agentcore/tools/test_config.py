import pytest

from bedrock_agentcore.tools.config import (
    BrowserConfiguration,
    BrowserSigningConfiguration,
    CodeInterpreterConfiguration,
    NetworkConfiguration,
    RecordingConfiguration,
    S3Location,
    ViewportConfiguration,
    VpcConfig,
    create_browser_config,
)


class TestVpcConfig:
    def test_vpc_config_creation(self):
        # Arrange & Act
        vpc_config = VpcConfig(security_groups=["sg-123", "sg-456"], subnets=["subnet-abc", "subnet-def"])

        # Assert
        assert vpc_config.security_groups == ["sg-123", "sg-456"]
        assert vpc_config.subnets == ["subnet-abc", "subnet-def"]

    def test_vpc_config_to_dict(self):
        # Arrange
        vpc_config = VpcConfig(security_groups=["sg-123"], subnets=["subnet-abc"])

        # Act
        result = vpc_config.to_dict()

        # Assert
        assert result == {"securityGroups": ["sg-123"], "subnets": ["subnet-abc"]}


class TestNetworkConfiguration:
    def test_public_network_config(self):
        # Arrange & Act
        network_config = NetworkConfiguration.public()

        # Assert
        assert network_config.network_mode == "PUBLIC"
        assert network_config.vpc_config is None

    def test_public_network_config_to_dict(self):
        # Arrange
        network_config = NetworkConfiguration.public()

        # Act
        result = network_config.to_dict()

        # Assert
        assert result == {"networkMode": "PUBLIC"}

    def test_vpc_network_config(self):
        # Arrange & Act
        network_config = NetworkConfiguration.vpc(security_groups=["sg-123"], subnets=["subnet-abc"])

        # Assert
        assert network_config.network_mode == "VPC"
        assert network_config.vpc_config is not None
        assert network_config.vpc_config.security_groups == ["sg-123"]
        assert network_config.vpc_config.subnets == ["subnet-abc"]

    def test_vpc_network_config_to_dict(self):
        # Arrange
        network_config = NetworkConfiguration.vpc(security_groups=["sg-123"], subnets=["subnet-abc"])

        # Act
        result = network_config.to_dict()

        # Assert
        assert result == {
            "networkMode": "VPC",
            "vpcConfig": {"securityGroups": ["sg-123"], "subnets": ["subnet-abc"]},
        }

    def test_invalid_network_mode(self):
        # Act & Assert
        with pytest.raises(ValueError, match="network_mode must be 'PUBLIC' or 'VPC'"):
            NetworkConfiguration(network_mode="INVALID")

    def test_vpc_mode_without_vpc_config(self):
        # Act & Assert
        with pytest.raises(ValueError, match="vpc_config is required when network_mode is 'VPC'"):
            NetworkConfiguration(network_mode="VPC")


class TestS3Location:
    def test_s3_location_with_prefix(self):
        # Arrange & Act
        s3_location = S3Location(bucket="my-bucket", key_prefix="recordings/")

        # Assert
        assert s3_location.bucket == "my-bucket"
        assert s3_location.key_prefix == "recordings/"

    def test_s3_location_without_prefix(self):
        # Arrange & Act
        s3_location = S3Location(bucket="my-bucket")

        # Assert
        assert s3_location.bucket == "my-bucket"
        assert s3_location.key_prefix is None

    def test_s3_location_to_dict_with_prefix(self):
        # Arrange
        s3_location = S3Location(bucket="my-bucket", key_prefix="recordings/")

        # Act
        result = s3_location.to_dict()

        # Assert
        assert result == {"bucket": "my-bucket", "keyPrefix": "recordings/"}

    def test_s3_location_to_dict_without_prefix(self):
        # Arrange
        s3_location = S3Location(bucket="my-bucket")

        # Act
        result = s3_location.to_dict()

        # Assert
        assert result == {"bucket": "my-bucket"}


class TestRecordingConfiguration:
    def test_recording_disabled(self):
        # Arrange & Act
        recording_config = RecordingConfiguration.disabled()

        # Assert
        assert recording_config.enabled is False
        assert recording_config.s3_location is None

    def test_recording_disabled_to_dict(self):
        # Arrange
        recording_config = RecordingConfiguration.disabled()

        # Act
        result = recording_config.to_dict()

        # Assert
        assert result == {"enabled": False}

    def test_recording_enabled_with_location(self):
        # Arrange & Act
        recording_config = RecordingConfiguration.enabled_with_location(bucket="my-bucket", key_prefix="recordings/")

        # Assert
        assert recording_config.enabled is True
        assert recording_config.s3_location is not None
        assert recording_config.s3_location.bucket == "my-bucket"
        assert recording_config.s3_location.key_prefix == "recordings/"

    def test_recording_enabled_with_location_to_dict(self):
        # Arrange
        recording_config = RecordingConfiguration.enabled_with_location(bucket="my-bucket", key_prefix="recordings/")

        # Act
        result = recording_config.to_dict()

        # Assert
        assert result == {
            "enabled": True,
            "s3Location": {"bucket": "my-bucket", "keyPrefix": "recordings/"},
        }

    def test_recording_enabled_without_prefix(self):
        # Arrange & Act
        recording_config = RecordingConfiguration.enabled_with_location(bucket="my-bucket")

        # Act
        result = recording_config.to_dict()

        # Assert
        assert result == {
            "enabled": True,
            "s3Location": {"bucket": "my-bucket"},
        }


class TestBrowserSigningConfiguration:
    def test_browser_signing_enabled(self):
        # Arrange & Act
        signing_config = BrowserSigningConfiguration.enabled_config()

        # Assert
        assert signing_config.enabled is True

    def test_browser_signing_disabled(self):
        # Arrange & Act
        signing_config = BrowserSigningConfiguration.disabled_config()

        # Assert
        assert signing_config.enabled is False

    def test_browser_signing_to_dict_enabled(self):
        # Arrange
        signing_config = BrowserSigningConfiguration.enabled_config()

        # Act
        result = signing_config.to_dict()

        # Assert
        assert result == {"enabled": True}

    def test_browser_signing_to_dict_disabled(self):
        # Arrange
        signing_config = BrowserSigningConfiguration.disabled_config()

        # Act
        result = signing_config.to_dict()

        # Assert
        assert result == {"enabled": False}


class TestViewportConfiguration:
    def test_custom_viewport(self):
        # Arrange & Act
        viewport = ViewportConfiguration(width=1920, height=1080)

        # Assert
        assert viewport.width == 1920
        assert viewport.height == 1080

    def test_viewport_to_dict(self):
        # Arrange
        viewport = ViewportConfiguration(width=1920, height=1080)

        # Act
        result = viewport.to_dict()

        # Assert
        assert result == {"width": 1920, "height": 1080}

    def test_desktop_hd_preset(self):
        # Arrange & Act
        viewport = ViewportConfiguration.desktop_hd()

        # Assert
        assert viewport.width == 1920
        assert viewport.height == 1080

    def test_desktop_4k_preset(self):
        # Arrange & Act
        viewport = ViewportConfiguration.desktop_4k()

        # Assert
        assert viewport.width == 3840
        assert viewport.height == 2160

    def test_laptop_preset(self):
        # Arrange & Act
        viewport = ViewportConfiguration.laptop()

        # Assert
        assert viewport.width == 1366
        assert viewport.height == 768

    def test_tablet_preset(self):
        # Arrange & Act
        viewport = ViewportConfiguration.tablet()

        # Assert
        assert viewport.width == 768
        assert viewport.height == 1024

    def test_mobile_preset(self):
        # Arrange & Act
        viewport = ViewportConfiguration.mobile()

        # Assert
        assert viewport.width == 375
        assert viewport.height == 667


class TestBrowserConfiguration:
    def test_minimal_browser_config(self):
        # Arrange & Act
        config = BrowserConfiguration(
            name="test_browser",
            execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
            network_configuration=NetworkConfiguration.public(),
        )

        # Assert
        assert config.name == "test_browser"
        assert config.execution_role_arn == "arn:aws:iam::123456789012:role/BrowserRole"
        assert config.description is None
        assert config.recording is None
        assert config.browser_signing is None
        assert config.tags == {}

    def test_full_browser_config_to_dict(self):
        # Arrange
        config = BrowserConfiguration(
            name="test_browser",
            execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
            network_configuration=NetworkConfiguration.public(),
            description="Test browser",
            recording=RecordingConfiguration.enabled_with_location("my-bucket", "recordings/"),
            browser_signing=BrowserSigningConfiguration.enabled_config(),
            tags={"Environment": "Test"},
        )

        # Act
        result = config.to_dict()

        # Assert
        assert result == {
            "name": "test_browser",
            "executionRoleArn": "arn:aws:iam::123456789012:role/BrowserRole",
            "networkConfiguration": {"networkMode": "PUBLIC"},
            "description": "Test browser",
            "recording": {"enabled": True, "s3Location": {"bucket": "my-bucket", "keyPrefix": "recordings/"}},
            "browserSigning": {"enabled": True},
            "tags": {"Environment": "Test"},
        }


class TestCodeInterpreterConfiguration:
    def test_minimal_interpreter_config(self):
        # Arrange & Act
        config = CodeInterpreterConfiguration(
            name="test_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/InterpreterRole",
            network_configuration=NetworkConfiguration.public(),
        )

        # Assert
        assert config.name == "test_interpreter"
        assert config.execution_role_arn == "arn:aws:iam::123456789012:role/InterpreterRole"
        assert config.description is None
        assert config.tags == {}

    def test_full_interpreter_config_to_dict(self):
        # Arrange
        config = CodeInterpreterConfiguration(
            name="test_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/InterpreterRole",
            network_configuration=NetworkConfiguration.vpc(["sg-123"], ["subnet-abc"]),
            description="Test interpreter",
            tags={"Environment": "Test"},
        )

        # Act
        result = config.to_dict()

        # Assert
        assert result == {
            "name": "test_interpreter",
            "executionRoleArn": "arn:aws:iam::123456789012:role/InterpreterRole",
            "networkConfiguration": {
                "networkMode": "VPC",
                "vpcConfig": {"securityGroups": ["sg-123"], "subnets": ["subnet-abc"]},
            },
            "description": "Test interpreter",
            "tags": {"Environment": "Test"},
        }


class TestCreateBrowserConfig:
    def test_create_browser_config_minimal(self):
        # Arrange & Act
        config = create_browser_config(
            name="test_browser",
            execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
        )

        # Assert
        assert config.name == "test_browser"
        assert config.execution_role_arn == "arn:aws:iam::123456789012:role/BrowserRole"
        assert config.network_configuration.network_mode == "PUBLIC"
        assert config.recording is None
        assert config.browser_signing is None

    def test_create_browser_config_with_web_bot_auth(self):
        # Arrange & Act
        config = create_browser_config(
            name="test_browser",
            execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
            enable_web_bot_auth=True,
        )

        # Assert
        assert config.browser_signing is not None
        assert config.browser_signing.enabled is True

    def test_create_browser_config_with_recording(self):
        # Arrange & Act
        config = create_browser_config(
            name="test_browser",
            execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
            enable_recording=True,
            recording_bucket="my-bucket",
            recording_prefix="recordings/",
        )

        # Assert
        assert config.recording is not None
        assert config.recording.enabled is True
        assert config.recording.s3_location.bucket == "my-bucket"
        assert config.recording.s3_location.key_prefix == "recordings/"

    def test_create_browser_config_recording_without_bucket(self):
        # Act & Assert
        with pytest.raises(ValueError, match="recording_bucket is required when enable_recording=True"):
            create_browser_config(
                name="test_browser",
                execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
                enable_recording=True,
            )

    def test_create_browser_config_with_vpc(self):
        # Arrange & Act
        config = create_browser_config(
            name="test_browser",
            execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
            use_vpc=True,
            security_groups=["sg-123"],
            subnets=["subnet-abc"],
        )

        # Assert
        assert config.network_configuration.network_mode == "VPC"
        assert config.network_configuration.vpc_config.security_groups == ["sg-123"]
        assert config.network_configuration.vpc_config.subnets == ["subnet-abc"]

    def test_create_browser_config_vpc_without_security_groups(self):
        # Act & Assert
        with pytest.raises(ValueError, match="security_groups and subnets are required when use_vpc=True"):
            create_browser_config(
                name="test_browser",
                execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
                use_vpc=True,
            )

    def test_create_browser_config_full(self):
        # Arrange & Act
        config = create_browser_config(
            name="test_browser",
            execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
            enable_web_bot_auth=True,
            enable_recording=True,
            recording_bucket="my-bucket",
            recording_prefix="recordings/",
            use_vpc=True,
            security_groups=["sg-123"],
            subnets=["subnet-abc"],
            description="Full test browser",
            tags={"Environment": "Test"},
        )

        # Act
        result = config.to_dict()

        # Assert
        assert result["name"] == "test_browser"
        assert result["executionRoleArn"] == "arn:aws:iam::123456789012:role/BrowserRole"
        assert result["networkConfiguration"]["networkMode"] == "VPC"
        assert result["description"] == "Full test browser"
        assert result["recording"]["enabled"] is True
        assert result["browserSigning"]["enabled"] is True
        assert result["tags"] == {"Environment": "Test"}
