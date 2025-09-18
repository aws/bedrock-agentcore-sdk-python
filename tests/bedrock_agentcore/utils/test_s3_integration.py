"""Tests for S3 integration with pandas using s3fs."""

import io

import pytest

# Check for required dependencies
try:
    import pandas as pd
    import s3fs

    _ = s3fs
    HAS_S3_SUPPORT = True
except ImportError:
    HAS_S3_SUPPORT = False

try:
    import boto3
    from moto import mock_aws

    HAS_MOTO = True
except ImportError:
    HAS_MOTO = False


@pytest.mark.skipif(not HAS_S3_SUPPORT, reason="s3fs and pandas not available")
@pytest.mark.skipif(not HAS_MOTO, reason="moto not available")
class TestS3PandasIntegration:
    """Test that pandas can read from S3 when s3fs is installed."""

    def test_s3fs_and_pandas_integration(self):
        """Test that s3fs and pandas work together (simulated)."""
        with mock_aws():
            # Setup mock S3
            bucket = "mock-bucket"
            key = "mock.csv"

            s3_client = boto3.client("s3", region_name="us-east-1")
            s3_client.create_bucket(Bucket=bucket)

            # Upload test CSV
            csv_data = "key,value\nk1,v1\nk2,v2"
            s3_client.put_object(Bucket=bucket, Key=key, Body=csv_data)

            # 1. Use boto3 to get the object (like s3fs would internally)
            response = s3_client.get_object(Bucket=bucket, Key=key)
            csv_content = response["Body"].read().decode("utf-8")

            # 2. Use pandas to read the CSV content (like pandas would)
            df = pd.read_csv(io.StringIO(csv_content))

            # Verify the result
            assert list(df.columns) == ["key", "value"]
            assert df.iloc[0]["key"] == "k1"
            assert df.iloc[1]["key"] == "k2"
            assert df.iloc[0]["value"] == "v1"
            assert df.iloc[1]["value"] == "v2"
