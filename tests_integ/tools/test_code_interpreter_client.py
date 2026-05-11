"""Integration tests for CodeInterpreter client.

Run with:
    uv run pytest tests_integ/tools/test_code_interpreter_client.py -xvs
"""

import os

import pytest

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

# 67 bytes of binary data; base64 encoding would produce 92 bytes
PAYLOAD = b"\x89PNG\r\n\x1a\n" + bytes(range(59))
EXPECTED_SIZE = len(PAYLOAD)  # 67


def _extract_stdout(stream):
    """Extract stdout content from an execute_code stream response.

    Returns the stdout string or raises AssertionError if not found.
    """
    for event in stream:
        r = event.get("result", {})
        stdout = r.get("structuredContent", {}).get("stdout", "")
        if stdout:
            return stdout
        content = r.get("content", "")
        if content:
            return str(content)
    raise AssertionError("stdout not found in stream response")


@pytest.mark.integration
class TestCodeInterpreterClient:
    """Integration tests for CodeInterpreter client."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-east-1")
        cls.client = CodeInterpreter(cls.region)
        cls.client.start()
        cls.client.upload_file(path="test.bin", content=PAYLOAD)

    @classmethod
    def teardown_class(cls):
        cls.client.stop()

    def test_upload_file_writes_correct_size(self):
        """upload_file with binary bytes writes the correct size to disk (not double-base64 encoded)."""
        result = self.client.execute_code("import os\nprint(os.path.getsize('test.bin'))")

        stdout = _extract_stdout(result["stream"])
        disk_size = None
        for line in stdout.splitlines():
            if line.strip().isdigit():
                disk_size = int(line.strip())

        assert disk_size is not None, "Could not parse disk size from stdout"
        assert disk_size == EXPECTED_SIZE, (
            f"Expected {EXPECTED_SIZE} bytes on disk, got {disk_size} (92 would indicate double-base64 encoding)"
        )

    def test_download_file_returns_original_bytes(self):
        """download_file returns the exact original bytes that were uploaded."""
        downloaded = self.client.download_file("test.bin")

        assert isinstance(downloaded, bytes), f"Expected bytes, got {type(downloaded).__name__}"
        assert downloaded == PAYLOAD, (
            f"Downloaded content does not match original payload. "
            f"Got {len(downloaded)} bytes, expected {EXPECTED_SIZE} bytes."
        )
