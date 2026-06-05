"""Tests for ShellFramer and related types."""

import json

import pytest

from bedrock_agentcore.runtime.shell import ShellChannel, ShellFrame, ShellFramer


class TestShellChannel:
    def test_values(self):
        assert ShellChannel.STDIN == 0x00
        assert ShellChannel.STDOUT == 0x01
        assert ShellChannel.STDERR == 0x02
        assert ShellChannel.STATUS == 0x03
        assert ShellChannel.RESIZE == 0x04
        assert ShellChannel.HEARTBEAT == 0x05
        assert ShellChannel.CLOSE == 0xFF


class TestShellFrame:
    def test_text_property(self):
        frame = ShellFrame(channel=ShellChannel.STDOUT, raw_channel_byte=0x01, payload=b"hello")
        assert frame.text == "hello"

    def test_text_property_replaces_invalid_bytes(self):
        frame = ShellFrame(channel=ShellChannel.STDOUT, raw_channel_byte=0x01, payload=b"\xff\xfe")
        assert "�" in frame.text

    def test_json_property(self):
        payload = json.dumps({"exitCode": 0}).encode()
        frame = ShellFrame(channel=ShellChannel.STATUS, raw_channel_byte=0x03, payload=payload)
        assert frame.json() == {"exitCode": 0}

    def test_json_raises_on_invalid_payload(self):
        frame = ShellFrame(channel=ShellChannel.STATUS, raw_channel_byte=0x03, payload=b"not json")
        with pytest.raises(json.JSONDecodeError):
            frame.json()


class TestShellFramerDecode:
    def setup_method(self):
        self.framer = ShellFramer()

    def test_decode_stdout(self):
        raw = bytes([ShellChannel.STDOUT]) + b"hello"
        frame = self.framer.decode(raw)
        assert frame.channel == ShellChannel.STDOUT
        assert frame.payload == b"hello"

    def test_decode_stdin(self):
        raw = bytes([ShellChannel.STDIN]) + b"ls\n"
        frame = self.framer.decode(raw)
        assert frame.channel == ShellChannel.STDIN
        assert frame.payload == b"ls\n"

    def test_decode_status_exit(self):
        payload = json.dumps(
            {
                "kind": "Status",
                "apiVersion": "v1",
                "metadata": {},
                "status": "Failure",
                "reason": "NonZeroExitCode",
                "details": {"causes": [{"reason": "ExitCode", "message": "1"}]},
            }
        ).encode()
        raw = bytes([ShellChannel.STATUS]) + payload
        frame = self.framer.decode(raw)
        assert frame.channel == ShellChannel.STATUS
        causes = frame.json()["details"]["causes"]
        assert causes[0]["message"] == "1"

    def test_decode_status_confirmation(self):
        payload = json.dumps(
            {
                "kind": "Status",
                "apiVersion": "v1",
                "metadata": {"shellId": "my-shell", "reconnected": False},
                "status": "Success",
            }
        ).encode()
        raw = bytes([ShellChannel.STATUS]) + payload
        frame = self.framer.decode(raw)
        assert frame.channel == ShellChannel.STATUS
        assert frame.json()["metadata"]["shellId"] == "my-shell"

    def test_decode_resize(self):
        payload = json.dumps({"width": 80, "height": 24}).encode()
        raw = bytes([ShellChannel.RESIZE]) + payload
        frame = self.framer.decode(raw)
        assert frame.channel == ShellChannel.RESIZE

    def test_decode_close_empty_payload(self):
        raw = bytes([ShellChannel.CLOSE])
        frame = self.framer.decode(raw)
        assert frame.channel == ShellChannel.CLOSE
        assert frame.payload == b""

    def test_decode_unknown_channel_preserved(self):
        raw = bytes([0x42]) + b"future data"
        frame = self.framer.decode(raw)
        assert frame.channel == ShellChannel.UNKNOWN
        assert frame.raw_channel_byte == 0x42
        assert frame.payload == b"future data"

    def test_decode_known_channel_raw_byte_matches(self):
        raw = bytes([ShellChannel.STDOUT]) + b"hello"
        frame = self.framer.decode(raw)
        assert frame.channel == ShellChannel.STDOUT
        assert frame.raw_channel_byte == ShellChannel.STDOUT

    def test_decode_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.framer.decode(b"")


class TestShellFramerEncode:
    def setup_method(self):
        self.framer = ShellFramer()

    def test_encode_stdin_str(self):
        frame = self.framer.encode_stdin("ls\n")
        assert frame[0] == ShellChannel.STDIN
        assert frame[1:] == b"ls\n"

    def test_encode_stdin_bytes(self):
        frame = self.framer.encode_stdin(b"\x1b[A")
        assert frame[0] == ShellChannel.STDIN
        assert frame[1:] == b"\x1b[A"

    def test_encode_stdin_exceeds_limit_raises(self):
        big = "x" * (ShellFramer.MAX_FRAME_SIZE)
        with pytest.raises(ValueError, match="64 KB"):
            self.framer.encode_stdin(big)

    def test_encode_stdin_at_limit_ok(self):
        data = b"x" * (ShellFramer.MAX_FRAME_SIZE - 1)
        frame = self.framer.encode_stdin(data)
        assert len(frame) == ShellFramer.MAX_FRAME_SIZE

    def test_encode_resize(self):
        frame = self.framer.encode_resize(220, 50)
        assert frame[0] == ShellChannel.RESIZE
        payload = json.loads(frame[1:])
        assert payload == {"width": 220, "height": 50}

    @pytest.mark.parametrize("width,height", [(0, 24), (80, 0), (-1, 24), (80, -1), (0, 0)])
    def test_encode_resize_invalid_dimensions_raises(self, width, height):
        with pytest.raises(ValueError, match="positive integers"):
            self.framer.encode_resize(width, height)

    def test_encode_heartbeat(self):
        frame = self.framer.encode_heartbeat()
        assert frame == bytes([ShellChannel.HEARTBEAT])

    def test_encode_close(self):
        frame = self.framer.encode_close()
        assert frame == bytes([ShellChannel.CLOSE])

    def test_round_trip_stdin(self):
        original = "echo hello\n"
        encoded = self.framer.encode_stdin(original)
        decoded = self.framer.decode(encoded)
        assert decoded.channel == ShellChannel.STDIN
        assert decoded.text == original
