"""Binary channel-prefix framer for InvokeAgentRuntimeCommandShell.

Wire format (identical to Kubernetes v5.channel.k8s.io):
  [1-byte channel ID][payload bytes]

Channels:
  0x00  STDIN      Raw bytes, client → shell
  0x01  STDOUT     Raw bytes, shell → client
  0x02  STDERR     UTF-8 text (platform diagnostics), shell → client
  0x03  STATUS     metav1.Status JSON (shell → client):
                     Connection confirmation: metadata.shellId present
                     {"kind":"Status","apiVersion":"v1",
                      "metadata":{"shellId":"…","reconnected":bool},"status":"Success"}
                     Shell exit (code 0):
                     {"kind":"Status","apiVersion":"v1","metadata":{},"status":"Success"}
                     Shell exit (non-zero):
                     {"kind":"Status","apiVersion":"v1","metadata":{},"status":"Failure",
                      "reason":"NonZeroExitCode","message":"command terminated with exit code N",
                      "details":{"causes":[{"reason":"ExitCode","message":"N"}]}}
                     Shell killed by signal (e.g. SIGKILL/137):
                     {"kind":"Status","apiVersion":"v1","metadata":{},"status":"Failure",
                      "reason":"NonZeroExitCode","message":"command terminated with exit code N",
                      "details":{"causes":[{"reason":"ExitCode","message":"N"},
                                           {"reason":"Signal","message":"<signum>"}]}}
                     Transient platform error (e.g. init failure):
                     {"kind":"Status","apiVersion":"v1","metadata":{},"status":"Failure",
                      "reason":"InternalError","message":"…","code":500}
  0x04  RESIZE     JSON {"width":N,"height":N}, client → shell
  0x05  HEARTBEAT  Empty payload, bidirectional — browser app-level keepalive (echo back)
  0xFF  CLOSE      Empty payload, bidirectional — client → server to request graceful
                   shutdown; server → client for platform-initiated shutdown (VM eviction,
                   TTL expiry). Not sent on normal shell exit.
"""

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Union


class ShellChannel(IntEnum):
    """Wire channel identifiers for the binary channel-prefix protocol."""

    STDIN = 0x00
    STDOUT = 0x01
    STDERR = 0x02
    STATUS = 0x03
    RESIZE = 0x04
    HEARTBEAT = 0x05
    CLOSE = 0xFF
    UNKNOWN = -1  # sentinel for unrecognised channel bytes; not a wire value


@dataclass
class ShellFrame:
    """A single decoded WebSocket frame from the shell stream.

    Attributes:
        channel: The channel this frame belongs to.  For unrecognised channel
            bytes (future protocol extensions) this is ``ShellChannel.UNKNOWN``; use
            ``raw_channel_byte`` to retrieve the original wire value.
        raw_channel_byte: The original channel byte from the wire, always
            present regardless of whether the byte maps to a known ShellChannel.
        payload: Raw bytes of the frame payload (everything after the channel byte).
    """

    channel: ShellChannel
    raw_channel_byte: int
    payload: bytes

    @property
    def text(self) -> str:
        """Decode payload as UTF-8 text (replacement char on invalid bytes)."""
        return self.payload.decode("utf-8", errors="replace")

    def json(self) -> Dict[str, Any]:
        """Parse payload as JSON.

        Returns:
            Parsed JSON object.

        Raises:
            json.JSONDecodeError: If the payload is not valid JSON.
        """
        return json.loads(self.payload)


class ShellFramer:
    r"""Encodes and decodes binary channel-prefix WebSocket frames.

    Stateless — a single instance is safe to reuse across frames.

    Example:
        framer = ShellFramer()

        # Decode inbound frame
        frame = framer.decode(raw_bytes)
        if frame.channel == ShellChannel.STDOUT:
            sys.stdout.write(frame.text)
        elif frame.channel == ShellChannel.STATUS:
            status = frame.json()
            if status.get("metadata", {}).get("shellId"):
                print(f"connected: {status['metadata']['shellId']}")
            elif status.get("status") == "Failure":
                causes = status.get("details", {}).get("causes", [])
                code = next((c["message"] for c in causes if c["reason"] == "ExitCode"), None)
                print(f"shell exited with code {code}")

        # Encode outbound frames
        ws.send(framer.encode_stdin("ls /workspace\\n"))
        ws.send(framer.encode_resize(220, 50))
        ws.send(framer.encode_close())
    """

    MAX_FRAME_SIZE = 64 * 1024  # matches DP WebSocketFlowController limit

    def decode(self, frame: bytes) -> ShellFrame:
        """Decode one raw WebSocket binary message into a ShellFrame.

        Args:
            frame: Raw bytes received from the WebSocket.

        Returns:
            Decoded ShellFrame.

        Raises:
            ValueError: If the frame is empty.
        """
        if not frame:
            raise ValueError("Cannot decode empty frame")
        raw_byte = frame[0]
        try:
            channel = ShellChannel(raw_byte)
        except ValueError:
            # Unknown channel byte — future protocol extension.  Preserve the
            # payload so callers can inspect or forward it; use raw_channel_byte
            # to recover the original wire value.
            channel = ShellChannel.UNKNOWN
        return ShellFrame(channel=channel, raw_channel_byte=raw_byte, payload=frame[1:])

    def encode_stdin(self, data: Union[str, bytes]) -> bytes:
        """Encode keyboard input or paste data as a STDIN frame.

        Large pastes must be chunked into <64 KB segments before calling this
        method; the server closes the connection on oversized frames.

        Args:
            data: Text (encoded as UTF-8) or raw bytes to send to the shell.

        Returns:
            Binary WebSocket frame ready to send.

        Raises:
            ValueError: If the encoded payload would exceed the 64 KB frame limit.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        if len(data) > self.MAX_FRAME_SIZE - 1:
            raise ValueError(
                f"Payload {len(data)} bytes exceeds the 64 KB frame limit. "
                "Split large pastes into multiple encode_stdin() calls."
            )
        return bytes([ShellChannel.STDIN]) + data

    def encode_resize(self, width: int, height: int) -> bytes:
        """Encode a terminal resize event as a RESIZE frame.

        Send this whenever the terminal window changes size so the PTY
        reflows output correctly (e.g., on xterm.js onResize).

        Args:
            width: New terminal width in columns.
            height: New terminal height in rows.

        Returns:
            Binary WebSocket frame ready to send.

        Raises:
            ValueError: If width or height is not a positive integer.
        """
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError(
                f"width and height must be integers, got {type(width).__name__} and {type(height).__name__}"
            )
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive integers, got width={width}, height={height}")
        payload = json.dumps({"width": width, "height": height}).encode("utf-8")
        return bytes([ShellChannel.RESIZE]) + payload

    def encode_heartbeat(self) -> bytes:
        """Encode an app-level heartbeat frame (channel 0x05, empty payload).

        Browser clients cannot send RFC 6455 Ping frames, so the spec defines
        channel 0x05 as an application-level keepalive: the client sends this
        every 30 seconds, and the server echoes a single [0x05] back.  Both
        directions reset the KARP idle timer, preventing the ~15-minute proxy
        timeout on quiet connections.

        SDK/CLI clients should prefer RFC 6455 Ping frames (handled automatically
        by all standard WebSocket libraries).  Use this only when building a
        browser relay.

        Returns:
            Binary WebSocket frame ready to send.
        """
        return bytes([ShellChannel.HEARTBEAT])

    def encode_close(self) -> bytes:
        """Encode a graceful-shutdown CLOSE frame (empty payload).

        Returns:
            Binary WebSocket frame ready to send.
        """
        return bytes([ShellChannel.CLOSE])
