"""High-level async context manager for an interactive shell session."""

import asyncio
import json
import logging
import random
import uuid
from collections import deque
from typing import TYPE_CHECKING, AsyncIterator, Deque, Optional

import websockets
import websockets.exceptions

from ..models import SESSION_HEADER, SHELL_ID_HEADER
from ._validation import parse_runtime_arn, validate_shell_id
from .auth import AuthMode, OAuthAuth, PresignedAuth
from .config import _DEFAULT_METADATA_TIMEOUT, ReconnectConfig
from .protocol import ShellChannel, ShellFrame, ShellFramer

if TYPE_CHECKING:
    from ..agent_core_runtime_client import AgentCoreRuntimeClient

logger = logging.getLogger(__name__)


class ShellSession:
    r"""Async context manager wrapping a live interactive shell WebSocket session.

    Connects on ``__aenter__``, reads the mandatory metadata frame that carries
    ``shellId`` and ``reconnected``, and exposes typed send/resize/iterate/close.
    When ``reconnect_config`` is provided, transparently reconnects on unexpected
    disconnects using the same ``shell_id`` and ``session_id`` so the shell's
    working directory, environment, background jobs, and up to 256 KB of buffered
    output are preserved.

    Usage — SigV4 (default, server-side Python):
        client = AgentCoreRuntimeClient("us-west-2")
        async with client.open_shell(runtime_arn) as shell:
            await shell.send("cat /etc/os-release\\n")
            async for frame in shell:
                if frame.channel == ShellChannel.STDOUT:
                    print(frame.text, end="", flush=True)
                elif frame.channel == ShellChannel.STATUS:
                    # Termination: empty metadata (no shellId).
                    if not frame.json().get("metadata", {}).get("shellId"):
                        break

    Usage — presigned URL:
        async with client.open_shell(
            runtime_arn,
            auth=PresignedAuth(expires=120),
        ) as shell:
            ...

    Usage — OAuth (browser relay or OAuth-only environments):
        async with client.open_shell(
            runtime_arn,
            auth=OAuthAuth(bearer_token=await get_oauth_token()),
        ) as shell:
            ...

    Usage — auto-reconnect:
        config = ReconnectConfig(max_retries=5, on_reconnect=my_callback)
        async with client.open_shell(
            runtime_arn,
            shell_id="debug",
            reconnect_config=config,
        ) as shell:
            async for frame in shell:      # iterator survives network blips
                ...

    Usage — manual reconnect after network drop:
        # Both shell_id AND session_id are required.  shell_id names the PTY;
        # session_id routes to the VM that hosts it.
        # After an abrupt drop _ws is None, so __aexit__ skips the CLOSE frame
        # and the PTY stays alive on the server for the reconnect.
        shell_id = "debug"
        session_id = str(uuid.uuid4())   # generate once, reuse on every reconnect

        async with client.open_shell(
            runtime_arn, shell_id=shell_id, session_id=session_id
        ) as shell:
            async for frame in shell:
                ...   # StopAsyncIteration when the network drops

        async with client.open_shell(
            runtime_arn, shell_id=shell_id, session_id=session_id
        ) as shell:
            assert shell.reconnected       # True → up to 256 KB buffered output follows
            async for frame in shell:
                ...

    Attributes:
        shell_id: Confirmed shell identifier echoed by the server in
            the initial STATUS frame.  Preserve this value across your process
            restarts — passing the same ID to ``open_shell`` reconnects to the
            same PTY.
        session_id: Runtime session ID that routes to the VM hosting this shell.
            Auto-generated if not supplied.  Preserve this alongside ``shell_id``
            when reconnecting across process restarts — passing a different (or
            omitted) session ID may cause the platform to provision a fresh VM
            where the PTY no longer exists.
        reconnected: ``True`` when the session resumed an existing PTY (buffered
            output will arrive as STDOUT frames immediately after connect);
            ``False`` for a fresh shell.
        kicked: ``True`` when iteration stopped because another client connected
            with the same ``shell_id`` (close code 4000).  The PTY is
            still alive — a new ``open_shell`` call with the same ID will
            reconnect to it.
        bytes_dropped: Number of bytes lost from the PTY ring buffer during the
            most recent disconnect.  Non-zero only when the 256 KB ring buffer
            overflowed before reconnection completed.  Set after the post-drain
            STATUS confirmation frame arrives (which follows the buffered STDOUT
            burst).  Zero if no overflow occurred or on a fresh connection.
        exit_code: Exit code of the shell process.  ``None`` until the shell
            exits or if the platform terminated the session without providing an
            exit code (e.g. an InternalError).  ``0`` for a clean exit;
            non-zero for a failed command or a signal-killed process.  Set when
            the termination STATUS frame is processed — check this after the
            ``async for`` loop ends.

    Example::

        async with client.open_shell(runtime_arn) as shell:
            await shell.send("make build\\n")
            async for frame in shell:
                if frame.channel == ShellChannel.STDOUT:
                    print(frame.text, end="", flush=True)

        if shell.exit_code:  # None = no code available; 0 = clean exit
            raise RuntimeError(f"Build failed with exit code {shell.exit_code}")
    """

    def __init__(
        self,
        client: "AgentCoreRuntimeClient",
        runtime_arn: str,
        session_id: Optional[str] = None,
        shell_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        auth: AuthMode = "sigv4",
        reconnect_config: Optional[ReconnectConfig] = None,
    ) -> None:
        """Validate inputs and initialise session state; does not connect."""
        parsed = parse_runtime_arn(runtime_arn)
        client_region = getattr(client, "region", None)
        if isinstance(client_region, str) and parsed["region"] != client_region:
            raise ValueError(
                f"ARN region {parsed['region']!r} does not match client region {client_region!r}. "
                "Create a client for the same region as the runtime ARN, or use the ARN's region."
            )
        self._client = client
        self._runtime_arn = runtime_arn
        self._endpoint_name = endpoint_name
        self._auth = auth
        self._reconnect_config = reconnect_config
        self._framer = ShellFramer()
        self._ws: Optional[object] = None
        self._closed = False
        # Frames received during _read_metadata_frame that arrived before the
        # 0x03 confirmation (e.g. first shell prompt on 0x01).
        self._pending_frames: Deque[ShellFrame] = deque()

        if shell_id is not None:
            validate_shell_id(shell_id)
        # Auto-generate stable reconnect handles when the caller omits them.
        # Without a fixed session_id, each _connect() would route to a different
        # VM and shell_id would never be found → reconnected=False always.
        self.shell_id: str = shell_id or str(uuid.uuid4())
        self.session_id: str = session_id or str(uuid.uuid4())
        self.reconnected: bool = False
        self.kicked: bool = False
        self.bytes_dropped: int = 0
        self.exit_code: Optional[int] = None

    async def __aenter__(self) -> "ShellSession":
        """Connect and read the initial metadata frame."""
        try:
            await self._connect()
        except Exception as exc:
            logger.error("Failed to connect (shell_id=%r): %s", self.shell_id, exc)
            self._closed = True
            self._ws = None
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        """Send a graceful CLOSE frame and close the WebSocket."""
        await self.close()

    # ── Connection management ─────────────────────────────────────────────────

    async def _connect(self) -> None:
        """Open the WebSocket and consume the initial STATUS metadata frame."""
        self._pending_frames.clear()
        self.reconnected = False
        self.kicked = False

        auth = self._auth
        try:
            if isinstance(auth, OAuthAuth):
                url, subprotocols = self._client.connect_shell_oauth(
                    self._runtime_arn,
                    bearer_token=auth.bearer_token,
                    session_id=self.session_id,
                    endpoint_name=self._endpoint_name,
                    shell_id=self.shell_id,
                )
                self._ws = await websockets.connect(url, subprotocols=subprotocols)
            elif isinstance(auth, PresignedAuth):
                url = self._client.connect_shell_presigned(
                    self._runtime_arn,
                    session_id=self.session_id,
                    endpoint_name=self._endpoint_name,
                    shell_id=self.shell_id,
                    expires=auth.expires,
                )
                self._ws = await websockets.connect(url)
            else:
                # "sigv4" — default path
                url, headers = self._client.connect_shell(
                    self._runtime_arn,
                    session_id=self.session_id,
                    endpoint_name=self._endpoint_name,
                    shell_id=self.shell_id,
                )
                self._ws = await websockets.connect(url, additional_headers=headers)
        except websockets.exceptions.InvalidStatus as exc:
            body = exc.response.body.decode("utf-8", errors="replace")
            logger.error(
                "Server rejected WebSocket connection: HTTP %d %s%s (shell_id=%r)",
                exc.response.status_code,
                exc.response.reason_phrase,
                f" — {body}" if body else "",
                self.shell_id,
            )
            raise

        # read shellId from the 101 response header (primary path for
        # non-browser clients). The 0x03 frame is the fallback for browser
        # clients that cannot read 101 headers.
        response_headers = getattr(self._ws.response, "headers", {})
        header_csid = response_headers.get(SHELL_ID_HEADER)
        if header_csid:
            self.shell_id = header_csid
            logger.debug("shellId from 101 header: %r", header_csid)
        header_sid = response_headers.get(SESSION_HEADER)
        if header_sid:
            self.session_id = header_sid
            logger.debug("sessionId from 101 header: %r", header_sid)

        await self._read_metadata_frame()
        if self._ws is not None:
            self._closed = False

    async def _read_metadata_frame(self) -> None:
        """Consume the first STATUS frame carrying connection confirmation.

        Both the 0x03 confirmation and the first 0x01 stdout
        frame are sent after upgrade but their order is non-deterministic.  We
        wait for a 0x03 frame with metadata.shellId.  Any 0x01 frames
        that arrive first are stashed in self._pending_frames so __anext__ can
        yield them in order.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _DEFAULT_METADATA_TIMEOUT
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    "STATUS confirmation not received within %.1fs (deadline exceeded "
                    "processing earlier frames). Proceeding with client-generated "
                    "shell_id=%r — reconnected flag may be incorrect.",
                    _DEFAULT_METADATA_TIMEOUT,
                    self.shell_id,
                )
                return
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out waiting for STATUS confirmation after %.1fs "
                    "(server did not respond). Proceeding with client-generated "
                    "shell_id=%r — reconnected flag may be incorrect.",
                    _DEFAULT_METADATA_TIMEOUT,
                    self.shell_id,
                )
                return
            except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError) as exc:
                # Server closed before sending STATUS confirmation — session never became usable.
                logger.warning("WebSocket closed before STATUS confirmation (shell_id=%r): %s", self.shell_id, exc)
                self._ws = None
                self._closed = True
                raise
            except Exception as exc:
                logger.error("Unexpected error waiting for STATUS frame (shell_id=%r): %s", self.shell_id, exc)
                self._ws = None
                self._closed = True
                raise
            if not isinstance(raw, bytes):
                continue
            frame = self._framer.decode(raw)
            if frame.channel == ShellChannel.STATUS:
                try:
                    status = frame.json()
                    meta = status.get("metadata", {})
                    if meta.get("shellId"):
                        # This is the connection confirmation frame.
                        self.shell_id = meta["shellId"]
                        self.reconnected = bool(meta.get("reconnected", False))
                        return
                    else:
                        # Termination status — stash for __anext__.
                        self._pending_frames.append(frame)
                        return
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Received malformed STATUS frame during connect; skipping: %r", raw)
            else:
                # Non-STATUS frame (e.g. first stdout prompt) — stash for __anext__.
                self._pending_frames.append(frame)

    async def _run_inner_retry_loop(self, cfg: ReconnectConfig) -> bool:
        """Run one inner retry loop (up to max_retries attempts with exponential backoff).

        Returns True if a reconnect succeeded, False if all attempts failed.
        """
        delay = cfg.base_delay
        attempt = 0

        while attempt < cfg.max_retries:
            attempt += 1
            logger.info("Reconnect attempt %d (shell_id=%s)", attempt, self.shell_id)
            try:
                await self._connect()
                logger.info("Reconnected (reconnected=%s)", self.reconnected)
                if cfg.on_reconnect is not None:
                    result = cfg.on_reconnect(self.reconnected)
                    if asyncio.iscoroutine(result):
                        await result
                return True
            except Exception as exc:
                logger.info("Reconnect attempt %d failed: %s", attempt, exc)
                if attempt >= cfg.max_retries:
                    break
                jitter = random.uniform(-0.25 * delay, 0.25 * delay)
                await asyncio.sleep(min(delay + jitter, cfg.max_delay))
                delay = min(delay * 2, cfg.max_delay)

        return False

    async def _reconnect_with_backoff(self) -> bool:
        """Attempt to reconnect using a two-layer retry strategy.

        Inner loop: exponential backoff up to max_retries attempts.
        Outer loop: after inner loop exhaustion, wait outer_loop_delay seconds
            and run a fresh inner loop — until reconnect_window seconds have
            elapsed since the first disconnect.

        Returns:
            True if reconnect succeeded, False if the reconnection window expired.
        """
        cfg = self._reconnect_config
        if cfg is None:
            return False

        if cfg.reconnect_window is not None and cfg.reconnect_window <= 0:
            logger.warning(
                "reconnect_window=%.1f — reconnection disabled (shell_id=%s)",
                cfg.reconnect_window,
                self.shell_id,
            )
            return False

        loop = asyncio.get_running_loop()
        window_start = loop.time()
        outer_attempt = 0

        while True:
            outer_attempt += 1
            if outer_attempt > 1:
                logger.info(
                    "Starting outer retry cycle %d (shell_id=%s)",
                    outer_attempt,
                    self.shell_id,
                )
            if await self._run_inner_retry_loop(cfg):
                return True

            # Inner loop exhausted — check if the reconnection window allows another cycle.
            if cfg.reconnect_window is not None:
                elapsed = loop.time() - window_start
                if elapsed >= cfg.reconnect_window:
                    logger.warning(
                        "Reconnection window of %.0fs expired after %.0fs, giving up (shell_id=%s)",
                        cfg.reconnect_window,
                        elapsed,
                        self.shell_id,
                    )
                    return False

            logger.info(
                "Inner loop exhausted, waiting %.0fs before next outer retry cycle (shell_id=%s)",
                cfg.outer_loop_delay,
                self.shell_id,
            )
            await asyncio.sleep(cfg.outer_loop_delay)

    # ── Outbound ──────────────────────────────────────────────────────────────

    async def send(self, data: str) -> None:
        """Send keystrokes or paste text to the shell as a STDIN frame.

        Args:
            data: Text to send.  Encoded as UTF-8 before framing.
        """
        if self._ws is None:
            logger.error("send() called on a closed ShellSession (shell_id=%r)", self.shell_id)
            raise RuntimeError("Cannot send on a closed ShellSession")
        try:
            await self._ws.send(self._framer.encode_stdin(data))
        except Exception as exc:
            logger.error("Failed to send STDIN frame (shell_id=%r): %s", self.shell_id, exc)
            raise

    async def send_bytes(self, data: bytes) -> None:
        """Send raw bytes to the shell as a STDIN frame (e.g. escape sequences).

        Args:
            data: Raw bytes to send.
        """
        if self._ws is None:
            logger.error(
                "send_bytes() called on a closed ShellSession (shell_id=%r)",
                self.shell_id,
            )
            raise RuntimeError("Cannot send on a closed ShellSession")
        try:
            await self._ws.send(self._framer.encode_stdin(data))
        except Exception as exc:
            logger.error(
                "Failed to send STDIN (bytes) frame (shell_id=%r): %s",
                self.shell_id,
                exc,
            )
            raise

    async def resize(self, width: int, height: int) -> None:
        """Notify the PTY of a terminal resize.

        Args:
            width: New terminal width in columns.
            height: New terminal height in rows.
        """
        if self._ws is None:
            logger.error(
                "resize() called on a closed ShellSession (shell_id=%r)",
                self.shell_id,
            )
            raise RuntimeError("Cannot send on a closed ShellSession")
        try:
            await self._ws.send(self._framer.encode_resize(width, height))
        except Exception as exc:
            logger.error("Failed to send RESIZE frame (shell_id=%r): %s", self.shell_id, exc)
            raise

    async def close(self) -> None:
        """Send a graceful CLOSE frame and close the underlying WebSocket."""
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.send(self._framer.encode_close())
            except Exception as exc:
                # Best-effort — the connection may already be gone.
                logger.debug("Failed to send CLOSE frame during close() (shell_id=%r): %s", self.shell_id, exc)
            try:
                await self._ws.close()
            except Exception as exc:
                # Best-effort — swallow so close() never raises.
                logger.debug("Failed to close WebSocket during close() (shell_id=%r): %s", self.shell_id, exc)
            self._ws = None

    # ── Inbound ───────────────────────────────────────────────────────────────

    def __aiter__(self) -> AsyncIterator[ShellFrame]:
        """Return self as the async iterator."""
        return self

    @staticmethod
    def _is_termination_status(status: dict) -> bool:
        """Return True if a parsed STATUS frame payload signals shell exit or error.

        Connection confirmation frames have metadata.shellId present.
        Termination frames have an empty metadata dict.
        """
        meta = status.get("metadata", {})
        if meta.get("shellId"):
            # This is a connection confirmation frame, not a termination.
            return False
        # Empty metadata → shell exit (Success = code 0, Failure = non-zero / error).
        return True

    @staticmethod
    def _is_confirmation_status(status: dict) -> bool:
        """Return True if a parsed STATUS frame is a connection confirmation."""
        return bool(status.get("metadata", {}).get("shellId"))

    @staticmethod
    def _parse_exit_code(status: dict) -> Optional[int]:
        """Extract the integer exit code from a termination STATUS payload.

        Returns 0 for a successful exit, or the integer code from the ExitCode
        cause for a non-zero exit.  Returns None if the payload is missing the
        expected fields (e.g. a platform InternalError without an ExitCode cause)
        so callers can distinguish "exited cleanly" from "no exit code available".
        """
        if status.get("status") == "Success":
            return 0
        causes = status.get("details", {}).get("causes", [])
        for cause in causes:
            if cause.get("reason") == "ExitCode":
                try:
                    return int(cause["message"])
                except (ValueError, KeyError):
                    pass
        return None

    async def __anext__(self) -> ShellFrame:
        """Yield the next inbound frame, reconnecting on drop if configured.

        When ``reconnect_config`` is set, a WebSocket disconnect triggers an
        automatic reconnect attempt using the same ``shell_id``.  The
        iterator resumes transparently — callers do not need to re-enter the
        context manager.  The ``on_reconnect`` callback fires after each
        successful reconnect so callers can react to the ``reconnected`` flag
        and the incoming buffered-output burst.

        close code 4000 ("kicked by new connection") MUST NOT trigger auto-reconnect.
        The iterator stops and sets ``self.kicked = True`` so callers can distinguish this case.

        Raises:
            StopAsyncIteration: When the shell exits cleanly (CLOSE frame or
                STATUS frame with empty metadata), when ``close()`` has been
                called, or when reconnect attempts are exhausted.
        """
        while True:
            # Drain any frames buffered during the metadata handshake first.
            if self._pending_frames:
                frame = self._pending_frames.popleft()
                if frame.channel == ShellChannel.CLOSE:
                    logger.debug(
                        "CLOSE frame received from pending queue (shell_id=%r)",
                        self.shell_id,
                    )
                    raise StopAsyncIteration from None
                if frame.channel == ShellChannel.STATUS:
                    try:
                        status = frame.json()
                        if self._is_termination_status(status):
                            self.exit_code = self._parse_exit_code(status)
                            self._closed = True
                            return frame
                    except (json.JSONDecodeError, KeyError):
                        logger.warning(
                            "Received malformed STATUS frame in pending queue; skipping termination check: %r",
                            frame,
                        )
                return frame

            if self._closed or self._ws is None:
                logger.debug("Session already closed or disconnected (shell_id=%r)", self.shell_id)
                raise StopAsyncIteration from None

            try:
                raw = await self._ws.recv()
            except Exception as exc:
                if self._closed:
                    logger.debug(
                        "WebSocket error after close() (shell_id=%r): %s",
                        self.shell_id,
                        exc,
                    )
                    raise StopAsyncIteration from None
                # Detect close code 4000 (kicked by new connection) — must NOT reconnect.
                if isinstance(exc, websockets.exceptions.ConnectionClosedOK):
                    if exc.rcvd is not None and exc.rcvd.code == 1001:
                        # Code 1001 "Going Away" — DP/proxy restarting; MUST reconnect.
                        if self._reconnect_config is None:
                            logger.warning(
                                "Server sent 1001 Going Away but no reconnect_config "
                                "provided — stopping iteration. Reconnect with the same "
                                "shell_id=%r to resume the PTY.",
                                self.shell_id,
                            )
                            self._ws = None
                            self._closed = True
                            raise StopAsyncIteration from None
                        logger.info(
                            "WebSocket closed with 1001 Going Away — will reconnect (shell_id=%r)",
                            self.shell_id,
                        )
                    else:
                        # Code 1000 "Normal Closure" — shell exited or graceful shutdown.
                        logger.info(
                            "WebSocket closed cleanly (shell_id=%r): %s",
                            self.shell_id,
                            exc,
                        )
                        self._ws = None
                        self._closed = True
                        raise StopAsyncIteration from None
                if isinstance(exc, websockets.exceptions.ConnectionClosedError):
                    if exc.rcvd is not None and exc.rcvd.code == 4000:
                        logger.info(
                            "Shell session kicked (close code 4000) — will not reconnect (shell_id=%r)",
                            self.shell_id,
                        )
                        self._ws = None
                        self.kicked = True
                        raise StopAsyncIteration from None
                    logger.warning(
                        "WebSocket closed unexpectedly (shell_id=%r): %s",
                        self.shell_id,
                        exc,
                    )
                reconnected = await self._reconnect_with_backoff()
                if not reconnected:
                    logger.warning(
                        "Reconnect exhausted, stopping iteration (shell_id=%r)",
                        self.shell_id,
                    )
                    self._ws = None
                    self._closed = True
                    raise StopAsyncIteration from None
                # Iterator resumes from the top of the loop with the new WebSocket.
                continue

            if not isinstance(raw, bytes):
                continue

            frame = self._framer.decode(raw)

            if frame.channel == ShellChannel.CLOSE:
                logger.debug("CLOSE frame received (shell_id=%r)", self.shell_id)
                raise StopAsyncIteration from None

            if frame.channel == ShellChannel.HEARTBEAT:
                # Wire-level keepalive — swallow the server echo, don't surface to caller.
                continue

            if frame.channel == ShellChannel.STATUS:
                try:
                    status = frame.json()
                    if self._is_confirmation_status(status):
                        # Second confirmation frame (post-drain) — carries
                        # bytesDropped when the 256 KB ring buffer overflowed.
                        # Swallow it; surface bytesDropped via attribute + warning.
                        dropped = status.get("metadata", {}).get("bytesDropped", 0)
                        if dropped:
                            self.bytes_dropped = dropped
                            logger.warning(
                                "%d bytes of PTY output lost during disconnect (ring buffer overflow) (shell_id=%r)",
                                dropped,
                                self.shell_id,
                            )
                        continue
                    if self._is_termination_status(status):
                        # Shell exited — mark closed so the next __anext__ call
                        # stops immediately instead of triggering auto-reconnect.
                        self.exit_code = self._parse_exit_code(status)
                        self._closed = True
                        return frame
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Received malformed STATUS frame; skipping termination check: %r", frame)

            return frame
