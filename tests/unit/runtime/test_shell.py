"""Tests for ShellSession and ReconnectConfig."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

# websockets.connect is awaited in _connect(), so patches must be AsyncMock.
import pytest

from bedrock_agentcore.runtime.shell import (
    OAuthAuth,
    PresignedAuth,
    ReconnectConfig,
    ShellChannel,
    ShellFramer,
    ShellSession,
)

FAKE_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime"
FAKE_URL = "wss://bedrock-agentcore.us-west-2.amazonaws.com/runtimes/FAKE/ws/commands"
FAKE_HEADERS = {"Authorization": "AWS4-HMAC…", "Host": "bedrock-agentcore.us-west-2.amazonaws.com"}


def _make_client(url=FAKE_URL, headers=FAKE_HEADERS):
    client = MagicMock()
    client.connect_shell.return_value = (url, headers)
    return client


def _make_ws(*frames, end_with=None):
    """Build a mock WebSocket that yields the given raw frames then raises end_with.

    end_with defaults to ConnectionClosedOK — signals clean close to __anext__,
    which raises StopAsyncIteration without triggering reconnect.
    """
    import websockets.exceptions

    if end_with is None:
        end_with = websockets.exceptions.ConnectionClosedOK(None, None)

    ws = AsyncMock()
    call_count = 0

    async def recv():
        nonlocal call_count
        if call_count < len(frames):
            result = frames[call_count]
            call_count += 1
            return result
        raise end_with

    ws.recv = recv
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _metadata_frame(shell_id: str = "test-session", reconnected: bool = False) -> bytes:
    payload = json.dumps(
        {
            "kind": "Status",
            "apiVersion": "v1",
            "metadata": {"commandSessionId": shell_id, "reconnected": reconnected},
            "status": "Success",
        }
    ).encode()
    return bytes([ShellChannel.STATUS]) + payload


def _stdout_frame(text: str) -> bytes:
    return bytes([ShellChannel.STDOUT]) + text.encode()


def _exit_frame(exit_code: int = 0) -> bytes:
    if exit_code == 0:
        payload = json.dumps({"kind": "Status", "apiVersion": "v1", "metadata": {}, "status": "Success"}).encode()
    else:
        payload = json.dumps(
            {
                "kind": "Status",
                "apiVersion": "v1",
                "metadata": {},
                "status": "Failure",
                "reason": "NonZeroExitCode",
                "details": {"causes": [{"reason": "ExitCode", "message": str(exit_code)}]},
            }
        ).encode()
    return bytes([ShellChannel.STATUS]) + payload


def _close_frame() -> bytes:
    return bytes([ShellChannel.CLOSE])


class TestShellSessionConnect:
    @pytest.mark.asyncio
    async def test_connect_reads_metadata_frame(self):
        client = _make_client()
        ws = _make_ws(_metadata_frame("my-shell", reconnected=False))

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            session = ShellSession(client, FAKE_ARN, shell_id="my-shell")
            await session._connect()

        assert session.shell_id == "my-shell"
        assert session.reconnected is False

    @pytest.mark.asyncio
    async def test_connect_sets_reconnected_true(self):
        client = _make_client()
        ws = _make_ws(_metadata_frame("my-shell", reconnected=True))

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            session = ShellSession(client, FAKE_ARN, shell_id="my-shell")
            await session._connect()

        assert session.reconnected is True

    @pytest.mark.asyncio
    async def test_connect_tolerates_missing_metadata_frame(self):
        client = _make_client()
        ws = AsyncMock()
        ws.recv = AsyncMock(side_effect=asyncio.TimeoutError)
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            session = ShellSession(client, FAKE_ARN)
            await session._connect()  # must not raise

        assert session.shell_id is not None

    @pytest.mark.asyncio
    async def test_connect_ignores_non_status_first_frame(self):
        """STDOUT frames arriving before STATUS are stashed; STATUS is still found."""
        client = _make_client()
        ws = _make_ws(_stdout_frame("some output"), _metadata_frame("x"))

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            session = ShellSession(client, FAKE_ARN, shell_id="x")
            await session._connect()

        assert session.reconnected is False
        assert session.shell_id == "x"
        assert len(session._pending_frames) == 1  # stdout frame was stashed

    @pytest.mark.asyncio
    async def test_connect_raises_on_connection_closed_error_before_status(self):
        """ConnectionClosedError before STATUS arrives propagates out of __aenter__."""
        import websockets.exceptions

        client = _make_client()
        ws = AsyncMock()
        ws.recv = AsyncMock(side_effect=websockets.exceptions.ConnectionClosedError(None, None))
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            session = ShellSession(client, FAKE_ARN)
            with pytest.raises(websockets.exceptions.ConnectionClosedError):
                await session._connect()

        assert session._ws is None

    @pytest.mark.asyncio
    async def test_connect_raises_on_connection_closed_ok_before_status(self):
        """ConnectionClosedOK before STATUS arrives propagates out of __aenter__."""
        import websockets.exceptions

        client = _make_client()
        ws = AsyncMock()
        ws.recv = AsyncMock(side_effect=websockets.exceptions.ConnectionClosedOK(None, None))
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            session = ShellSession(client, FAKE_ARN)
            with pytest.raises(websockets.exceptions.ConnectionClosedOK):
                await session._connect()

        assert session._ws is None

    @pytest.mark.asyncio
    async def test_connect_timeout_proceeds_with_warning(self, caplog):
        """TimeoutError waiting for STATUS logs a warning but session stays usable."""
        import logging

        client = _make_client()
        ws = AsyncMock()
        ws.recv = AsyncMock(side_effect=asyncio.TimeoutError)
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            with caplog.at_level(logging.WARNING, logger="bedrock_agentcore.runtime.shell"):
                session = ShellSession(client, FAKE_ARN)
                await session._connect()  # must not raise

        assert session._ws is ws  # connection still alive
        assert "server did not respond" in caplog.text


class TestShellSessionInit:
    def test_invalid_arn_raises_at_construction(self):
        client = _make_client()
        with pytest.raises(ValueError):
            ShellSession(client, "not-an-arn")

    def test_none_shell_id_autogenerates_uuid(self):
        client = _make_client()
        session = ShellSession(client, FAKE_ARN, shell_id=None)
        assert session._shell_id is not None
        assert len(session._shell_id) > 0

    @pytest.mark.parametrize("bad", [0, False, b"", 1.5])
    def test_non_string_shell_id_raises_at_construction(self, bad):
        """Non-string values must be rejected immediately in __init__, not silently
        replaced with a UUID or deferred until __aenter__."""
        client = _make_client()
        with pytest.raises(ValueError, match="must be str"):
            ShellSession(client, FAKE_ARN, shell_id=bad)

    def test_region_mismatch_raises_at_construction(self):
        client = MagicMock()
        client.region = "us-east-1"
        with pytest.raises(ValueError, match="does not match client region"):
            ShellSession(client, FAKE_ARN)  # FAKE_ARN is us-west-2

    def test_region_match_does_not_raise(self):
        client = MagicMock()
        client.region = "us-west-2"
        session = ShellSession(client, FAKE_ARN)
        assert session is not None


class TestShellSessionConnectInvalidStatus:
    @pytest.mark.asyncio
    async def test_invalid_status_logs_body_and_reraises(self, caplog):
        import logging

        import websockets.datastructures
        import websockets.exceptions
        import websockets.http11

        client = _make_client()
        response = websockets.http11.Response(
            status_code=404,
            reason_phrase="Not Found",
            headers=websockets.datastructures.Headers(),
            body=b"No endpoint found for qualifier 'DEFAULT'",
        )
        exc = websockets.exceptions.InvalidStatus(response)

        with patch("websockets.connect", new=AsyncMock(side_effect=exc)):
            session = ShellSession(client, FAKE_ARN)
            with caplog.at_level(logging.ERROR, logger="bedrock_agentcore.runtime.shell"):
                with pytest.raises(websockets.exceptions.InvalidStatus):
                    await session._connect()

        assert "404" in caplog.text
        assert "No endpoint found" in caplog.text


class TestShellSessionAenterFailure:
    @pytest.mark.asyncio
    async def test_aenter_failure_marks_closed(self):
        """If _connect() raises, __aenter__ must leave _closed=True so the session
        cannot be iterated."""
        client = _make_client()

        with patch("websockets.connect", new=AsyncMock(side_effect=ConnectionRefusedError("refused"))):
            session = ShellSession(client, FAKE_ARN)
            with pytest.raises(ConnectionRefusedError):
                await session.__aenter__()

        assert session._closed is True
        assert session._ws is None


class TestShellSessionContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exit(self):
        client = _make_client()
        ws = _make_ws(_metadata_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                assert shell._ws is not None
            assert shell._ws is None
            ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_sends_close_frame(self):
        framer = ShellFramer()
        client = _make_client()
        ws = _make_ws(_metadata_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as _:
                pass

        sent_frames = [call.args[0] for call in ws.send.call_args_list]
        assert framer.encode_close() in sent_frames


class TestShellSessionSend:
    @pytest.mark.asyncio
    async def test_send_encodes_stdin_frame(self):
        framer = ShellFramer()
        client = _make_client()
        ws = _make_ws(_metadata_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                await shell.send("ls\n")

        sent = [call.args[0] for call in ws.send.call_args_list]
        assert framer.encode_stdin("ls\n") in sent

    @pytest.mark.asyncio
    async def test_send_bytes(self):
        framer = ShellFramer()
        client = _make_client()
        ws = _make_ws(_metadata_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                await shell.send_bytes(b"\x1b[A")

        sent = [call.args[0] for call in ws.send.call_args_list]
        assert framer.encode_stdin(b"\x1b[A") in sent

    @pytest.mark.asyncio
    async def test_resize(self):
        framer = ShellFramer()
        client = _make_client()
        ws = _make_ws(_metadata_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                await shell.resize(220, 50)

        sent = [call.args[0] for call in ws.send.call_args_list]
        assert framer.encode_resize(220, 50) in sent


class TestShellSessionSendAfterClose:
    @pytest.mark.asyncio
    async def test_send_raises_after_close(self):
        client = _make_client()
        ws = _make_ws(_metadata_frame())
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                pass
        with pytest.raises(RuntimeError, match="closed"):
            await shell.send("ls\n")

    @pytest.mark.asyncio
    async def test_send_bytes_raises_after_close(self):
        client = _make_client()
        ws = _make_ws(_metadata_frame())
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                pass
        with pytest.raises(RuntimeError, match="closed"):
            await shell.send_bytes(b"\x1b[A")

    @pytest.mark.asyncio
    async def test_resize_raises_after_close(self):
        client = _make_client()
        ws = _make_ws(_metadata_frame())
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                pass
        with pytest.raises(RuntimeError, match="closed"):
            await shell.resize(80, 24)


class TestShellSessionIterate:
    @pytest.mark.asyncio
    async def test_iterates_stdout_frames(self):
        client = _make_client()
        ws = _make_ws(
            _metadata_frame(),
            _stdout_frame("hello"),
            _stdout_frame(" world"),
            _close_frame(),
        )

        output = []
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                async for frame in shell:
                    if frame.channel == ShellChannel.STDOUT:
                        output.append(frame.text)

        assert output == ["hello", " world"]

    @pytest.mark.asyncio
    async def test_stops_on_exit_status_frame(self):
        client = _make_client()
        ws = _make_ws(
            _metadata_frame(),
            _stdout_frame("output"),
            _exit_frame(0),
        )

        frames = []
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                async for frame in shell:
                    frames.append(frame)

        channels = [f.channel for f in frames]
        assert ShellChannel.STDOUT in channels
        # The exit STATUS frame is yielded (so caller can read exit code), then iteration stops.
        assert ShellChannel.STATUS in channels
        # The exit STATUS frame has empty metadata (not a connection confirmation).
        status_frames = [f for f in frames if f.channel == ShellChannel.STATUS]
        assert all(not f.json().get("metadata", {}).get("commandSessionId") for f in status_frames)

    @pytest.mark.asyncio
    async def test_stops_on_close_frame(self):
        client = _make_client()
        ws = _make_ws(_metadata_frame(), _close_frame())

        frames = []
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                async for frame in shell:
                    frames.append(frame)

        assert frames == []

    @pytest.mark.asyncio
    async def test_stops_on_connection_closed_without_reconnect(self):
        client = _make_client()
        ws = _make_ws(_metadata_frame())  # ConnectionClosed raised on next recv

        frames = []
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                async for frame in shell:
                    frames.append(frame)

        assert frames == []


class TestShellSessionCleanClose:
    @pytest.mark.asyncio
    async def test_connection_closed_ok_stops_without_reconnect(self):
        """ConnectionClosedOK (server graceful close, code 1000) must not trigger auto-reconnect."""
        import websockets.exceptions

        client = _make_client()
        reconnect_calls = []

        async def on_reconnect(reconnected: bool) -> None:
            reconnect_calls.append(reconnected)

        ws = AsyncMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        call_count = 0

        async def recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _metadata_frame()
            raise websockets.exceptions.ConnectionClosedOK(None, None)

        ws.recv = recv

        config = ReconnectConfig(max_retries=3, base_delay=0.0, on_reconnect=on_reconnect)
        frames = []
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN, reconnect_config=config) as shell:
                async for frame in shell:
                    frames.append(frame)

        assert frames == []
        assert reconnect_calls == []  # no reconnect attempted

    @pytest.mark.asyncio
    async def test_termination_status_prevents_reconnect(self):
        """Shell exit STATUS frame must set _closed so reconnect_config does not reopen the shell."""
        import websockets.exceptions

        client = _make_client()
        reconnect_calls = []

        async def on_reconnect(reconnected: bool) -> None:
            reconnect_calls.append(reconnected)

        ws = AsyncMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        call_count = 0

        async def recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _metadata_frame()
            if call_count == 2:
                return _exit_frame(0)
            # If _closed wasn't set, __anext__ would hit this and try to reconnect
            raise websockets.exceptions.ConnectionClosedOK(None, None)

        ws.recv = recv

        config = ReconnectConfig(max_retries=3, base_delay=0.0, on_reconnect=on_reconnect)
        frames = []
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN, reconnect_config=config) as shell:
                async for frame in shell:
                    frames.append(frame)

        # The exit STATUS frame was yielded
        assert len(frames) == 1
        assert frames[0].channel == ShellChannel.STATUS
        # No reconnect was attempted after the shell exited
        assert reconnect_calls == []

    @pytest.mark.asyncio
    async def test_close_code_1001_triggers_reconnect(self):
        """Close code 1001 (Going Away — DP/proxy restart) MUST trigger auto-reconnect per spec §6."""
        import websockets.exceptions
        from websockets.frames import Close

        client = _make_client()
        reconnect_calls = []

        async def on_reconnect(reconnected: bool) -> None:
            reconnect_calls.append(reconnected)

        # First WebSocket: metadata then close 1001.
        # Second WebSocket: metadata then clean close 1000.
        ws1 = AsyncMock()
        ws1.send = AsyncMock()
        ws1.close = AsyncMock()
        ws1_count = 0

        async def recv1():
            nonlocal ws1_count
            ws1_count += 1
            if ws1_count == 1:
                return _metadata_frame("session-1")
            raise websockets.exceptions.ConnectionClosedOK(Close(1001, "Going Away"), None)

        ws1.recv = recv1

        ws2 = AsyncMock()
        ws2.send = AsyncMock()
        ws2.close = AsyncMock()
        ws2_count = 0

        async def recv2():
            nonlocal ws2_count
            ws2_count += 1
            if ws2_count == 1:
                return _metadata_frame("session-1", reconnected=True)
            if ws2_count == 2:
                return _stdout_frame("hello")
            raise websockets.exceptions.ConnectionClosedOK(None, None)

        ws2.recv = recv2

        connect_calls = iter([ws1, ws2])
        config = ReconnectConfig(max_retries=2, base_delay=0.0, reconnect_window=60.0, on_reconnect=on_reconnect)
        frames = []
        with patch("websockets.connect", new=AsyncMock(side_effect=lambda *a, **kw: next(connect_calls))):
            async with ShellSession(client, FAKE_ARN, shell_id="session-1", reconnect_config=config) as shell:
                async for frame in shell:
                    frames.append(frame)

        assert len(frames) == 1
        assert frames[0].channel == ShellChannel.STDOUT
        assert frames[0].text == "hello"
        assert len(reconnect_calls) == 1  # reconnected once after 1001


class TestShellSessionKicked:
    @pytest.mark.asyncio
    async def test_close_code_4000_stops_without_reconnect(self):
        """Close code 4000 (kicked by new connection) must NOT trigger auto-reconnect."""
        import websockets.exceptions
        from websockets.frames import Close

        client = _make_client()
        ws = AsyncMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        call_count = 0

        async def recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _metadata_frame()
            # Simulate kicked close code 4000
            close_obj = Close(code=4000, reason="replaced by new connection")
            raise websockets.exceptions.ConnectionClosedError(close_obj, None)

        ws.recv = recv

        config = ReconnectConfig(max_retries=3, base_delay=0.0)
        frames = []
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN, reconnect_config=config) as shell:
                async for frame in shell:
                    frames.append(frame)

        # Iteration stopped and no reconnect was attempted (only one connect call)
        assert frames == []

    @pytest.mark.asyncio
    async def test_close_code_4000_sets_kicked(self):
        """shell.kicked is True after close code 4000; shell.kicked is False for a clean close."""
        import websockets.exceptions
        from websockets.frames import Close

        client = _make_client()
        ws = AsyncMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        call_count = 0

        async def recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _metadata_frame()
            close_obj = Close(code=4000, reason="replaced by new connection")
            raise websockets.exceptions.ConnectionClosedError(close_obj, None)

        ws.recv = recv

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                async for _ in shell:
                    pass

        assert shell.kicked is True

    @pytest.mark.asyncio
    async def test_clean_close_does_not_set_kicked(self):
        """shell.kicked remains False when the session ends normally."""
        client = _make_client()
        ws = _make_ws(_metadata_frame())  # metadata frame then ConnectionClosedOK
        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                async for _ in shell:
                    pass

        assert shell.kicked is False


class TestShellSessionAutoReconnect:
    @pytest.mark.asyncio
    async def test_reconnects_and_resumes_iteration(self):
        import websockets.exceptions

        client = _make_client()

        ws1 = _make_ws(
            _metadata_frame("s", reconnected=False),
            _stdout_frame("before"),
            end_with=websockets.exceptions.ConnectionClosedError(None, None),
        )
        ws2 = _make_ws(_metadata_frame("s", reconnected=True), _stdout_frame("after"), _close_frame())

        connect_calls = [ws1, ws2]

        async def fake_connect(url, extra_headers=None, additional_headers=None, **_kw):
            return connect_calls.pop(0)

        on_reconnect_calls = []

        async def on_reconnect(reconnected: bool) -> None:
            on_reconnect_calls.append(reconnected)

        config = ReconnectConfig(max_retries=3, on_reconnect=on_reconnect)
        output = []

        with patch("websockets.connect", side_effect=fake_connect):
            async with ShellSession(client, FAKE_ARN, shell_id="s", reconnect_config=config) as shell:
                async for frame in shell:
                    if frame.channel == ShellChannel.STDOUT:
                        output.append(frame.text)

        assert output == ["before", "after"]
        assert on_reconnect_calls == [True]

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_stops(self):
        client = _make_client()
        ws = _make_ws(_metadata_frame())  # drops immediately after metadata

        async def fail_connect(url, extra_headers=None):
            raise ConnectionRefusedError("server down")

        async def first_connect(url, extra_headers=None):
            return ws

        call_count = 0

        async def fake_connect(url, extra_headers=None, additional_headers=None, **_kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ws
            raise ConnectionRefusedError("server down")

        config = ReconnectConfig(max_retries=2, base_delay=0.0, max_delay=0.0, reconnect_window=0.0)

        frames = []
        with patch("websockets.connect", side_effect=fake_connect):
            async with ShellSession(client, FAKE_ARN, reconnect_config=config) as shell:
                async for frame in shell:
                    frames.append(frame)

        # All reconnect attempts failed — iteration stopped gracefully
        assert frames == []

    @pytest.mark.asyncio
    async def test_pending_frames_cleared_on_reconnect(self):
        """_pending_frames must be cleared at the start of _connect() so frames
        buffered during a previous metadata handshake cannot bleed into a new session.

        Directly verifies the invariant: after __aenter__ plants a stale frame in
        _pending_frames, calling _connect() again must empty it.
        """
        client = _make_client()

        from bedrock_agentcore.runtime.shell.protocol import ShellFrame

        ws1 = _make_ws(_metadata_frame("s"))
        ws2 = _make_ws(_metadata_frame("s", reconnected=True), _stdout_frame("fresh"), _close_frame())
        connect_seq = [ws1, ws2]

        async def fake_connect(url, extra_headers=None, additional_headers=None, **_kw):
            return connect_seq.pop(0)

        with patch("websockets.connect", side_effect=fake_connect):
            session = ShellSession(client, FAKE_ARN, shell_id="s")
            await session.__aenter__()

            # Plant a stale frame as if left over from a prior metadata handshake.
            session._pending_frames.append(
                ShellFrame(channel=ShellChannel.STDOUT, raw_channel_byte=ShellChannel.STDOUT, payload=b"stale")
            )
            assert len(session._pending_frames) == 1

            # _connect() must clear it.
            await session._connect()
            assert len(session._pending_frames) == 0

            await session.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_sync_on_reconnect_callback_accepted(self):
        """A synchronous on_reconnect callback must also be accepted."""
        import websockets.exceptions

        client = _make_client()
        ws1 = _make_ws(_metadata_frame("s"), end_with=websockets.exceptions.ConnectionClosedError(None, None))
        ws2 = _make_ws(_metadata_frame("s", reconnected=True), _close_frame())
        connect_calls = [ws1, ws2]

        async def fake_connect(url, extra_headers=None, additional_headers=None, **_kw):
            return connect_calls.pop(0)

        sync_calls = []

        def sync_callback(reconnected: bool) -> None:
            sync_calls.append(reconnected)

        config = ReconnectConfig(max_retries=1, base_delay=0.0, on_reconnect=sync_callback)

        with patch("websockets.connect", side_effect=fake_connect):
            async with ShellSession(client, FAKE_ARN, shell_id="s", reconnect_config=config) as shell:
                async for _ in shell:
                    pass

        assert sync_calls == [True]

    @pytest.mark.asyncio
    async def test_outer_loop_retries_after_inner_exhaustion(self):
        """After inner loop exhaustion, outer loop waits then runs a fresh inner loop."""
        import websockets.exceptions

        client = _make_client()
        ws1 = _make_ws(_metadata_frame("s"), end_with=websockets.exceptions.ConnectionClosedError(None, None))
        ws2 = _make_ws(_metadata_frame("s", reconnected=True), _close_frame())

        call_count = 0

        async def fake_connect(url, extra_headers=None, additional_headers=None, **_kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ws1
            if call_count == 2:
                # Fail the inner loop's first attempt so outer loop kicks in
                raise ConnectionRefusedError("transient")
            return ws2

        config = ReconnectConfig(
            max_retries=1,  # inner loop gives up after 1 failed attempt
            base_delay=0.0,
            max_delay=0.0,
            reconnect_window=60.0,
            outer_loop_delay=0.0,  # no real sleep in tests
        )
        output = []
        with patch("websockets.connect", side_effect=fake_connect):
            async with ShellSession(client, FAKE_ARN, shell_id="s", reconnect_config=config) as shell:
                async for frame in shell:
                    if frame.channel == ShellChannel.STDOUT:
                        output.append(frame.text)

        # ws2 succeeded on the second outer cycle
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_reconnect_window_zero_skips_inner_loop(self):
        """reconnect_window=0.0 must give up immediately — no inner retry attempts at all."""
        client = _make_client()
        ws = _make_ws(_metadata_frame())
        connect_count = 0

        async def fake_connect(url, extra_headers=None, additional_headers=None, **_kw):
            nonlocal connect_count
            connect_count += 1
            if connect_count == 1:
                return ws
            raise ConnectionRefusedError("should never be reached")

        config = ReconnectConfig(max_retries=5, base_delay=0.0, reconnect_window=0.0)
        with patch("websockets.connect", side_effect=fake_connect):
            async with ShellSession(client, FAKE_ARN, reconnect_config=config) as shell:
                async for _ in shell:
                    pass

        assert connect_count == 1  # no retries attempted

    @pytest.mark.asyncio
    async def test_reconnect_window_expiry_stops_iteration(self):
        """When reconnect_window=0.0 the outer loop gives up immediately after inner exhaustion."""

        client = _make_client()
        ws = _make_ws(_metadata_frame())

        call_count = 0

        async def fake_connect(url, extra_headers=None, additional_headers=None, **_kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ws
            raise ConnectionRefusedError("server down")

        config = ReconnectConfig(
            max_retries=2,
            base_delay=0.0,
            max_delay=0.0,
            reconnect_window=0.0,  # window already expired — no outer loop
        )
        frames = []
        with patch("websockets.connect", side_effect=fake_connect):
            async with ShellSession(client, FAKE_ARN, reconnect_config=config) as shell:
                async for frame in shell:
                    frames.append(frame)

        assert frames == []


class TestShellSessionExitCode:
    @pytest.mark.asyncio
    async def test_exit_code_zero_on_clean_exit(self):
        """exit_code is 0 after a clean shell exit (status=Success)."""
        client = _make_client()
        ws = _make_ws(_metadata_frame(), _exit_frame(0))

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                async for _ in shell:
                    pass

        assert shell.exit_code == 0

    @pytest.mark.asyncio
    async def test_exit_code_nonzero(self):
        """exit_code reflects non-zero exit status from ExitCode cause."""
        client = _make_client()
        ws = _make_ws(_metadata_frame(), _exit_frame(42))

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                async for _ in shell:
                    pass

        assert shell.exit_code == 42

    @pytest.mark.asyncio
    async def test_exit_code_none_before_exit(self):
        """exit_code is None until the termination STATUS frame is processed."""
        client = _make_client()
        ws = _make_ws(_metadata_frame(), _stdout_frame("hi"), _exit_frame(1))

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                exit_code_mid_loop = None
                async for frame in shell:
                    if frame.channel == ShellChannel.STDOUT:
                        exit_code_mid_loop = shell.exit_code

        assert exit_code_mid_loop is None  # not set yet during STDOUT frame
        assert shell.exit_code == 1

    @pytest.mark.asyncio
    async def test_exit_code_set_via_pending_frames_path(self):
        """exit_code is set when the termination STATUS is drained from _pending_frames."""
        import websockets.exceptions

        client = _make_client()
        ws = AsyncMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        call_count = 0

        async def recv():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First recv during _connect() returns a STDOUT frame — stashed as pending
                return _stdout_frame("stashed")
            if call_count == 2:
                # Second recv during _connect() returns the metadata confirmation
                return _metadata_frame()
            if call_count == 3:
                return _exit_frame(5)
            raise websockets.exceptions.ConnectionClosedOK(None, None)

        ws.recv = recv

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                frames = [frame async for frame in shell]

        # stashed STDOUT frame drained first, then the exit STATUS frame
        assert frames[0].channel == ShellChannel.STDOUT
        assert frames[1].channel == ShellChannel.STATUS
        assert shell.exit_code == 5

    @pytest.mark.asyncio
    async def test_exit_code_platform_error_without_exit_code_cause(self):
        """exit_code is None when a Failure STATUS has no ExitCode cause (e.g. InternalError)."""
        import json

        platform_error = json.dumps(
            {
                "kind": "Status",
                "apiVersion": "v1",
                "metadata": {},
                "status": "Failure",
                "reason": "InternalError",
                "message": "init container failed",
                "code": 500,
            }
        ).encode()
        client = _make_client()
        ws = _make_ws(_metadata_frame(), bytes([ShellChannel.STATUS]) + platform_error)

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                async for _ in shell:
                    pass

        assert shell.exit_code is None


class TestShellSessionBytesDropped:
    def _second_confirmation_frame(self, shell_id: str = "test-session", bytes_dropped: int = 1024) -> bytes:
        payload = json.dumps(
            {
                "kind": "Status",
                "apiVersion": "v1",
                "metadata": {"commandSessionId": shell_id, "reconnected": True, "bytesDropped": bytes_dropped},
                "status": "Success",
            }
        ).encode()
        return bytes([0x03]) + payload

    @pytest.mark.asyncio
    async def test_bytes_dropped_set_on_second_confirmation(self):
        """Second confirmation frame with bytesDropped sets shell.bytes_dropped."""
        client = _make_client()
        ws = _make_ws(
            _metadata_frame("s"),
            _stdout_frame("output"),
            self._second_confirmation_frame("s", bytes_dropped=512),
            _close_frame(),
        )

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN, shell_id="s") as shell:
                frames = [frame async for frame in shell]

        assert shell.bytes_dropped == 512
        # Second confirmation must be swallowed — only stdout frame yielded
        assert len(frames) == 1
        assert frames[0].channel == ShellChannel.STDOUT

    @pytest.mark.asyncio
    async def test_bytes_dropped_zero_when_no_overflow(self):
        """bytes_dropped stays 0 when no second confirmation arrives."""
        client = _make_client()
        ws = _make_ws(_metadata_frame(), _close_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN) as shell:
                async for _ in shell:
                    pass

        assert shell.bytes_dropped == 0

    @pytest.mark.asyncio
    async def test_second_confirmation_without_bytes_dropped_swallowed(self):
        """Second confirmation with no bytesDropped field is still swallowed."""
        client = _make_client()
        # Second confirmation without bytesDropped (single overflow frame edge case)
        second_conf = json.dumps(
            {
                "kind": "Status",
                "apiVersion": "v1",
                "metadata": {"commandSessionId": "s", "reconnected": True},
                "status": "Success",
            }
        ).encode()
        ws = _make_ws(
            _metadata_frame("s"),
            bytes([0x03]) + second_conf,
            _close_frame(),
        )

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN, shell_id="s") as shell:
                frames = [frame async for frame in shell]

        assert shell.bytes_dropped == 0
        assert frames == []  # second confirmation swallowed, close frame stops iteration


class TestShellSessionAuthModes:
    """open_shell routes to the correct auth helper based on the auth= argument."""

    @pytest.mark.asyncio
    async def test_sigv4_default_uses_connect_shell(self):
        client = _make_client()
        ws = _make_ws(_metadata_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)) as mock_connect:
            async with ShellSession(client, FAKE_ARN, auth="sigv4") as _:
                pass

        client.connect_shell.assert_called_once()
        client.connect_shell_presigned.assert_not_called()
        client.connect_shell_oauth.assert_not_called()
        # SigV4 path passes additional_headers (websockets ≥13) or extra_headers (≤12)
        _, kwargs = mock_connect.call_args
        assert "additional_headers" in kwargs or "extra_headers" in kwargs

    @pytest.mark.asyncio
    async def test_presigned_uses_connect_shell_presigned(self):
        presigned_url = "wss://bedrock-agentcore.us-west-2.amazonaws.com/runtimes/X/ws/commands?X-Amz-Signature=abc"
        client = MagicMock()
        client.connect_shell_presigned.return_value = presigned_url
        ws = _make_ws(_metadata_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)) as mock_connect:
            async with ShellSession(client, FAKE_ARN, auth=PresignedAuth(expires=120)) as _:
                pass

        client.connect_shell_presigned.assert_called_once()
        client.connect_shell.assert_not_called()
        client.connect_shell_oauth.assert_not_called()
        # Presigned path passes NO headers or subprotocols
        _, kwargs = mock_connect.call_args
        assert "extra_headers" not in kwargs
        assert "additional_headers" not in kwargs
        assert "subprotocols" not in kwargs

    @pytest.mark.asyncio
    async def test_presigned_forwards_expires(self):
        client = MagicMock()
        client.connect_shell_presigned.return_value = FAKE_URL
        ws = _make_ws(_metadata_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN, auth=PresignedAuth(expires=60)) as _:
                pass

        _, kwargs = client.connect_shell_presigned.call_args
        assert kwargs["expires"] == 60

    @pytest.mark.asyncio
    async def test_oauth_uses_connect_shell_oauth(self):
        client = MagicMock()
        import base64

        encoded = base64.urlsafe_b64encode(b"tok").decode().rstrip("=")
        expected_protos = [f"base64UrlBearerAuthorization.{encoded}", "base64UrlBearerAuthorization"]
        client.connect_shell_oauth.return_value = (FAKE_URL, expected_protos)
        ws = _make_ws(_metadata_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)) as mock_connect:
            async with ShellSession(client, FAKE_ARN, auth=OAuthAuth(bearer_token="tok")) as _:
                pass

        client.connect_shell_oauth.assert_called_once()
        client.connect_shell.assert_not_called()
        client.connect_shell_presigned.assert_not_called()
        # OAuth path passes subprotocols
        _, kwargs = mock_connect.call_args
        assert kwargs.get("subprotocols") == expected_protos

    @pytest.mark.asyncio
    async def test_oauth_forwards_bearer_token(self):
        client = MagicMock()
        import base64

        encoded = base64.urlsafe_b64encode(b"my-token").decode().rstrip("=")
        protos = [f"base64UrlBearerAuthorization.{encoded}", "base64UrlBearerAuthorization"]
        client.connect_shell_oauth.return_value = (FAKE_URL, protos)
        ws = _make_ws(_metadata_frame())

        with patch("websockets.connect", new=AsyncMock(return_value=ws)):
            async with ShellSession(client, FAKE_ARN, auth=OAuthAuth(bearer_token="my-token")) as _:
                pass

        _, kwargs = client.connect_shell_oauth.call_args
        assert kwargs["bearer_token"] == "my-token"


class TestReconnectConfig:
    def test_defaults(self):
        config = ReconnectConfig()
        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 15.0
        assert config.on_reconnect is None

    def test_unlimited_retries_with_none_window(self):
        config = ReconnectConfig(reconnect_window=None)
        assert config.reconnect_window is None
