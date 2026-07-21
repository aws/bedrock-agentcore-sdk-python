"""Tests for the AgentCore event sender."""

import asyncio
import re
import threading
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest
from strands.memory import AggregateMemoryError
from strands.types.content import Message

from bedrock_agentcore.memory.integrations.strands.sender import (
    AgentCoreEventSender,
    _ecmascript_json_stringify_metadata,
)
from bedrock_agentcore.memory.integrations.strands.types import MetadataProvider, MetadataValue


def user_message(text: str) -> Message:
    """Build a user text message."""
    return {"role": "user", "content": [{"text": text}]}


def assistant_message(text: str) -> Message:
    """Build an assistant text message."""
    return {"role": "assistant", "content": [{"text": text}]}


TOOL_ONLY: Message = {
    "role": "user",
    "content": [{"toolUse": {"toolUseId": "t1", "name": "noop", "input": {}}}],
}


def make_sender(
    client: Mock,
    *,
    max_turns_per_event: int = 50,
    run_id: str | None = "run-1",
    metadata_provider: MetadataProvider | None = None,
    extraction_mode: str | None = None,
) -> AgentCoreEventSender:
    """Build a sender with deterministic identity."""
    return AgentCoreEventSender(
        client=client,
        memory_id="mem-1",
        actor_id="actor-1",
        session_id="sess-1",
        run_id=run_id,
        max_turns_per_event=max_turns_per_event,
        metadata_provider=metadata_provider,
        extraction_mode=extraction_mode,  # type: ignore[arg-type]
    )


def turns(call: Any) -> list[dict[str, str]]:
    """Extract role/text pairs from one mock call."""
    return [
        {
            "role": item["conversational"]["role"],
            "text": item["conversational"]["content"]["text"],
        }
        for item in call.kwargs["payload"]
    ]


async def test_packs_batch_into_one_role_tagged_event() -> None:
    """A whole flush becomes one event when under the cap."""
    client = Mock()
    client.create_event.return_value = {}
    await make_sender(client).send_batch([user_message("hello"), assistant_message("hi there"), user_message("again")])
    client.create_event.assert_called_once()
    assert client.create_event.call_args.kwargs["memoryId"] == "mem-1"
    assert turns(client.create_event.call_args) == [
        {"role": "USER", "text": "hello"},
        {"role": "ASSISTANT", "text": "hi there"},
        {"role": "USER", "text": "again"},
    ]


async def test_chunks_batch_at_max_turns() -> None:
    """Split a batch into ceil(n / cap) events."""
    client = Mock()
    client.create_event.return_value = {}
    await make_sender(client, max_turns_per_event=2).send_batch([user_message(value) for value in "abcde"])
    assert client.create_event.call_count == 3
    assert [[turn["text"] for turn in turns(call)] for call in client.create_event.call_args_list] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


async def test_skips_tool_only_empty_and_all_unsendable_batches() -> None:
    """Omit messages without extractable user/assistant text."""
    client = Mock()
    client.create_event.return_value = {}
    sender = make_sender(client)
    await sender.send_batch([TOOL_ONLY, user_message("real"), assistant_message("   ")])
    assert turns(client.create_event.call_args) == [{"role": "USER", "text": "real"}]
    client.reset_mock()
    await sender.send_batch([TOOL_ONLY])
    client.create_event.assert_not_called()


async def test_message_text_uses_ecmascript_trim_boundaries_on_wire() -> None:
    """Send U+0085 text unchanged and drop a BOM-only message as blank."""
    client = Mock()
    client.create_event.return_value = {}
    await make_sender(client).send_batch([user_message(" \u0085 "), user_message("\ufeff")])
    client.create_event.assert_called_once()
    assert turns(client.create_event.call_args) == [{"role": "USER", "text": "\u0085"}]


async def test_omits_client_token_without_complete_sequence_numbers() -> None:
    """A token is unsafe unless every covered message has a sequence number."""
    client = Mock()
    client.create_event.return_value = {}
    sender = make_sender(client)
    await sender.send_batch([user_message("x"), user_message("y")])
    assert "clientToken" not in client.create_event.call_args.kwargs
    client.reset_mock()
    await sender.send_batch([user_message("x"), user_message("y")], [7])
    assert "clientToken" not in client.create_event.call_args.kwargs


async def test_sequence_range_token_is_stable_and_chunk_specific() -> None:
    """Re-fires reuse a run-scoped deterministic range token."""
    client = Mock()
    client.create_event.return_value = {}
    sender = make_sender(client, max_turns_per_event=2)
    batch = [user_message("a"), user_message("b"), user_message("c")]
    await sender.send_batch(batch, [1, 2, 3])
    assert [call.kwargs["clientToken"] for call in client.create_event.call_args_list] == [
        "mem-1-actor-1-run-1-1-2",
        "mem-1-actor-1-run-1-3-3",
    ]
    client.reset_mock()
    await sender.send_batch(batch, [1, 2, 3])
    assert client.create_event.call_args_list[0].kwargs["clientToken"] == "mem-1-actor-1-run-1-1-2"


async def test_explicit_empty_run_id_is_preserved() -> None:
    """Use nullish rather than truthy defaulting, matching the source runtime."""
    client = Mock()
    client.create_event.return_value = {}
    await make_sender(client, run_id="").send_batch([user_message("x")], [0])
    assert client.create_event.call_args.kwargs["clientToken"] == "mem-1-actor-1--0-0"


async def test_run_id_distinguishes_sequence_resets_and_defaults_to_uuid() -> None:
    """Two runs cannot collide when sequence numbers restart at zero."""
    client = Mock()
    client.create_event.return_value = {}
    await make_sender(client, run_id="run-A").send_batch([user_message("x")], [0])
    await make_sender(client, run_id="run-B").send_batch([user_message("x")], [0])
    assert [call.kwargs["clientToken"] for call in client.create_event.call_args_list] == [
        "mem-1-actor-1-run-A-0-0",
        "mem-1-actor-1-run-B-0-0",
    ]
    client.reset_mock()
    await make_sender(client, run_id=None).send_batch([user_message("x")], [0])
    token = client.create_event.call_args.kwargs["clientToken"]
    assert re.fullmatch(r"mem-1-actor-1-[0-9a-f-]{36}-0-0", token)
    assert "sess-1" not in token


async def test_metadata_changes_split_only_consecutive_runs() -> None:
    """Metadata is per-event, so A,A,B,C,B forms four events."""
    client = Mock()
    client.create_event.return_value = {}

    def provider(message: Message) -> dict[str, MetadataValue]:
        return {"topic": message["content"][0]["text"]}

    await make_sender(client, metadata_provider=provider).send_batch(
        [user_message(value) for value in ["A", "A", "B", "C", "B"]]
    )
    assert client.create_event.call_count == 4
    assert [[turn["text"] for turn in turns(call)] for call in client.create_event.call_args_list] == [
        ["A", "A"],
        ["B"],
        ["C"],
        ["B"],
    ]
    assert client.create_event.call_args_list[0].kwargs["metadata"] == {"topic": {"stringValue": "A"}}


async def test_constant_metadata_is_mapped_and_empty_bag_omitted() -> None:
    """Map scalar metadata to the boto3 wire shape."""
    client = Mock()
    client.create_event.return_value = {}
    await make_sender(client, metadata_provider=lambda _message: {"source": "support", "priority": 3}).send_batch(
        [user_message("x"), user_message("y")]
    )
    assert client.create_event.call_count == 1
    assert client.create_event.call_args.kwargs["metadata"] == {
        "source": {"stringValue": "support"},
        "priority": {"stringValue": "3"},
    }
    client.reset_mock()
    await make_sender(client, metadata_provider=lambda _message: {}).send_batch([user_message("x")])
    assert "metadata" not in client.create_event.call_args.kwargs


@pytest.mark.parametrize(
    "metadata",
    [
        {"note": "billing,refund"},
        {"q": "why?"},
    ],
)
async def test_rejects_disallowed_metadata_before_network(
    metadata: dict[str, Any],
) -> None:
    """Surface AgentCore's metadata charset restriction locally."""
    client = Mock()
    with pytest.raises(ValueError, match="characters AgentCore rejects"):
        await make_sender(client, metadata_provider=lambda _message: metadata).send_batch([user_message("x")])
    client.create_event.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [
        "tab\tline\nvertical\vform\ffeed\rspace ",
        "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a",
        "\u2028\u2029\u202f\u205f\u3000\ufeff",
    ],
)
async def test_accepts_ecmascript_whitespace_metadata(value: str) -> None:
    """Accept every ECMAScript whitespace category, including BOM."""
    client = Mock()
    client.create_event.return_value = {}
    await make_sender(client, metadata_provider=lambda _message: {"space": value}).send_batch([user_message("x")])
    assert client.create_event.call_args.kwargs["metadata"] == {"space": {"stringValue": value}}


async def test_rejects_python_only_next_line_whitespace_metadata() -> None:
    r"""U+0085 is Python ``\s`` but not ECMAScript ``\s``."""
    client = Mock()
    with pytest.raises(ValueError, match="characters AgentCore rejects"):
        await make_sender(client, metadata_provider=lambda _message: {"space": "\u0085"}).send_batch(
            [user_message("x")]
        )
    client.create_event.assert_not_called()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
async def test_rejects_nullish_or_non_finite_metadata_before_network(value: object) -> None:
    """Reject values that have no safe scalar representation."""
    client = Mock()

    def provider(_message: Message) -> dict[str, MetadataValue]:
        return {"bad": value}  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="no valid string representation"):
        await make_sender(client, metadata_provider=provider).send_batch([user_message("x")])
    client.create_event.assert_not_called()


async def test_aggregates_failures_after_attempting_every_event() -> None:
    """Use all-settled behavior and preserve every failed event reason."""
    attempted: list[str] = []

    def create_event(**kwargs: Any) -> dict[str, Any]:
        text = kwargs["payload"][0]["conversational"]["content"]["text"]
        attempted.append(text)
        if text.startswith("bad"):
            raise RuntimeError(f"nope: {text}")
        return {}

    client = Mock()
    client.create_event.side_effect = create_event
    with pytest.raises(AggregateMemoryError, match="2 of 3.*first error: nope:") as raised:
        await make_sender(client, max_turns_per_event=1).send_batch(
            [user_message("good-1"), user_message("bad-1"), user_message("bad-2")]
        )
    assert len(raised.value.errors) == 2
    assert sorted(attempted) == ["bad-1", "bad-2", "good-1"]
    assert client.create_event.call_count == 3


async def test_sender_has_no_retry_layer() -> None:
    """One event failure produces one network attempt."""
    client = Mock()
    client.create_event.side_effect = RuntimeError("throttled by AgentCore")
    with pytest.raises(AggregateMemoryError, match="first error: throttled by AgentCore"):
        await make_sender(client).send_batch([user_message("x")])
    client.create_event.assert_called_once()


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_rejects_invalid_max_turns(value: object) -> None:
    """The event cap must be a positive integer."""
    with pytest.raises(ValueError, match="positive integer"):
        AgentCoreEventSender(
            client=Mock(),
            memory_id="m",
            actor_id="a",
            session_id="s",
            max_turns_per_event=value,  # type: ignore[arg-type]
        )


async def test_extraction_mode_is_optional_wire_passthrough() -> None:
    """Send SKIP exactly when configured."""
    client = Mock()
    client.create_event.return_value = {}
    await make_sender(client, extraction_mode="SKIP").send_batch([user_message("sensitive")])
    assert client.create_event.call_args.kwargs["extractionMode"] == "SKIP"
    client.reset_mock()
    await make_sender(client).send_batch([user_message("normal")])
    assert "extractionMode" not in client.create_event.call_args.kwargs


async def test_blocking_boto_call_runs_off_event_loop() -> None:
    """The synchronous boto3 call is delegated through ``asyncio.to_thread``."""
    client = Mock()
    client.create_event.return_value = {}
    loop_thread_seen: list[bool] = []
    original = asyncio.to_thread

    async def tracked(function: Callable[..., object], /, *args: object, **kwargs: object) -> object:
        loop_thread_seen.append(True)
        return await original(function, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(asyncio, "to_thread", tracked)
        await make_sender(client).send_batch([user_message("x")])
    assert loop_thread_seen == [True]


@pytest.mark.parametrize("value", [101, 1000])
def test_accepts_event_caps_above_python_service_assumption(value: int) -> None:
    """Match the source, which only requires a positive integer."""
    assert make_sender(Mock(), max_turns_per_event=value)._max_turns_per_event == value


async def test_oversized_turn_is_forwarded_without_python_only_validation() -> None:
    """Leave service payload validation to AgentCore, matching the source."""
    client = Mock()
    client.create_event.return_value = {}
    await make_sender(client).send_batch([user_message("x" * 100_001)])
    assert len(client.create_event.call_args.kwargs["payload"][0]["conversational"]["content"]["text"]) == 100_001


async def test_metadata_does_not_enforce_python_only_key_count_or_length_limits() -> None:
    """Only metadata values receive the source's local validation."""
    client = Mock()
    client.create_event.return_value = {}
    metadata = {f"key-{index}": "v" for index in range(16)}
    metadata["k" * 129] = "v" * 257
    await make_sender(client, metadata_provider=lambda _message: metadata).send_batch([user_message("x")])
    wire = client.create_event.call_args.kwargs["metadata"]
    assert len(wire) == 17
    assert wire["k" * 129] == {"stringValue": "v" * 257}


@pytest.mark.parametrize("value", [None, ["a"], {"nested": "value"}])
async def test_rejects_dynamic_non_scalar_metadata(value: object) -> None:
    """Dynamically supplied null, arrays, and objects fail with a scalar-only message."""
    client = Mock()

    def provider(_message: Message) -> dict[str, MetadataValue]:
        return {"bad": value}  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="scalar|valid string representation"):
        await make_sender(client, metadata_provider=provider).send_batch([user_message("x")])
    client.create_event.assert_not_called()


async def test_ecmascript_scalar_normalization_drives_grouping_and_wire_metadata() -> None:
    """Equivalent JS scalars share event groups and use JSON.stringify-compatible text."""
    client = Mock()
    client.create_event.return_value = {}
    values: list[int | float | bool] = [
        3,
        3.0,
        -0.0,
        0,
        True,
        1e-7,
        1e20,
        1e21,
        1e-6,
    ]
    by_text = dict(zip([str(index) for index in range(len(values))], values, strict=True))

    def provider(message: Message) -> dict[str, MetadataValue]:
        return {"value": by_text[message["content"][0]["text"]]}

    await make_sender(client, metadata_provider=provider).send_batch(
        [user_message(str(index)) for index in range(len(values))]
    )
    assert client.create_event.call_count == 7
    calls = client.create_event.call_args_list
    assert [turn["text"] for turn in turns(calls[0])] == ["0", "1"]
    assert calls[0].kwargs["metadata"] == {"value": {"stringValue": "3"}}
    assert [turn["text"] for turn in turns(calls[1])] == ["2", "3"]
    assert calls[1].kwargs["metadata"] == {"value": {"stringValue": "0"}}
    assert [call.kwargs["metadata"]["value"]["stringValue"] for call in calls[2:]] == [
        "true",
        "1e-7",
        "100000000000000000000",
        "1e+21",
        "0.000001",
    ]


async def test_raw_string_and_number_metadata_create_separate_groups() -> None:
    """JSON.stringify distinguishes raw string and numeric values before wire mapping."""
    client = Mock()
    client.create_event.return_value = {}

    def provider(message: Message) -> dict[str, MetadataValue]:
        text = message["content"][0]["text"]
        return {"v": "3" if text == "string" else 3}

    await make_sender(client, metadata_provider=provider).send_batch([user_message("string"), user_message("number")])
    assert client.create_event.call_count == 2
    assert [call.kwargs["metadata"] for call in client.create_event.call_args_list] == [
        {"v": {"stringValue": "3"}},
        {"v": {"stringValue": "3"}},
    ]


async def test_raw_string_and_boolean_metadata_create_separate_groups() -> None:
    """JSON.stringify distinguishes a raw boolean from its wire-equivalent string."""
    client = Mock()
    client.create_event.return_value = {}

    def provider(message: Message) -> dict[str, MetadataValue]:
        text = message["content"][0]["text"]
        return {"v": "true" if text == "string" else True}

    await make_sender(client, metadata_provider=provider).send_batch([user_message("string"), user_message("boolean")])
    assert client.create_event.call_count == 2
    assert [call.kwargs["metadata"] for call in client.create_event.call_args_list] == [
        {"v": {"stringValue": "true"}},
        {"v": {"stringValue": "true"}},
    ]


async def test_integer_like_metadata_keys_use_ecmascript_property_order() -> None:
    """Canonical array-index keys sort numerically before insertion-ordered keys."""
    client = Mock()
    client.create_event.return_value = {}

    def provider(message: Message) -> dict[str, MetadataValue]:
        if message["content"][0]["text"] == "first":
            return {"4294967294": "max", "10": "ten", "0": "zero", "2": "two", "ordinary": "value"}
        return {"0": "zero", "2": "two", "10": "ten", "4294967294": "max", "ordinary": "value"}

    await make_sender(client, metadata_provider=provider).send_batch([user_message("first"), user_message("second")])
    client.create_event.assert_called_once()
    assert [turn["text"] for turn in turns(client.create_event.call_args)] == ["first", "second"]


async def test_non_index_metadata_key_order_remains_part_of_raw_signature() -> None:
    """JSON.stringify preserves insertion order for ordinary property names."""
    client = Mock()
    client.create_event.return_value = {}

    def provider(message: Message) -> dict[str, MetadataValue]:
        if message["content"][0]["text"] == "first":
            return {"4294967295": "not-index", "a": "A"}
        return {"a": "A", "4294967295": "not-index"}

    await make_sender(client, metadata_provider=provider).send_batch([user_message("first"), user_message("second")])
    assert client.create_event.call_count == 2


async def test_empty_metadata_bag_has_object_signature_but_no_wire_metadata() -> None:
    """An empty JS object signs as ``{}`` while its wire metadata stays omitted."""
    assert _ecmascript_json_stringify_metadata({}) == "{}"

    client = Mock()
    client.create_event.return_value = {}
    await make_sender(client, metadata_provider=lambda _message: {}).send_batch([user_message("x"), user_message("y")])
    client.create_event.assert_called_once()
    assert "metadata" not in client.create_event.call_args.kwargs


async def test_cancellation_waits_for_delayed_failure_and_raises_aggregate_error() -> None:
    """A cancelled caller cannot detach a failed boto3 write from coordinator rollback."""
    started = threading.Event()
    release = threading.Event()

    def create_event(**_kwargs: Any) -> dict[str, Any]:
        started.set()
        assert release.wait(timeout=2)
        raise RuntimeError("delayed failure")

    client = Mock()
    client.create_event.side_effect = create_event
    task = asyncio.create_task(make_sender(client).send_batch([user_message("x")]))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(AggregateMemoryError, match="delayed failure"):
        await task


async def test_cancellation_propagates_after_successful_write_settles() -> None:
    """Preserve cancellation when all shielded writes eventually succeed."""
    started = threading.Event()
    release = threading.Event()

    def create_event(**_kwargs: Any) -> dict[str, Any]:
        started.set()
        assert release.wait(timeout=2)
        return {}

    client = Mock()
    client.create_event.side_effect = create_event
    task = asyncio.create_task(make_sender(client).send_batch([user_message("x")]))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
