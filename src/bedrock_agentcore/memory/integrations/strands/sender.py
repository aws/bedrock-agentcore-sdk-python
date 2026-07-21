"""AgentCore event sender used by the Strands long-term memory store."""

from __future__ import annotations

import asyncio
import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from strands.memory import AggregateMemoryError
from strands.types.content import Message

from ._format import extract_text, is_user_or_assistant_with_text, map_role
from .types import (
    DEFAULT_MAX_TURNS_PER_EVENT,
    AgentCoreDataPlaneClient,
    ExtractionMode,
    MetadataProvider,
    MetadataValue,
)

# ECMAScript \s: TAB-LF-VT-FF-CR, SPACE, NBSP, BOM, Unicode Space_Separator, and LS/PS.
_ECMASCRIPT_WHITESPACE = "\u0009-\u000d\u0020\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_METADATA_VALUE_PATTERN = re.compile(rf"^[a-zA-Z0-9{_ECMASCRIPT_WHITESPACE}._:/=+@-]*$")


@dataclass
class _SeqMessage:
    message: Message
    sequence_number: int | None


@dataclass
class _EventGroup:
    items: list[_SeqMessage]
    metadata: dict[str, dict[str, str]] | None


class AgentCoreEventSender:
    """Pack role-tagged Strands messages into AgentCore ``create_event`` calls."""

    def __init__(
        self,
        *,
        client: AgentCoreDataPlaneClient,
        memory_id: str,
        actor_id: str,
        session_id: str,
        metadata_provider: MetadataProvider | None = None,
        run_id: str | None = None,
        max_turns_per_event: int = DEFAULT_MAX_TURNS_PER_EVENT,
        extraction_mode: ExtractionMode | None = None,
    ) -> None:
        """Initialize the sender.

        Args:
            client: Boto3 AgentCore data-plane client.
            memory_id: AgentCore Memory resource identifier.
            actor_id: Actor identifier.
            session_id: Session identifier.
            metadata_provider: Optional per-message event metadata callback.
            run_id: Run-unique idempotency-token component. Defaults to a UUID.
            max_turns_per_event: Maximum role-tagged turns in one event.
            extraction_mode: Optional AgentCore long-term extraction mode.

        Raises:
            ValueError: If ``max_turns_per_event`` is not a positive integer.
        """
        if type(max_turns_per_event) is not int or max_turns_per_event < 1:
            raise ValueError(
                f"AgentCoreEventSender: max_turns_per_event must be a positive integer, got {max_turns_per_event}"
            )
        self._client = client
        self._memory_id = memory_id
        self._actor_id = actor_id
        self._session_id = session_id
        self._metadata_provider = metadata_provider
        self._run_id = run_id if run_id is not None else str(uuid.uuid4())
        self._max_turns_per_event = max_turns_per_event
        self._extraction_mode = extraction_mode

    async def send_batch(self, messages: list[Message], sequence_numbers: list[int] | None = None) -> None:
        """Send all eligible messages, attempting every prepared event concurrently.

        The complete all-event operation is shielded from caller cancellation. If
        cancellation arrives, this method waits for every in-flight boto3 call to
        settle. A failed write then wins as ``AggregateMemoryError`` so Strands can
        roll back the batch; otherwise the original cancellation propagates.

        Args:
            messages: Strands messages to write.
            sequence_numbers: Optional index-aligned message sequence numbers.

        Raises:
            AggregateMemoryError: If one or more AgentCore calls fail.
            ValueError: If metadata cannot be represented by AgentCore.
        """
        sendable = [
            _SeqMessage(
                message,
                sequence_numbers[index] if sequence_numbers and index < len(sequence_numbers) else None,
            )
            for index, message in enumerate(messages)
            if is_user_or_assistant_with_text(message)
        ]
        if not sendable:
            return

        # Validate every metadata bag before scheduling any network call. Grouping
        # follows JSON.stringify(rawMetadata); wire values are normalized separately.
        events = self._group_into_events(sendable)
        operation = asyncio.create_task(self._send_all(events))
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            # ``asyncio.to_thread`` cannot stop its worker. Keep cancellation
            # attached to this coroutine until every write has a known outcome.
            while not operation.done():
                try:
                    await asyncio.shield(operation)
                except asyncio.CancelledError:
                    continue
            # A write failure must reach the coordinator as Exception so its
            # high-water mark is rolled back instead of trimming pending data.
            operation.result()
            raise

    async def _send_all(self, events: list[_EventGroup]) -> None:
        results = await asyncio.gather(*(self._send_event(event) for event in events), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            first = str(failures[0])
            raise AggregateMemoryError(
                f"AgentCore create_event failed for {len(failures)} of {len(events)} event(s); first error: {first}",
                failures,
            )

    def _group_into_events(self, sendable: list[_SeqMessage]) -> list[_EventGroup]:
        groups: list[_EventGroup] = []
        current: _EventGroup | None = None
        current_signature: str | None = None
        for item in sendable:
            raw_metadata = dict(self._metadata_provider(item.message)) if self._metadata_provider else None
            signature = _ecmascript_json_stringify_metadata(raw_metadata) if raw_metadata is not None else ""
            metadata = _to_agentcore_metadata(raw_metadata) if raw_metadata else None
            at_cap = current is not None and len(current.items) >= self._max_turns_per_event
            if current is None or signature != current_signature or at_cap:
                current = _EventGroup(items=[], metadata=metadata)
                groups.append(current)
                current_signature = signature
            current.items.append(item)
        return groups

    async def _send_event(self, event: _EventGroup) -> None:
        payload = [
            {
                "conversational": {
                    "role": map_role(item.message),
                    "content": {"text": extract_text(item.message)},
                }
            }
            for item in event.items
        ]
        kwargs: dict[str, object] = {
            "memoryId": self._memory_id,
            "actorId": self._actor_id,
            "sessionId": self._session_id,
            "eventTimestamp": datetime.now(timezone.utc),
            "payload": payload,
        }
        token = self._token_for_sequence_numbers([item.sequence_number for item in event.items])
        if token is not None:
            kwargs["clientToken"] = token
        if event.metadata:
            kwargs["metadata"] = event.metadata
        if self._extraction_mode is not None:
            kwargs["extractionMode"] = self._extraction_mode
        await asyncio.to_thread(self._client.create_event, **kwargs)

    def _token_for_sequence_numbers(self, sequence_numbers: list[int | None]) -> str | None:
        if not sequence_numbers or any(number is None for number in sequence_numbers):
            return None
        first = sequence_numbers[0]
        last = sequence_numbers[-1]
        return f"{self._memory_id}-{self._actor_id}-{self._run_id}-{first}-{last}"


def _ecmascript_number_string(value: int | float) -> str:
    """Return the scalar text produced by ECMAScript ``JSON.stringify``.

    Python and ECMAScript use different fixed/exponent thresholds even though
    both choose shortest round-trippable binary64 digits. Normalize Python's
    shortest representation to ECMAScript's notation rules.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("number is not finite")
    if number == 0:
        return "0"

    shortest = repr(number).lower()
    absolute = abs(number)
    if 1e-6 <= absolute < 1e21:
        fixed = format(Decimal(shortest), "f")
        if "." in fixed:
            fixed = fixed.rstrip("0").rstrip(".")
        return fixed

    if "e" not in shortest:
        shortest = format(number, ".15e")
        coefficient, exponent = shortest.split("e")
        coefficient = coefficient.rstrip("0").rstrip(".")
    else:
        coefficient, exponent = shortest.split("e")
    sign = "+" if int(exponent) >= 0 else "-"
    return f"{coefficient}e{sign}{abs(int(exponent))}"


def _is_ecmascript_array_index(key: str) -> bool:
    """Return whether ``key`` is an ECMAScript array-index property name."""
    if not key or len(key) > 10 or not key.isascii() or not key.isdecimal():
        return False
    if key != "0" and key.startswith("0"):
        return False
    return int(key) <= 2**32 - 2


def _ecmascript_property_order(metadata: dict[str, MetadataValue]) -> list[tuple[str, MetadataValue]]:
    """Order own string keys as ``JSON.stringify``/``Object.keys`` does."""
    index_keys: list[tuple[int, str, MetadataValue]] = []
    other_keys: list[tuple[str, MetadataValue]] = []
    for key, value in metadata.items():
        if _is_ecmascript_array_index(key):
            index_keys.append((int(key), key, value))
        else:
            other_keys.append((key, value))
    index_keys.sort(key=lambda item: item[0])
    return [(key, value) for _, key, value in index_keys] + other_keys


def _ecmascript_json_stringify_metadata(metadata: dict[str, MetadataValue]) -> str:
    """Serialize supported raw metadata like ECMAScript ``JSON.stringify``."""
    entries: list[str] = []
    for key, value in _ecmascript_property_order(metadata):
        encoded_key = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
        if isinstance(value, str):
            encoded_value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        elif isinstance(value, bool):
            encoded_value = "true" if value else "false"
        else:
            # Reuse wire validation so unsupported/non-finite values keep the
            # public error contract while finite numbers use the JS formatter.
            encoded_value = _metadata_scalar_string(key, value)
        entries.append(f"{encoded_key}:{encoded_value}")
    return "{" + ",".join(entries) + "}"


def _metadata_scalar_string(key: str, value: object) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(
            f'AgentCoreEventSender: metadata value for key "{key}" is {value}, which has no valid string '
            "representation. Provide a finite number, boolean, or a string (omit the key instead of passing "
            "None)."
        )
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if type(value) is int or type(value) is float:
        try:
            return _ecmascript_number_string(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(
                f'AgentCoreEventSender: metadata value for key "{key}" is {value}, which has no valid string '
                "representation. Provide a finite number, boolean, or a string."
            ) from error
    raise ValueError(
        f'AgentCoreEventSender: metadata value for key "{key}" must be a scalar string, finite number, or boolean; '
        f"got {type(value).__name__}. Arrays, objects, and None are not supported."
    )


def _to_agentcore_metadata(metadata: dict[str, MetadataValue]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for key, value in metadata.items():
        string_value = _metadata_scalar_string(key, value)
        if not _METADATA_VALUE_PATTERN.fullmatch(string_value):
            raise ValueError(
                f'AgentCoreEventSender: metadata value for key "{key}" contains characters AgentCore rejects '
                "(allowed: letters, digits, whitespace, and ._:/=+@-). "
                f"Got {string_value!r}. Pass a pre-encoded scalar string using only the allowed characters."
            )
        output[key] = {"stringValue": string_value}
    return output
