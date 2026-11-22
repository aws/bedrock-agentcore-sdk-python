"""Integration test: multi-turn Strands conversation history preservation.

Ticket V2197498381 reported that the span serializer dropped all but the most
recent user turn and mixed prior assistant turns into the current-turn output.
This test runs a real multi-turn Strands agent, captures the emitted OTel
spans, and asserts that `convert_strands_to_adot` serializes the full
conversation in chronological order.

Run with:
    AWS_PROFILE=<profile> BEDROCK_TEST_REGION=us-west-2 \
        uv run pytest tests_integ/evaluation/integrations/strands_agents_evals/\
test_multi_turn_conversation_serialization.py -xvs --log-cli-level=INFO
"""

import logging
import os

import pytest
from strands import Agent
from strands_evals.telemetry import StrandsEvalsTelemetry

from bedrock_agentcore.evaluation.span_to_adot_serializer import convert_strands_to_adot

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGION = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning:pydantic.main")


@pytest.mark.integration
class TestMultiTurnConversationSerialization:
    """End-to-end validation that multi-turn history survives serialization."""

    def _run_multi_turn_agent(self, turns: list[str]) -> list:
        """Run an agent through several user turns and return the captured spans."""
        telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
        agent = Agent(
            system_prompt=(
                "You are a helpful assistant. Keep responses short — one sentence. "
                "Remember details the user shares across turns."
            ),
        )
        for user_msg in turns:
            response = agent(user_msg)
            logger.info("user=%r -> assistant=%r", user_msg, str(response)[:80])

        return list(telemetry.in_memory_exporter.get_finished_spans())

    @staticmethod
    def _flatten(msg: dict) -> str:
        """Extract the text payload from a serialized message entry."""
        content = msg.get("content", "")
        if isinstance(content, dict):
            for key in ("content", "message"):
                val = content.get(key)
                if isinstance(val, str):
                    return val
            return str(content)
        return str(content)

    def _find_multi_turn_chat_record(self, adot_docs: list, raw_spans: list) -> tuple[dict, int]:
        """Locate the ADOT log record derived from the final multi-turn Strands chat span.

        Strands emits three span kinds per turn; only ``chat`` spans carry the
        full conversation history as span events. This test exercises that span
        specifically. Returns the log record and the number of prior
        assistant-message history events observed on the source span.
        """
        chat_spans = [s for s in raw_spans if s.name == "chat"]
        assert chat_spans, f"Expected at least one Strands 'chat' span, got span names: {[s.name for s in raw_spans]}"
        final_chat_span = chat_spans[-1]
        history_asst_events = [e for e in final_chat_span.events if e.name == "gen_ai.assistant.message"]
        expected_history_count = len(history_asst_events)

        trace_id = format(final_chat_span.context.trace_id, "032x")
        span_id = format(final_chat_span.context.span_id, "016x")
        matching = [
            d
            for d in adot_docs
            if d.get("traceId") == trace_id
            and d.get("spanId") == span_id
            and "body" in d
            and "input" in d.get("body", {})
        ]
        assert matching, f"Could not find log record for final 'chat' span traceId={trace_id} spanId={span_id}"
        return matching[0], expected_history_count

    def test_multi_turn_history_preserved_in_adot_serialization(self):
        """The final-turn chat span must serialize full conversation history."""
        turns = [
            "My name is Taro.",
            "I live in Seattle.",
            "I work on distributed systems.",
            "What is my name and where do I live?",
        ]

        raw_spans = self._run_multi_turn_agent(turns)
        assert raw_spans, "Expected Strands to emit OTel spans"

        adot_docs = convert_strands_to_adot(raw_spans)
        assert adot_docs, "convert_strands_to_adot returned nothing"

        record, expected_history_count = self._find_multi_turn_chat_record(adot_docs, raw_spans)
        input_messages = record["body"]["input"]["messages"]
        output_messages = record["body"]["output"]["messages"]

        user_entries = [m for m in input_messages if m.get("role") == "user"]
        assistant_entries = [m for m in input_messages if m.get("role") == "assistant"]

        user_texts = [self._flatten(m) for m in user_entries]
        assistant_texts = [self._flatten(m) for m in assistant_entries]
        output_texts = [self._flatten(m) for m in output_messages]

        logger.info("final chat span expected prior-assistant history events: %d", expected_history_count)
        logger.info("final-turn input.messages user entries: %d", len(user_entries))
        logger.info("final-turn input.messages assistant entries: %d", len(assistant_entries))
        logger.info("final-turn output.messages entries: %d", len(output_messages))
        for i, t in enumerate(user_texts):
            logger.info("  user[%d]: %r", i, t[:100])
        for i, t in enumerate(assistant_texts):
            logger.info("  history-assistant[%d]: %r", i, t[:100])
        for i, t in enumerate(output_texts):
            logger.info("  output[%d]: %r", i, t[:100])

        # Bug 1 regression: every prior user turn must survive serialization.
        assert len(user_entries) == len(turns), (
            f"Pre-fix bug dropped all but newest user turn. Expected {len(turns)} user entries "
            f"on the final chat span, got {len(user_entries)}. texts={user_texts}"
        )
        joined_users = " | ".join(user_texts)
        for probe in ("Taro", "Seattle", "distributed systems"):
            assert probe in joined_users, (
                f"Expected earlier user turn containing {probe!r} to survive; "
                f"final-turn user entries were: {user_texts}"
            )

        # Bug 2 regression: prior assistant turns must land in input, not output.
        assert len(assistant_entries) == expected_history_count, (
            f"Expected {expected_history_count} prior assistant turns in input.messages "
            f"(matching the gen_ai.assistant.message events on the span); "
            f"got {len(assistant_entries)}"
        )

        # Output on the last turn must be exactly one entry (gen_ai.choice), not
        # N+1 entries from history being mixed in.
        assert len(output_messages) == 1, (
            f"Pre-fix bug mixed history into output.messages. Expected exactly 1 output "
            f"entry on final turn, got {len(output_messages)}: {output_texts}"
        )

        # Output must not contain stale assistant-history content.
        joined_output = " ".join(output_texts).lower()
        assert joined_output, "output.messages was empty"
        for stale in (
            "nice to meet you, taro",
            "got it, taro from seattle",
            "that's interesting work",
        ):
            assert stale not in joined_output, (
                f"output.messages contains stale history entry containing {stale!r}: {output_texts}"
            )

        # Chronological ordering: roles must interleave, not group by role.
        roles_in_order = [m["role"] for m in input_messages]
        # For N user turns with N-1 prior assistant turns, the expected order is
        # user, assistant, user, assistant, ..., user.
        expected_pattern = []
        for i in range(len(turns)):
            expected_pattern.append("user")
            if i < expected_history_count:
                expected_pattern.append("assistant")
        assert roles_in_order == expected_pattern, (
            f"Expected chronological role interleaving {expected_pattern}; got {roles_in_order}. "
            f"Pre-fix role-grouped ordering would have emitted "
            f"[{'user, ' * len(turns)}{'assistant, ' * expected_history_count}] instead."
        )
