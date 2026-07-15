"""E2E test configuration and shared fixtures.

Prerequisites:
    ada credentials update --account=442782095125 --provider=isengard --role=Admin --once

Evaluator IDs must be set as environment variables or configured below.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import pytest

REGION = os.environ.get("E2E_REGION", "us-west-2")
ENDPOINT_URL = os.environ.get("E2E_ENDPOINT_URL", "https://bedrock-agentcore.us-west-2.amazonaws.com")
FIXTURES_DIR = Path(__file__).parent / "fixtures"

DEPLOYED_STATE_PATH = Path("/Users/haomiao/Desktop/AWS/E2eEvalProject/agentcore/.cli/deployed-state.json")


def _load_evaluator_ids() -> Dict[str, str]:
    """Load evaluator IDs from deployed-state.json or environment variables."""
    ids = {}
    if DEPLOYED_STATE_PATH.exists():
        with open(DEPLOYED_STATE_PATH) as f:
            state = json.load(f)
        evaluators = state.get("targets", {}).get("default", {}).get("resources", {}).get("evaluators", {})
        for name, info in evaluators.items():
            ids[name] = info.get("evaluatorId", "")
    return ids


_DEPLOYED_IDS = _load_evaluator_ids()

EVALUATOR_IDS = {
    "deepeval-answer-relevancy": os.environ.get("EVALUATOR_ID_DEEPEVAL_ANSWER_RELEVANCY", _DEPLOYED_IDS.get("answer_relevancy", "")),
    "deepeval-faithfulness": os.environ.get("EVALUATOR_ID_DEEPEVAL_FAITHFULNESS", _DEPLOYED_IDS.get("faithfulness", "")),
    "deepeval-hallucination": os.environ.get("EVALUATOR_ID_DEEPEVAL_HALLUCINATION", _DEPLOYED_IDS.get("hallucination", "")),
    "deepeval-tool-correctness": os.environ.get("EVALUATOR_ID_DEEPEVAL_TOOL_CORRECTNESS", _DEPLOYED_IDS.get("tool_correctness", "")),
    "deepeval-argument-correctness": os.environ.get("EVALUATOR_ID_DEEPEVAL_ARGUMENT_CORRECTNESS", _DEPLOYED_IDS.get("argument_correctness", "")),
    "deepeval-geval": os.environ.get("EVALUATOR_ID_DEEPEVAL_GEVAL", _DEPLOYED_IDS.get("geval_helpfulness", "")),
    "deepeval-contextual-precision": os.environ.get("EVALUATOR_ID_DEEPEVAL_CONTEXTUAL_PRECISION", _DEPLOYED_IDS.get("contextual_precision", "")),
    "deepeval-json-correctness": os.environ.get("EVALUATOR_ID_DEEPEVAL_JSON_CORRECTNESS", _DEPLOYED_IDS.get("json_correctness", "")),
    "deepeval-bias": os.environ.get("EVALUATOR_ID_DEEPEVAL_BIAS", _DEPLOYED_IDS.get("bias", "")),
    "deepeval-toxicity": os.environ.get("EVALUATOR_ID_DEEPEVAL_TOXICITY", _DEPLOYED_IDS.get("toxicity", "")),
    "autoevals-factuality": os.environ.get("EVALUATOR_ID_AUTOEVALS_FACTUALITY", _DEPLOYED_IDS.get("factuality", "")),
    "autoevals-closedqa": os.environ.get("EVALUATOR_ID_AUTOEVALS_CLOSEDQA", _DEPLOYED_IDS.get("closed_qa", "")),
    "autoevals-answer-correctness": os.environ.get("EVALUATOR_ID_AUTOEVALS_ANSWER_CORRECTNESS", _DEPLOYED_IDS.get("autoevals_relevancy", "")),
    "autoevals-exact-match": os.environ.get("EVALUATOR_ID_AUTOEVALS_EXACT_MATCH", _DEPLOYED_IDS.get("exact_match", "")),
    "deepeval-custom-flat": os.environ.get("EVALUATOR_ID_DEEPEVAL_CUSTOM_FLAT", _DEPLOYED_IDS.get("custom_flat", "")),
    "deepeval-custom-nested": os.environ.get("EVALUATOR_ID_DEEPEVAL_CUSTOM_NESTED", _DEPLOYED_IDS.get("custom_nested", "")),
    "autoevals-custom": os.environ.get("EVALUATOR_ID_AUTOEVALS_CUSTOM", _DEPLOYED_IDS.get("custom_autoevals", "")),
}


@pytest.fixture(scope="session")
def dp_client():
    """AgentCore evaluation dataplane client."""
    return boto3.client(
        "agentcore-evaluation-dataplane",
        region_name=REGION,
        endpoint_url=ENDPOINT_URL,
    )


def load_fixture(name: str) -> Dict[str, Any]:
    """Load a span fixture file by name.

    Returns the raw fixture as-is — the service handles span merging internally.
    """
    path = FIXTURES_DIR / name
    with open(path) as f:
        return json.load(f)


def _merge_span_entries(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge CloudWatch ADOT split format: span metadata + log events.

    In CloudWatch format, each span produces two log entries:
    - Span entry: has 'kind', 'startTimeUnixNano', 'endTimeUnixNano', 'status'
    - Log event entry: has 'body', 'severityNumber', 'timeUnixNano'

    The service merges them by spanId, attaching log event bodies as span_events.
    """
    from collections import defaultdict

    span_entries: Dict[str, Dict[str, Any]] = {}
    log_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry in spans:
        span_id = entry.get("spanId", "")
        if not span_id:
            continue

        # Distinguish: span entries have 'kind' or 'endTimeUnixNano', log events have 'body'
        if "body" in entry and "endTimeUnixNano" not in entry:
            log_entries[span_id].append(entry)
        else:
            span_entries[span_id] = entry

    # Merge: attach log event bodies as span_events on the span entry
    merged = []
    for span_id, span in span_entries.items():
        if span_id in log_entries:
            span["span_events"] = [
                {"body": log.get("body"), "event_name": log.get("scope", {}).get("name", "")}
                for log in log_entries[span_id]
            ]
        merged.append(span)

    return merged if merged else spans


def get_fixture_context(name: str) -> Dict[str, Any]:
    """Get the spanContext (sessionId + traceId) from a fixture for reference_inputs."""
    data = load_fixture(name)
    spans = data["sessionSpans"]
    session_id = "default_session"
    trace_id = ""
    for s in spans:
        if not trace_id:
            trace_id = s.get("traceId", "")
        sid = s.get("attributes", {}).get("session.id")
        if sid:
            session_id = sid
    return {"spanContext": {"sessionId": session_id, "traceId": trace_id}}


def build_reference_input(fixture_name: str, **kwargs) -> List[Dict[str, Any]]:
    """Build evaluationReferenceInputs with required context field.

    Args:
        fixture_name: Fixture to extract context from.
        **kwargs: Additional fields (expectedResponse, expectedTrajectory, assertions).

    Returns:
        List with one reference input dict.
    """
    ref = {"context": get_fixture_context(fixture_name)}
    ref.update(kwargs)
    return [ref]


def run_evaluation(
    dp_client,
    evaluator_id: str,
    fixture_name: str,
    reference_inputs: Optional[List[Dict[str, Any]]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Run an on-demand evaluation and return the first result.

    Args:
        dp_client: Boto3 dataplane client.
        evaluator_id: Registered evaluator ID.
        fixture_name: Span fixture filename in fixtures/.
        reference_inputs: Optional ground-truth reference inputs.
        timeout: Max wait time in seconds.

    Returns:
        The first evaluationResults entry from the response.

    Raises:
        AssertionError: If evaluation fails or times out.
    """
    assert evaluator_id, f"Evaluator ID not configured — set the environment variable"

    evaluation_input = load_fixture(fixture_name)

    kwargs = {
        "evaluatorId": evaluator_id,
        "evaluationInput": evaluation_input,
    }
    if reference_inputs:
        kwargs["evaluationReferenceInputs"] = reference_inputs

    start = time.time()
    response = dp_client.evaluate(**kwargs)
    elapsed = time.time() - start

    assert elapsed < timeout, f"Evaluation took {elapsed:.1f}s, exceeded {timeout}s timeout"

    response.pop("ResponseMetadata", None)
    results = response.get("evaluationResults", [])
    assert len(results) > 0, "No evaluation results returned"

    return results[0]


def assert_success(result: Dict[str, Any], expect_explanation: bool = True):
    """Assert a successful evaluation result.

    Args:
        result: Single evaluation result dict.
        expect_explanation: Whether to assert non-empty explanation (False for non-LLM metrics).
    """
    assert "errorCode" not in result, (
        f"Evaluation returned error: {result.get('errorCode')}: {result.get('errorMessage')}"
    )
    assert result.get("label") != "Error", (
        f"Evaluation returned Error label (service stripped error details). Result: {result}"
    )
    assert result.get("value") is not None, "Missing score value"
    assert 0.0 <= result["value"] <= 1.0, f"Score {result['value']} not in [0, 1]"
    assert result.get("label") in ("Pass", "Fail"), f"Unexpected label: {result.get('label')}"
    if expect_explanation:
        assert result.get("explanation"), "Missing explanation from LLM judge"
