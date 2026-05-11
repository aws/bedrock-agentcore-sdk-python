"""Shared helpers for evaluation integration tests."""

import json
import logging
import time
import uuid
from typing import Any, Dict

import boto3

from bedrock_agentcore.evaluation.runner.invoker_types import AgentInvokerFn, AgentInvokerInput, AgentInvokerOutput

logger = logging.getLogger(__name__)

DEFAULT_AGENT_NAME = "sdk_integ_echo_bundled"
DEFAULT_REGION = "us-west-2"
DEFAULT_ROLE_ARN = "arn:aws:iam::619071331382:role/Kyros-Bedrock-AgentCore"
DEFAULT_S3_BUCKET = "codetest-us-west-2-619071331382-do-not-delete"
DEFAULT_S3_KEY = "integ-test-agents/echo-agent-bundled.zip"


def get_or_create_agent_runtime(
    agent_name: str = DEFAULT_AGENT_NAME,
    region: str = DEFAULT_REGION,
    role_arn: str = DEFAULT_ROLE_ARN,
    s3_bucket: str = DEFAULT_S3_BUCKET,
    s3_key: str = DEFAULT_S3_KEY,
) -> Dict[str, str]:
    """Find an existing READY agent runtime by name, or create one.

    The agent is never deleted — it's reused across test runs.

    Returns:
        {"runtime_id": "...", "runtime_arn": "...", "endpoint_arn": "..."}
    """
    cp = boto3.client("bedrock-agentcore-control", region_name=region)

    # Search for existing runtime by name
    runtime = _find_runtime_by_name(cp, agent_name)

    if runtime is None:
        logger.info("No runtime found with name=%s, creating...", agent_name)
        runtime = _create_runtime(cp, agent_name, role_arn, s3_bucket, s3_key)

    runtime_id = runtime["agentRuntimeId"]
    runtime_arn = runtime["agentRuntimeArn"]
    logger.info("Using runtime: %s (%s)", runtime_id, runtime_arn)

    # Find or create endpoint
    endpoint_arn = _get_or_create_endpoint(cp, runtime_id)

    # Warm up the agent to avoid cold start timeouts on first invoke
    _warmup_agent(runtime_arn, region)

    return {
        "runtime_id": runtime_id,
        "runtime_arn": runtime_arn,
        "endpoint_arn": endpoint_arn,
    }


def make_agent_invoker(runtime_arn: str, region: str = DEFAULT_REGION) -> AgentInvokerFn:
    """Create an AgentInvokerFn that calls the deployed agent via invoke_agent_runtime."""
    dp = boto3.client("bedrock-agentcore", region_name=region)

    def invoker(input: AgentInvokerInput) -> AgentInvokerOutput:
        payload = input.payload if isinstance(input.payload, str) else json.dumps(input.payload)
        last_error = None
        for attempt in range(3):
            try:
                response = dp.invoke_agent_runtime(
                    agentRuntimeArn=runtime_arn,
                    payload=payload.encode(),
                    runtimeSessionId=input.session_id if input.session_id and len(input.session_id) >= 33 else str(uuid.uuid4()),
                )
                body = response.get("response", "")
                if hasattr(body, "read"):
                    body = body.read().decode()
                result = json.loads(body) if body else {}
                return AgentInvokerOutput(agent_output=result.get("output", result))
            except Exception as e:
                last_error = e
                if "initialization time exceeded" in str(e).lower() and attempt < 2:
                    import time as _t
                    _t.sleep(5)
                    continue
                raise
        raise last_error  # type: ignore[misc]

    return invoker


# --- Private helpers ---


def _find_runtime_by_name(cp, agent_name: str) -> Any:
    """List runtimes and find one matching the name with READY status."""
    next_token = None
    while True:
        kwargs: Dict[str, Any] = {"maxResults": 100}
        if next_token:
            kwargs["nextToken"] = next_token
        response = cp.list_agent_runtimes(**kwargs)
        for rt in response.get("agentRuntimes", []):
            if rt.get("agentRuntimeName") == agent_name and rt.get("status") == "READY":
                return rt
        next_token = response.get("nextToken")
        if not next_token:
            break
    return None


def _create_runtime(cp, agent_name: str, role_arn: str, s3_bucket: str, s3_key: str) -> Dict[str, Any]:
    """Create a runtime and wait for READY."""
    response = cp.create_agent_runtime(
        agentRuntimeName=agent_name,
        roleArn=role_arn,
        networkConfiguration={"networkMode": "PUBLIC"},
        agentRuntimeArtifact={
            "codeConfiguration": {
                "code": {"s3": {"bucket": s3_bucket, "prefix": s3_key}},
                "runtime": "PYTHON_3_12",
                "entryPoint": ["agent.py"],
            }
        },
    )
    runtime_id = response["agentRuntimeId"]
    logger.info("Created runtime %s, waiting for READY...", runtime_id)

    for _ in range(60):
        rt = cp.get_agent_runtime(agentRuntimeId=runtime_id)
        status = rt["status"]
        if status == "READY":
            return rt
        if "FAILED" in status:
            raise RuntimeError(f"Runtime creation failed: {rt.get('statusReasons')}")
        time.sleep(5)

    raise TimeoutError(f"Runtime {runtime_id} did not reach READY within 300s")


def _warmup_agent(runtime_arn: str, region: str) -> None:
    """Send a warmup invoke to avoid cold start timeouts on first real call."""
    dp = boto3.client("bedrock-agentcore", region_name=region)
    for attempt in range(5):
        try:
            resp = dp.invoke_agent_runtime(
                agentRuntimeArn=runtime_arn,
                payload=json.dumps({"input": "warmup"}).encode(),
                runtimeSessionId=str(uuid.uuid4()),
            )
            body = resp.get("response", "")
            if hasattr(body, "read"):
                body.read()
            logger.info("Agent warmup successful on attempt %d", attempt + 1)
            return
        except Exception as e:
            logger.debug("Warmup attempt %d failed: %s", attempt + 1, e)
            time.sleep(10)
    logger.warning("Agent warmup failed after 5 attempts — cold start may occur")


def _get_or_create_endpoint(cp, runtime_id: str) -> str:
    """Find a READY endpoint or create one."""
    eps = cp.list_agent_runtime_endpoints(agentRuntimeId=runtime_id)
    for ep in eps.get("runtimeEndpoints", []):
        if ep.get("status") == "READY":
            return ep["agentRuntimeEndpointArn"]

    # Create endpoint
    logger.info("No READY endpoint found, creating...")
    cp.create_agent_runtime_endpoint(agentRuntimeId=runtime_id, name="default")

    for _ in range(60):
        eps = cp.list_agent_runtime_endpoints(agentRuntimeId=runtime_id)
        for ep in eps.get("runtimeEndpoints", []):
            if ep.get("status") == "READY":
                return ep["agentRuntimeEndpointArn"]
        time.sleep(5)

    raise TimeoutError(f"Endpoint for runtime {runtime_id} did not reach READY within 300s")
