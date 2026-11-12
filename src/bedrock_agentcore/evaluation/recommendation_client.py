"""RecommendationClient for AgentCore recommendations.

Supports the SYSTEM_PROMPT_RECOMMENDATION, TOOL_DESCRIPTION_RECOMMENDATION and
HARNESS_RECOMMENDATION recommendation types.
"""

import logging
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config

from bedrock_agentcore._utils.config import WaitConfig
from bedrock_agentcore._utils.endpoints import DP_ENDPOINT_OVERRIDE
from bedrock_agentcore._utils.polling import wait_until
from bedrock_agentcore._utils.snake_case import accept_snake_case_kwargs, convert_kwargs
from bedrock_agentcore._utils.user_agent import build_user_agent_suffix
from bedrock_agentcore.config_bundle.bundle import ConfigBundleRef

logger = logging.getLogger(__name__)


class RecommendationClient:
    """Client for AgentCore recommendations.

    Provides pass-through access to the recommendation APIs on the
    bedrock-agentcore service, plus ``start_recommendation_and_wait`` for the
    asynchronous start-then-poll flow every recommendation needs, and
    ``get_recommended_bundle_versions`` to read the configuration bundle version
    references a HarnessRecommendation produced.

    A HarnessRecommendation request sets ``type="HARNESS_RECOMMENDATION"`` and
    wraps its config as ``recommendation_config={"harnessRecommendationConfig": {...}}``,
    with an ``evaluationSource`` (batch evaluation, online evaluation config, or
    agent insights) and a ``harnessConfiguration`` (``configurationBundle`` or
    ``inlineHarnessConfig``). See the `StartRecommendation API reference
    <https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_StartRecommendation.html>`_
    for the full request shape and its limits.

    HarnessRecommendation runs are long. Pass ``WaitConfig(max_wait=1800)`` (or longer)
    to ``start_recommendation_and_wait``; the default of 5 minutes is usually too short.

    .. warning::
        This feature is in preview and may change in future releases.

    Example::

        from bedrock_agentcore.config_bundle.client import ConfigBundleClient
        from bedrock_agentcore.evaluation import WaitConfig

        client = RecommendationClient(region_name="us-west-2")

        # System prompt recommendation
        recommendation = client.start_recommendation_and_wait(
            name="my-prompt-rec",
            type="SYSTEM_PROMPT_RECOMMENDATION",
            recommendationConfig={"systemPromptRecommendationConfig": {
                "systemPrompt": {"text": "You are a helpful assistant."},
                "agentTraces": {"batchEvaluation": {"batchEvaluationArn": arn}},
                "evaluationConfig": {"evaluators": [
                    {"evaluatorArn": "arn:aws:bedrock-agentcore:::evaluator/Builtin.GoalSuccessRate"}
                ]},
            }},
        )
        print(recommendation["recommendationResult"])

        # HarnessRecommendation from a batch evaluation, configuration bundle form
        recommendation = client.start_recommendation_and_wait(
            wait_config=WaitConfig(max_wait=1800, poll_interval=15),
            name="orders-harness-batch",
            type="HARNESS_RECOMMENDATION",
            recommendation_config={"harnessRecommendationConfig": {
                "evaluationSource": {"batchEvaluationSource": {"batchEvaluationId": batch_id}},
                "harnessConfiguration": {"configurationBundle": {
                    "bundleArn": bundle_arn,
                    "versionId": bundle_version_id,
                    "systemPromptJsonPath": "$.agent.systemPrompt",
                }},
            }},
        )

        # Fetch the bundle versions it produced with your own ConfigBundleClient
        # (read-only; nothing is applied to the agent)
        bundle_client = ConfigBundleClient(region_name="us-west-2")
        for ref in client.get_recommended_bundle_versions(recommendation):
            version = bundle_client.get_configuration_bundle_version(
                bundleId=ref.bundle_id, versionId=ref.bundle_version
            )
            print(version["versionId"], version["components"])

        # Pass-through to the other recommendation APIs
        client.list_recommendations(max_results=10, agent_insight_id=insight_id)
    """

    _ALLOWED_DP_METHODS = {
        "start_recommendation",
        "get_recommendation",
        "list_recommendations",
        "delete_recommendation",
    }

    def __init__(
        self,
        region_name: Optional[str] = None,
        integration_source: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
    ):
        """Initialize the RecommendationClient.

        Args:
            region_name: AWS region. Falls back to boto3 session region or us-west-2.
            integration_source: Optional integration framework identifier for telemetry.
            boto3_session: Optional boto3 Session. If not provided, a default is created.
        """
        session = boto3_session if boto3_session else boto3.Session()
        self.region_name = region_name or session.region_name or "us-west-2"
        self.integration_source = integration_source

        client_kwargs: Dict[str, Any] = {
            "region_name": self.region_name,
            "config": Config(user_agent_extra=build_user_agent_suffix(integration_source)),
        }
        if DP_ENDPOINT_OVERRIDE:
            client_kwargs["endpoint_url"] = DP_ENDPOINT_OVERRIDE
        self._dp_client = session.client("bedrock-agentcore", **client_kwargs)

        logger.info("Initialized RecommendationClient in region %s", self.region_name)

    def __getattr__(self, name: str):
        """Dynamically forward allowlisted method calls to the boto3 client."""
        if "_dp_client" not in self.__dict__:
            raise AttributeError(name)

        if name in self._ALLOWED_DP_METHODS and hasattr(self._dp_client, name):
            method = getattr(self._dp_client, name)
            logger.debug("Forwarding method '%s' to _dp_client", name)
            return accept_snake_case_kwargs(method)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Method not found on data plane client. "
            f"Available methods can be found in the boto3 documentation for "
            f"'bedrock-agentcore' service."
        )

    def start_recommendation_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Start a recommendation and wait for it to reach COMPLETED status.

        Recommendations typically take several minutes, so the default
        ``WaitConfig`` (5 minutes) may be too short. Pass a longer ``max_wait``
        for large trace sets, and for HarnessRecommendation runs use at least
        ``WaitConfig(max_wait=1800)``.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the start_recommendation API.

        Returns:
            The GetRecommendation response when COMPLETED. The generated result is
            under ``recommendationResult``. Note that ``recommendationConfig``
            echoes the input you submitted, so read the result from
            ``recommendationResult`` rather than searching the response.

        Raises:
            RuntimeError: If the recommendation reaches FAILED status.
            TimeoutError: If the recommendation doesn't complete within max_wait.
        """
        response = self._dp_client.start_recommendation(**convert_kwargs(kwargs))
        recommendation_id = response["recommendationId"]
        logger.info("Started recommendation %s", recommendation_id)
        return wait_until(
            lambda: self._dp_client.get_recommendation(recommendationId=recommendation_id),
            "COMPLETED",
            {"FAILED"},
            wait_config,
            error_field="recommendationResult",
        )

    def get_recommended_bundle_versions(self, recommendation: Dict[str, Any]) -> List[ConfigBundleRef]:
        """Return the configuration bundle versions a HarnessRecommendation produced.

        Reads ``recommendationResult.harnessRecommendationResult.recommendations``
        and returns a ``ConfigBundleRef`` for each entry whose ``recommendedConfig``
        is a ``configurationBundle``, in recommendation order. This method only
        parses the response; it makes no network calls. Fetch the version content
        with your own ``ConfigBundleClient``, for example
        ``bundle_client.get_configuration_bundle_version(bundleId=ref.bundle_id,
        versionId=ref.bundle_version)``.

        No-change entries (no ``recommendedConfig``) and inline entries
        (``inlineHarnessConfig``) are skipped. Failed entries (``errorCode`` and
        ``errorMessage``) are skipped and logged at WARNING level.

        Args:
            recommendation: A GetRecommendation response, such as the one returned
                by ``start_recommendation_and_wait``.

        Returns:
            One ``ConfigBundleRef`` per recommended bundle version, in recommendation
            order. An empty list when the recommendation produced no bundle version.

        Raises:
            ValueError: If the recommendation is not a HARNESS_RECOMMENDATION, its
                status is not COMPLETED, or it has no ``harnessRecommendationResult``
                (for example, a ``list_recommendations`` summary).
            RuntimeError: If every entry in the result failed.
        """
        rec_type = recommendation["type"]
        if rec_type != "HARNESS_RECOMMENDATION":
            raise ValueError(f"Recommendation is a {rec_type}, not a HARNESS_RECOMMENDATION")
        status = recommendation.get("status")
        if status != "COMPLETED":
            raise ValueError(f"Recommendation status is {status}; bundle versions are only available when COMPLETED")
        harness_result = (recommendation.get("recommendationResult") or {}).get("harnessRecommendationResult")
        if harness_result is None:
            raise ValueError("Recommendation has no harnessRecommendationResult; pass a GetRecommendation response")

        entries = harness_result.get("recommendations", [])
        failures = [entry for entry in entries if "errorCode" in entry]
        if entries and len(failures) == len(entries):
            details = "; ".join(f"{entry['errorCode']}: {entry.get('errorMessage')}" for entry in failures)
            raise RuntimeError(f"all {len(failures)} recommendations failed: {details}")
        for entry in failures:
            logger.warning(
                "Skipping failed recommendation entry: %s: %s", entry["errorCode"], entry.get("errorMessage")
            )

        refs = []
        for entry in entries:
            bundle = (entry.get("recommendedConfig") or {}).get("configurationBundle")
            if bundle is not None:
                refs.append(ConfigBundleRef(bundle_arn=bundle["bundleArn"], bundle_version=bundle["versionId"]))
        return refs
