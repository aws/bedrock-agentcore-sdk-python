"""RecommendationClient for AgentCore system prompt and tool description recommendations."""

import logging
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config

from bedrock_agentcore._utils.config import WaitConfig
from bedrock_agentcore._utils.endpoints import DP_ENDPOINT_OVERRIDE
from bedrock_agentcore._utils.polling import wait_until
from bedrock_agentcore._utils.snake_case import accept_snake_case_kwargs, convert_kwargs
from bedrock_agentcore._utils.user_agent import build_user_agent_suffix

logger = logging.getLogger(__name__)


class RecommendationClient:
    """Client for AgentCore recommendations.

    Provides pass-through access to the recommendation APIs on the
    bedrock-agentcore service, plus ``start_recommendation_and_wait`` for the
    asynchronous start-then-poll flow every recommendation needs.

    .. warning::
        This feature is in preview and may change in future releases.

    Example::

        client = RecommendationClient(region_name="us-west-2")

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

        # Pass-through to any recommendation API
        client.list_recommendations(maxResults=10)
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
        for large trace sets.

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
