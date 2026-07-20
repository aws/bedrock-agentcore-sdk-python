"""DeepEval adapter for AgentCore code-based evaluators."""

import logging
from typing import Any, Callable, Dict, List, Optional

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ToolCall

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)


class DeepEvalAdapter(BaseAdapter):
    """Adapter that runs a DeepEval metric against AgentCore evaluation events.

    Example (default span mapping)::

        from deepeval.metrics import AnswerRelevancyMetric
        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.deepeval import DeepEvalAdapter

        metric = AnswerRelevancyMetric(threshold=0.7)
        adapter = DeepEvalAdapter(metric=metric)

    Example (custom mapper returning LLMTestCase)::

        from deepeval.test_case import LLMTestCase

        def my_mapper(ev: EvaluatorInput) -> LLMTestCase:
            return LLMTestCase(
                input=ev.session_spans[0]["attributes"]["user_query"],
                actual_output=ev.session_spans[0]["attributes"]["response"],
            )

        adapter = DeepEvalAdapter(
            metric=AnswerRelevancyMetric(threshold=0.7),
            custom_mapper=my_mapper,
        )
    """

    def __init__(
        self,
        metric: BaseMetric,
        custom_mapper: Optional[Callable[[EvaluatorInput], LLMTestCase]] = None,
    ):
        """Initialize the adapter.

        Args:
            metric: A DeepEval BaseMetric instance (e.g. AnswerRelevancyMetric).
            custom_mapper: Optional callable that receives the EvaluatorInput and
                returns a LLMTestCase. Bypasses default span mapping when provided.
        """
        self.metric = metric
        self.custom_mapper = custom_mapper

    def _run(self, evaluator_input: EvaluatorInput) -> EvaluatorOutput:
        """Run the DeepEval metric pipeline."""
        if self.custom_mapper is not None:
            test_case = self.custom_mapper(evaluator_input)
        else:
            result = self._default_extract(evaluator_input)
            if not result.input or not result.actual_output:
                missing = []
                if not result.input:
                    missing.append("input")
                if not result.actual_output:
                    missing.append("actual_output")
                metric_name = type(self.metric).__name__
                return EvaluatorOutput(
                    label="Error",
                    errorCode="MISSING_REQUIRED_FIELD",
                    errorMessage=f"Field(s) {missing} required by {metric_name} but not found in evaluation event. "
                    f"Provide a custom_mapper or ensure spans contain the necessary data.",
                )
            # context = ground truth assertions only (not tool outputs)
            context = result.assertions if result.assertions else None
            test_case = LLMTestCase(
                input=result.input,
                actual_output=result.actual_output,
                expected_output=result.expected_output,
                context=context,
                retrieval_context=result.retrieval_context,
                tools_called=self._build_tool_calls(result.tools_called) if result.tools_called else None,
                expected_tools=self._build_tool_calls(result.expected_tools) if result.expected_tools else None,
            )

        try:
            self.metric.measure(test_case)
        except MissingTestCaseParamsError as e:
            return EvaluatorOutput(
                label="Error",
                errorCode="MISSING_REQUIRED_FIELD",
                errorMessage=f"{type(self.metric).__name__} requires fields not extracted from spans: {e}. "
                f"Provide a custom_mapper to supply custom fields from your trace data.",
            )
        except Exception:
            raise

        score = self.metric.score
        reason = getattr(self.metric, "reason", None) or ""
        threshold = getattr(self.metric, "threshold", 0.5)
        success = getattr(self.metric, "success", score is not None and score >= threshold)
        label = "Pass" if success else "Fail"

        return EvaluatorOutput(value=score, label=label, explanation=reason)

    @staticmethod
    def _build_tool_calls(tools_called: List[Dict[str, Any]]) -> List[ToolCall]:
        """Convert extracted tool call dicts to DeepEval ToolCall objects."""
        result: List[ToolCall] = []
        for tc in tools_called:
            name = tc.get("name", "")
            if not name:
                continue
            result.append(ToolCall(
                name=name,
                input_parameters=tc.get("input_parameters"),
                output=tc.get("output"),
            ))
        return result
