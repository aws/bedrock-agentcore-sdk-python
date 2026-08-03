"""RAGAS adapter for AgentCore code-based evaluators.

Scores metrics through RAGAS's per-sample APIs (``metric.single_turn_score()``
/ ``metric.multi_turn_score()`` for legacy metrics, ``metric.score(**kwargs)``
for ``ragas.metrics.collections`` metrics) rather than the batch
``ragas.evaluate()`` pipeline. This keeps the adapter free of the heavyweight
``datasets``/``pyarrow``/``pandas`` stack, so it can run in zip-based Lambda
deployments with a slim ragas install.
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)

# Separators used to embed reference/context in user messages when the trace
# format has no dedicated fields for them (e.g. build_adot_docs() recipes).
_REFERENCE_SEPARATOR = "\n\nReference Answer:\n"
_CONTEXT_SEPARATOR = "\n\nContext:\n"


class RAGASAdapter(BaseAdapter):
    """Adapter that runs a RAGAS metric against AgentCore evaluation events.

    Supports both RAGAS metric generations:

    - Legacy single-turn metrics (``ragas.metrics``): scored via
      ``metric.single_turn_score(SingleTurnSample(...))``
    - Legacy multi-turn metrics (e.g. ToolCallAccuracy): scored via
      ``metric.multi_turn_score(MultiTurnSample(...))``, with conversation
      messages built from extracted turns and tool calls
    - Collections metrics (``ragas.metrics.collections``) and decorator-based
      custom metrics (``@discrete_metric`` / ``@numeric_metric``): scored via
      ``metric.score(**fields)``

    Numeric results produce value + Pass/Fail label; discrete (string-valued)
    metrics produce a categorical label. When the metric provides reasoning
    (``MetricResult.reason``), it is surfaced as the explanation.

    The adapter evaluates one event at a time by design; it does not use the
    batch ``ragas.evaluate()`` API.

    Multi-turn notes: predicted tool calls are attached to the final AI
    message, and ``reference_tool_calls`` built from
    ``expected_trajectory.toolNames`` carry names without arguments — metrics
    that compare tool arguments need a custom_mapper supplying full
    ``ragas.messages.ToolCall`` objects. Fields with no span source (e.g.
    ``reference_topics`` for TopicAdherenceScore) also require a custom_mapper.

    Example (default span mapping)::

        from ragas.metrics import Faithfulness
        from langchain_aws import ChatBedrockConverse
        from ragas.llms import LangchainLLMWrapper
        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        eval_llm = LangchainLLMWrapper(ChatBedrockConverse(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name="us-east-1",
        ))
        adapter = RAGASAdapter(metric=Faithfulness(), llm=eval_llm)

    Example (custom mapper returning RAGAS field dict)::

        from typing import Dict, Any

        def my_mapper(ev: EvaluatorInput) -> Dict[str, Any]:
            return {
                "user_input": ev.session_spans[0]["attributes"]["question"],
                "response": ev.session_spans[0]["attributes"]["answer"],
                "retrieved_contexts": ["some context"],
            }

        adapter = RAGASAdapter(
            metric=Faithfulness(),
            llm=eval_llm,
            custom_mapper=my_mapper,
        )
    """

    def __init__(
        self,
        metric: Any,
        custom_mapper: Optional[Callable[[EvaluatorInput], Dict[str, Any]]] = None,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ):
        """Initialize the adapter.

        Args:
            metric: A RAGAS metric instance (e.g., Faithfulness(), ContextRecall()).
                Legacy (``ragas.metrics``) and collections
                (``ragas.metrics.collections``) metrics are both supported.
            custom_mapper: Optional callable that receives the EvaluatorInput and
                returns a dict with RAGAS-native field keys
                (user_input, response, reference, retrieved_contexts, reference_contexts).
                Bypasses default span mapping when provided.
            llm: Optional LLM wrapper to set on the metric (e.g., LangchainLLMWrapper).
                Required for most RAGAS metrics when not using OpenAI.
            embeddings: Optional embeddings wrapper to set on the metric.
                Required for embedding-based metrics (AnswerSimilarity, AnswerCorrectness).
        """
        self.metric = metric
        self.custom_mapper = custom_mapper

        if llm is not None:
            self.metric.llm = llm
        if embeddings is not None and hasattr(self.metric, "embeddings"):
            self.metric.embeddings = embeddings

    def _run(self, evaluator_input: EvaluatorInput) -> EvaluatorOutput:
        """Run the RAGAS metric pipeline."""
        span_result = None
        if self.custom_mapper is not None:
            fields = self.custom_mapper(evaluator_input)
        else:
            span_result = self._default_extract(evaluator_input)
            if not span_result.input or not span_result.actual_output:
                missing: List[str] = []
                if not span_result.input:
                    missing.append("input")
                if not span_result.actual_output:
                    missing.append("actual_output")
                metric_name = type(self.metric).__name__
                return EvaluatorOutput(
                    errorCode="MISSING_REQUIRED_FIELD",
                    errorMessage=f"Field(s) {missing} required by {metric_name} but not found in evaluation event. "
                    f"Provide a custom_mapper or ensure spans contain the necessary data.",
                )
            fields = self._build_fields(span_result)

        # Dispatch on metric generation. Legacy single-turn metrics expose
        # single_turn_score(), legacy multi-turn metrics expose
        # multi_turn_score(), and collections metrics expose score(**kwargs).
        if hasattr(self.metric, "single_turn_score"):
            outcome = self._score_legacy(fields)
        elif hasattr(self.metric, "multi_turn_score"):
            outcome = self._score_multi_turn(fields, span_result)
        elif hasattr(self.metric, "ascore") and hasattr(self.metric, "score"):
            outcome = self._score_collections(fields)
        else:
            return EvaluatorOutput(
                errorCode="UNSUPPORTED_METRIC",
                errorMessage=f"{type(self.metric).__name__} does not expose a supported RAGAS scoring "
                f"API (single_turn_score, multi_turn_score, or score). Pass a ragas.metrics "
                f"or ragas.metrics.collections metric instance.",
            )

        if isinstance(outcome, EvaluatorOutput):
            return outcome
        score, metric_reason = outcome

        if math.isnan(score):
            return EvaluatorOutput(
                errorCode="INVALID_SCORE",
                errorMessage=f"RAGAS metric '{self._metric_name()}' returned NaN. This usually means "
                f"required fields were empty or the metric could not be computed.",
            )

        # Some metrics (e.g. SemanticSimilarity) set threshold=None explicitly;
        # getattr's default only applies when the attribute is missing entirely.
        threshold = getattr(self.metric, "threshold", None)
        threshold = threshold if threshold is not None else 0.5
        label = "Pass" if score >= threshold else "Fail"
        explanation = metric_reason or f"RAGAS {self._metric_name()}: {score:.4f} (threshold={threshold})"

        return EvaluatorOutput(value=score, label=label, explanation=explanation)

    def _build_fields(self, result: Any) -> Dict[str, Any]:
        r"""Map a SpanMapResult to RAGAS-native field names.

        Also parses reference and context embedded in the user input field.
        Dataset recipes may embed these as
        ``"{user_input}\n\nContext:\n{context}\n\nReference Answer:\n{reference}"``
        since ADOT trace formats have no dedicated reference/context fields.
        """
        user_input = result.input
        embedded_reference: Optional[str] = None
        embedded_context: Optional[str] = None

        if _REFERENCE_SEPARATOR in user_input:
            user_input, embedded_reference = user_input.split(_REFERENCE_SEPARATOR, 1)

        if _CONTEXT_SEPARATOR in user_input:
            user_input, embedded_context = user_input.split(_CONTEXT_SEPARATOR, 1)

        fields: Dict[str, Any] = {
            "user_input": user_input,
            "response": result.actual_output,
        }

        # Reference priority: reference_inputs > embedded > assertions
        if result.expected_output:
            fields["reference"] = result.expected_output
        elif embedded_reference:
            fields["reference"] = embedded_reference
        elif result.assertions:
            fields["reference"] = "\n".join(result.assertions)

        # Retrieval context priority: span tool results > embedded
        if result.retrieval_context:
            fields["retrieved_contexts"] = result.retrieval_context
            fields["reference_contexts"] = result.retrieval_context
        elif embedded_context:
            fields["retrieved_contexts"] = [embedded_context]
            fields["reference_contexts"] = [embedded_context]

        return fields

    def _score_legacy(self, fields: Dict[str, Any]) -> Any:
        """Score with a legacy (``ragas.metrics``) single-turn metric.

        Returns a ``(score, reason)`` tuple, or an EvaluatorOutput error.
        """
        from ragas.dataset_schema import SingleTurnSample

        required = getattr(self.metric, "required_columns", {}).get("SINGLE_TURN", set())
        missing = sorted(c for c in required if not fields.get(c))
        if missing:
            return self._missing_fields_error(missing)

        sample_fields = {k: v for k, v in fields.items() if k in SingleTurnSample.model_fields}
        sample = SingleTurnSample(**sample_fields)

        try:
            return float(self.metric.single_turn_score(sample)), None
        except ImportError as e:
            return self._missing_dependency_error(e)
        except Exception as e:
            hint = self._dependency_hint(e)
            if hint:
                return hint
            raise

    def _score_collections(self, fields: Dict[str, Any]) -> Any:
        """Score with a collections (``ragas.metrics.collections``) metric.

        Collections metrics declare their inputs as typed ``ascore()``
        parameters, so filter the extracted fields to that signature.
        Decorator-based metrics (``@discrete_metric`` / ``@numeric_metric``)
        expose ``ascore(*args, **kwargs)`` instead; for those, all fields are
        passed through and the metric validates its own inputs (pair them with
        a custom_mapper whose keys match the wrapped function's parameters).

        Returns a ``(score, reason)`` tuple, an EvaluatorOutput error, or a
        categorical EvaluatorOutput for discrete (string-valued) metrics.
        """
        import inspect

        params = inspect.signature(self.metric.ascore).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            kwargs = dict(fields)
        else:
            accepted = {name for name in params if name != "self"}
            required = sorted(
                name
                for name, p in params.items()
                if name != "self" and p.default is inspect.Parameter.empty and not fields.get(name)
            )
            if required:
                return self._missing_fields_error(required)
            kwargs = {k: v for k, v in fields.items() if k in accepted}

        try:
            result = self.metric.score(**kwargs)
        except ImportError as e:
            return self._missing_dependency_error(e)
        except Exception as e:
            hint = self._dependency_hint(e)
            if hint:
                return hint
            raise

        value = result.value
        reason = getattr(result, "reason", None)
        if isinstance(value, str):
            # Discrete metrics return categorical labels with user-defined
            # allowed_values; surface them as-is rather than guessing a number.
            return EvaluatorOutput(label=value, explanation=reason)
        return float(value), reason

    def _score_multi_turn(self, fields: Dict[str, Any], span_result: Any) -> Any:
        """Score with a legacy multi-turn metric (e.g. ToolCallAccuracy).

        Builds a MultiTurnSample from extracted turns and tool calls when
        using default span mapping. With a custom_mapper, the fields dict
        supplies MultiTurnSample fields directly (``user_input`` as a list of
        ``ragas.messages`` objects, ``reference_tool_calls``, etc.).

        Returns a ``(score, reason)`` tuple, or an EvaluatorOutput error.
        """
        from ragas.dataset_schema import MultiTurnSample

        mt_fields = dict(fields)
        if span_result is not None:
            mt_fields["user_input"] = self._build_messages(span_result)
            if span_result.expected_tools:
                mt_fields["reference_tool_calls"] = self._build_ragas_tool_calls(span_result.expected_tools)

        required = getattr(self.metric, "required_columns", {}).get("MULTI_TURN", set())
        missing = sorted(c for c in required if not mt_fields.get(c))
        if missing:
            return self._missing_fields_error(missing)

        sample_fields = {k: v for k, v in mt_fields.items() if k in MultiTurnSample.model_fields}
        sample = MultiTurnSample(**sample_fields)

        try:
            return float(self.metric.multi_turn_score(sample)), None
        except ImportError as e:
            return self._missing_dependency_error(e)
        except Exception as e:
            hint = self._dependency_hint(e)
            if hint:
                return hint
            raise

    def _build_messages(self, span_result: Any) -> List[Any]:
        """Build a ragas message list from extracted turns and tool calls.

        Uses SpanMapResult.turns when the session has multiple invocations,
        falling back to a single user/assistant pair. Predicted tool calls
        are attached to the final AI message.
        """
        from ragas.messages import AIMessage, HumanMessage

        messages: List[Any] = []
        if span_result.turns:
            for turn in span_result.turns:
                role = turn.get("role")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        else:
            messages = [
                HumanMessage(content=span_result.input or ""),
                AIMessage(content=span_result.actual_output or ""),
            ]

        if span_result.tools_called:
            tool_calls = self._build_ragas_tool_calls(span_result.tools_called)
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    messages[i] = AIMessage(content=messages[i].content, tool_calls=tool_calls)
                    break

        return messages

    @staticmethod
    def _build_ragas_tool_calls(tool_dicts: List[Dict[str, Any]]) -> List[Any]:
        """Convert extracted tool call dicts to ragas ToolCall objects.

        Reference tool calls from expected_trajectory carry names only, so
        args default to an empty dict.
        """
        from ragas.messages import ToolCall

        result: List[Any] = []
        for tc in tool_dicts:
            name = tc.get("name", "")
            if not name:
                continue
            args = tc.get("input_parameters")
            result.append(ToolCall(name=name, args=args if isinstance(args, dict) else {}))
        return result

    def _missing_fields_error(self, missing: List[str]) -> EvaluatorOutput:
        """Build a MISSING_REQUIRED_FIELD error for the metric's declared inputs."""
        return EvaluatorOutput(
            errorCode="MISSING_REQUIRED_FIELD",
            errorMessage=f"RAGAS metric '{self._metric_name()}' requires field(s) {missing} which were not "
            f"found in the evaluation event. Provide ground truth via evaluationReferenceInputs, "
            f"embed it in the user message, or supply a custom_mapper.",
        )

    def _missing_dependency_error(self, e: ImportError) -> EvaluatorOutput:
        """Build a MISSING_DEPENDENCY error for packages absent from the environment."""
        return EvaluatorOutput(
            errorCode="MISSING_DEPENDENCY",
            errorMessage=f"RAGAS metric '{self._metric_name()}' requires a package that is not "
            f"installed in this environment: {e}. Install the missing dependency or use a "
            f"metric with a lighter footprint.",
        )

    def _dependency_hint(self, e: Exception) -> Optional[EvaluatorOutput]:
        """Return a targeted error when the metric is missing its LLM/embeddings."""
        msg = str(e).lower()
        if "not set" in msg and ("llm" in msg or "embedding" in msg):
            return EvaluatorOutput(
                errorCode="METRIC_ERROR",
                errorMessage=f"RAGAS metric '{self._metric_name()}' failed: {e}. Pass an LLM or embeddings "
                f"wrapper to the adapter, e.g. RAGASAdapter(metric=..., "
                f"llm=LangchainLLMWrapper(ChatBedrockConverse(...))).",
            )
        return None

    def _metric_name(self) -> str:
        """The metric's declared name, falling back to the class name."""
        return getattr(self.metric, "name", None) or type(self.metric).__name__


# Alias following the repo's brand-casing convention (DeepEvalAdapter, AutoEvalsAdapter).
RagasAdapter = RAGASAdapter
