"""RAGAS adapter for AgentCore code-based evaluators.

Scores metrics through RAGAS's per-sample APIs (``metric.single_turn_score()``
/ ``metric.multi_turn_score()`` for legacy metrics, ``metric.score(**kwargs)``
for ``ragas.metrics.collections`` metrics) rather than the batch
``ragas.evaluate()`` pipeline. The adapter itself therefore adds no dependency
on the heavyweight ``datasets``/``pyarrow``/``pandas`` stack. Note that ragas
<1.0 still imports ``datasets`` when the ragas package itself is imported, so
size-constrained deployments (e.g. zip-based Lambda) additionally need a ragas
build with that import chain trimmed; this adapter is compatible with such
builds because it never calls the ``datasets``-backed APIs.
"""

import json
import logging
import math
from collections import defaultdict, deque
from collections.abc import Sequence
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_origin

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)

# A scoring call yields either a (score, reason) pair or an EvaluatorOutput
# carrying an error / categorical label.
_ScoreOutcome = Union[Tuple[float, Optional[str]], EvaluatorOutput]

# Separators for ground truth and retrieval context embedded in the user
# message. ADOT trace formats have no dedicated fields for these, so dataset
# preparation commonly appends them to the user input with known markers.
_REFERENCE_SEPARATOR = "\n\nReference Answer:\n"
_CONTEXT_SEPARATOR = "\n\nContext:\n"

# RAGAS field names whose values are conversation-shaped (message lists / tool
# call objects) rather than the flat strings used by single-turn metrics.
_CONVERSATION_FIELDS = frozenset({"user_input", "reference_tool_calls"})


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

    Conversation-shaped metrics receive the same converted inputs whichever
    generation they belong to: the legacy multi-turn classes via
    ``multi_turn_score()``, and collections metrics (e.g. collections
    ToolCallAccuracy) via ``score()`` when their signature declares sequence
    types. Predicted tool calls are attached to the final AI message.
    ``reference_tool_calls`` come from ``expected_trajectory.toolNames``, which
    carries names only; the predicted arguments of same-named calls are adopted
    so argument-aware metrics score tool selection and sequence. Supply a
    custom_mapper with full ``ragas.messages.ToolCall`` objects to compare
    arguments against independent ground truth.

    Fields with no span source (e.g. ``reference_topics`` for
    TopicAdherenceScore, ``reference_contexts`` for non-LLM context comparison
    metrics) require a custom_mapper.

    Metrics supporting both modes (e.g. AspectCritic) are always scored
    single-turn; wrap the metric or use a custom evaluator for multi-turn use.

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
        threshold: Optional[float] = None,
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
            threshold: Optional Pass/Fail threshold override. When None, the
                metric's own threshold is used, defaulting to 0.5 (useful for
                collections metrics, which carry no threshold of their own).
        """
        self.metric = metric
        self.custom_mapper = custom_mapper
        self.threshold = threshold

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
                return EvaluatorOutput(
                    errorCode="MISSING_REQUIRED_FIELD",
                    errorMessage=f"Field(s) {missing} required by {self._metric_name()} but not found in "
                    f"evaluation event. Provide a custom_mapper or ensure spans contain the necessary data.",
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
            outcome = self._score_collections(fields, span_result)
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

        # Adapter override > metric threshold > 0.5 default. Some metrics
        # (e.g. SemanticSimilarity) set threshold=None explicitly; getattr's
        # default only applies when the attribute is missing entirely.
        threshold = self.threshold
        if threshold is None:
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
        user_input, embedded_reference, embedded_context = self._split_embedded(result.input)

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

        # Retrieval context priority: span tool results > embedded.
        # reference_contexts (ground-truth contexts) is deliberately NOT
        # defaulted from retrieved contexts: doing so would make reference-
        # comparison metrics score retrieval output against itself. Supply it
        # via a custom_mapper when a genuine ground-truth source exists.
        if result.retrieval_context:
            fields["retrieved_contexts"] = result.retrieval_context
        elif embedded_context:
            fields["retrieved_contexts"] = self._interpret_embedded_context(embedded_context)

        return fields

    @staticmethod
    def _split_embedded(text: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Split embedded reference/context sections out of a user message.

        Returns ``(clean_text, reference, context)`` where reference/context
        are None when the corresponding marker is absent.
        """
        reference: Optional[str] = None
        context: Optional[str] = None
        if _REFERENCE_SEPARATOR in text:
            text, reference = text.split(_REFERENCE_SEPARATOR, 1)
        if _CONTEXT_SEPARATOR in text:
            text, context = text.split(_CONTEXT_SEPARATOR, 1)
        return text, reference, context

    @staticmethod
    def _interpret_embedded_context(embedded_context: str) -> List[str]:
        """Recover a ranked chunk list from embedded context text.

        Multi-chunk retrieval contexts are embedded as a JSON-serialized list
        of strings so chunk boundaries and rank order survive the round trip.
        This matters for rank-aware metrics (LLMContextPrecision and friends):
        joining N ranked chunks into one string collapses precision@k to a
        binary judgment on a single blob.

        If the text is not a JSON list of strings, it is treated as a single
        chunk (backward compatible with plain-text contexts).
        """
        try:
            parsed = json.loads(embedded_context)
        except (json.JSONDecodeError, ValueError):
            return [embedded_context]
        if isinstance(parsed, list) and parsed and all(isinstance(item, str) for item in parsed):
            return parsed
        return [embedded_context]

    def _invoke(self, score_fn: Callable[[], _ScoreOutcome]) -> _ScoreOutcome:
        """Run a ragas scoring call, mapping known failure modes to error outputs."""
        try:
            return score_fn()
        except ImportError as e:
            return self._missing_dependency_error(e)
        except Exception as e:
            hint = self._dependency_hint(e)
            if hint:
                return hint
            raise

    def _score_sample(
        self,
        fields: Dict[str, Any],
        sample_cls: Any,
        required_key: str,
        score_fn: Callable[[Any], float],
    ) -> _ScoreOutcome:
        """Validate declared required fields, then score through a legacy Sample class.

        Shared by the single-turn and multi-turn paths, which differ only in the
        sample class, the ``required_columns`` key, and the metric's scoring method.
        """
        required = getattr(self.metric, "required_columns", {}).get(required_key, set())
        missing = sorted(c for c in required if not fields.get(c))
        if missing:
            return self._missing_fields_error(missing)

        sample = sample_cls(**{k: v for k, v in fields.items() if k in sample_cls.model_fields})
        return self._invoke(lambda: (float(score_fn(sample)), None))

    def _score_legacy(self, fields: Dict[str, Any]) -> _ScoreOutcome:
        """Score with a legacy (``ragas.metrics``) single-turn metric."""
        from ragas.dataset_schema import SingleTurnSample

        return self._score_sample(fields, SingleTurnSample, "SINGLE_TURN", self.metric.single_turn_score)

    def _score_collections(self, fields: Dict[str, Any], span_result: Any = None) -> _ScoreOutcome:
        """Score with a collections (``ragas.metrics.collections``) metric.

        Collections metrics declare their inputs as typed ``ascore()``
        parameters, so filter the extracted fields to that signature.
        Conversation-shaped metrics (e.g. collections ToolCallAccuracy, whose
        ``user_input`` is a message list rather than a string) receive the same
        converted messages and reference tool calls the legacy multi-turn path
        builds, selected by the parameter's declared type.

        Decorator-based metrics (``@discrete_metric`` / ``@numeric_metric``)
        expose ``ascore(*args, **kwargs)`` instead; for those, all fields are
        passed through and the metric validates its own inputs (pair them with
        a custom_mapper whose keys match the wrapped function's parameters).

        Returns a ``(score, reason)`` tuple, an EvaluatorOutput error, or a
        categorical EvaluatorOutput for discrete (string-valued) metrics.
        """
        params = signature(self.metric.ascore).parameters
        if any(p.kind is Parameter.VAR_KEYWORD for p in params.values()):
            kwargs = dict(fields)
        else:
            resolved = dict(fields)
            # Parameters naming a conversation field with a sequence type need
            # the converted messages / tool calls rather than the flat
            # string values used by single-turn metrics.
            conversation_params = [
                name
                for name in _CONVERSATION_FIELDS
                if name in params and self._expects_sequence(params[name].annotation)
            ]
            if span_result is not None and conversation_params:
                conv = self._conversation_fields(span_result)
                resolved.update({name: conv[name] for name in conversation_params if name in conv})

            accepted = {name for name in params if name != "self"}
            required = sorted(
                name
                for name, p in params.items()
                if name != "self" and p.default is Parameter.empty and not resolved.get(name)
            )
            if required:
                return self._missing_fields_error(required)
            kwargs = {k: v for k, v in resolved.items() if k in accepted}

        def score() -> _ScoreOutcome:
            result = self.metric.score(**kwargs)
            value = result.value
            reason = getattr(result, "reason", None)
            if isinstance(value, str):
                # Discrete metrics return categorical labels with user-defined
                # allowed_values; surface them as-is rather than guessing a number.
                return EvaluatorOutput(label=value, explanation=reason)
            return float(value), reason

        return self._invoke(score)

    def _score_multi_turn(self, fields: Dict[str, Any], span_result: Any) -> _ScoreOutcome:
        """Score with a legacy multi-turn metric (e.g. ToolCallAccuracy).

        Builds conversation messages and reference tool calls from the span
        extraction. With a custom_mapper, the fields dict supplies
        MultiTurnSample fields directly (``user_input`` as a list of
        ``ragas.messages`` objects, ``reference_tool_calls``, etc.).
        """
        from ragas.dataset_schema import MultiTurnSample

        if span_result is not None:
            fields = {**fields, **self._conversation_fields(span_result)}
        return self._score_sample(fields, MultiTurnSample, "MULTI_TURN", self.metric.multi_turn_score)

    def _conversation_fields(self, span_result: Any) -> Dict[str, Any]:
        """Build the conversation-shaped fields shared by multi-turn scoring paths.

        Returns ``user_input`` as a ragas message list, plus
        ``reference_tool_calls`` when the reference inputs supplied an
        expected tool trajectory.
        """
        conv: Dict[str, Any] = {"user_input": self._build_messages(span_result)}
        if span_result.expected_tools:
            reference = self._build_ragas_tool_calls(span_result.expected_tools)
            # expectedTrajectory.toolNames expresses "call these tools, in this
            # order" and carries no arguments. Adopt the predicted arguments for
            # same-named calls so argument-aware metrics score tool selection and
            # sequence instead of counting every argument as a mismatch.
            if span_result.tools_called:
                reference = self._align_reference_args(reference, span_result.tools_called)
            conv["reference_tool_calls"] = reference
        return conv

    @staticmethod
    def _align_reference_args(reference: List[Any], tools_called: List[Dict[str, Any]]) -> List[Any]:
        """Fill argument-less reference tool calls from the predicted calls of the same name.

        Only applies to references with no arguments at all (i.e. built from
        ``expectedTrajectory.toolNames``); references supplied with real
        arguments are left untouched. Predicted arguments are consumed in call
        order per tool name, so a tool expected several times pairs with each
        invocation rather than reusing the first one's arguments.
        """
        from ragas.messages import ToolCall

        pending: Dict[str, Any] = defaultdict(deque)
        for tc in tools_called:
            name = tc.get("name")
            args = tc.get("input_parameters")
            if name and isinstance(args, dict):
                pending[name].append(args)

        aligned: List[Any] = []
        for call in reference:
            if not call.args and pending.get(call.name):
                aligned.append(ToolCall(name=call.name, args=pending[call.name].popleft()))
            else:
                aligned.append(call)
        return aligned

    @staticmethod
    def _expects_sequence(annotation: Any) -> bool:
        """Whether a parameter annotation declares a list/sequence type."""
        # Bare `list` has no typing origin; parameterized `List[x]` does.
        if annotation in (list, tuple, Sequence):
            return True
        if get_origin(annotation) in (list, tuple, Sequence):
            return True
        if isinstance(annotation, str):
            # Unresolved annotations, e.g. under `from __future__ import annotations`.
            return annotation.lstrip().startswith(("List[", "list[", "Sequence[", "typing.List["))
        return False

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
                    # Strip embedded reference/context so ground truth does not
                    # leak into the conversation the metric judges.
                    clean, _, _ = self._split_embedded(content)
                    messages.append(HumanMessage(content=clean))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        else:
            clean, _, _ = self._split_embedded(span_result.input or "")
            messages = [
                HumanMessage(content=clean),
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

        Used for both predicted calls (which carry ``input_parameters``) and
        reference calls from ``expected_trajectory``, which name tools without
        arguments and so default to an empty dict.
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
