"""RAGAS adapter for AgentCore code-based evaluators."""

import logging
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset
from ragas import evaluate

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)

# Separators used to embed reference/context in user messages when the trace
# format has no dedicated fields for them (e.g. build_adot_docs() recipes).
_REFERENCE_SEPARATOR = "\n\nReference Answer:\n"
_CONTEXT_SEPARATOR = "\n\nContext:\n"


class RagasAdapter(BaseAdapter):
    """Adapter that runs a RAGAS metric against AgentCore evaluation events.

    Example (default span mapping)::

        from ragas.metrics import Faithfulness
        from langchain_aws import ChatBedrockConverse
        from ragas.llms import LangchainLLMWrapper
        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RagasAdapter

        eval_llm = LangchainLLMWrapper(ChatBedrockConverse(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name="us-east-1",
        ))
        adapter = RagasAdapter(metric=Faithfulness(), llm=eval_llm)

    Example (custom mapper returning RAGAS dataset dict)::

        from typing import Dict, Any

        def my_mapper(ev: EvaluatorInput) -> Dict[str, Any]:
            return {
                "user_input": ev.session_spans[0]["attributes"]["question"],
                "response": ev.session_spans[0]["attributes"]["answer"],
                "retrieved_contexts": ["some context"],
            }

        adapter = RagasAdapter(
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
                Must have a .name attribute and support ragas.evaluate().
            custom_mapper: Optional callable that receives the EvaluatorInput and
                returns a dict with RAGAS-compatible dataset column keys
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
        if self.custom_mapper is not None:
            dataset_dict = self.custom_mapper(evaluator_input)
            # Wrap scalar values in lists for Dataset.from_dict
            dataset_dict = {k: [v] for k, v in dataset_dict.items()}
        else:
            result = self._default_extract(evaluator_input)
            if not result.input or not result.actual_output:
                missing: List[str] = []
                if not result.input:
                    missing.append("input")
                if not result.actual_output:
                    missing.append("actual_output")
                metric_name = type(self.metric).__name__
                return EvaluatorOutput(
                    errorCode="MISSING_REQUIRED_FIELD",
                    errorMessage=f"Field(s) {missing} required by {metric_name} but not found in evaluation event. "
                    f"Provide a custom_mapper or ensure spans contain the necessary data.",
                )

            # Parse embedded reference and context from the user_input field.
            # Dataset recipes may embed these as:
            #   "{user_input}\n\nContext:\n{context}\n\nReference Answer:\n{reference}"
            # since ADOT trace formats have no dedicated reference/context fields.
            user_input = result.input
            embedded_reference: Optional[str] = None
            embedded_context: Optional[str] = None

            if _REFERENCE_SEPARATOR in user_input:
                user_input, embedded_reference = user_input.split(_REFERENCE_SEPARATOR, 1)

            if _CONTEXT_SEPARATOR in user_input:
                user_input, embedded_context = user_input.split(_CONTEXT_SEPARATOR, 1)

            # Map SpanMapResult fields to RAGAS dataset columns
            dataset_dict: Dict[str, list] = {
                "user_input": [user_input],
                "response": [result.actual_output],
            }

            # Reference priority: reference_inputs > embedded > assertions
            if result.expected_output:
                dataset_dict["reference"] = [result.expected_output]
            elif embedded_reference:
                dataset_dict["reference"] = [embedded_reference]
            elif result.assertions:
                dataset_dict["reference"] = ["\n".join(result.assertions)]

            # Retrieval context priority: span tool results > embedded
            if result.retrieval_context:
                dataset_dict["retrieved_contexts"] = [result.retrieval_context]
                dataset_dict["reference_contexts"] = [result.retrieval_context]
            elif embedded_context:
                dataset_dict["retrieved_contexts"] = [[embedded_context]]
                dataset_dict["reference_contexts"] = [[embedded_context]]

        dataset = Dataset.from_dict(dataset_dict)

        # Run RAGAS evaluation
        eval_result = evaluate(dataset=dataset, metrics=[self.metric])
        df = eval_result.to_pandas()

        # Find the score column — RAGAS uses metric.name as column prefix,
        # sometimes with a suffix like "(mode=f1)"
        score_col = None
        for col in df.columns:
            if col.startswith(self.metric.name):
                score_col = col
                break

        if score_col is None:
            # Fallback: find any numeric column that's not an input field
            input_cols = {"user_input", "response", "reference", "retrieved_contexts", "reference_contexts"}
            for col in df.columns:
                if col not in input_cols and df[col].dtype in ("float64", "int64", "float32"):
                    score_col = col
                    break

        if score_col is None:
            return EvaluatorOutput(
                errorCode="NO_SCORE_FOUND",
                errorMessage=(
                    f"RAGAS evaluate() produced no score column for metric "
                    f"'{self.metric.name}'. Columns: {list(df.columns)}"
                ),
            )

        score = float(df[score_col].iloc[0])

        # Some metrics (e.g. SemanticSimilarity) set threshold=None explicitly;
        # getattr's default only applies when the attribute is missing entirely.
        threshold = getattr(self.metric, "threshold", None)
        threshold = threshold if threshold is not None else 0.5
        label = "Pass" if score >= threshold else "Fail"
        reason = f"RAGAS {self.metric.name}: {score:.4f} (threshold={threshold})"

        return EvaluatorOutput(value=score, label=label, explanation=reason)
