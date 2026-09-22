# Evaluation span helpers

Use public helpers to identify tool spans and select evaluation targets from
collected span dictionaries:

```python
from bedrock_agentcore.evaluation.spans import is_tool_span, tool_span_ids, trace_ids

spans = [
    {
        "traceId": "trace-1",
        "spanId": "span-1",
        "attributes": {"gen_ai.operation.name": "execute_tool"},
    },
]

assert is_tool_span(spans[0])
assert tool_span_ids(spans, trace_id="trace-1") == ["span-1"]
assert trace_ids(spans) == ["trace-1"]
```

These helpers are also exported from `bedrock_agentcore.evaluation`.
They recognize OTel GenAI, OpenInference, and Traceloop tool attributes and
require no AWS client or credentials. Missing or non-dictionary attributes
are treated as non-tool spans.

`tool_span_ids` preserves input order and duplicate IDs, skipping missing or
empty span IDs. Omitting `trace_id`, or passing an empty string, includes all
traces. `trace_ids` skips missing or empty trace IDs and returns each ID once,
in order of first appearance.

To look up an evaluator's level, use `EvaluationClient`:

```python
from bedrock_agentcore.evaluation import EvaluationClient

client = EvaluationClient(region_name="us-west-2")
level = client.get_evaluator_level("Builtin.Helpfulness")
```

This method calls the control plane API and caches the result per evaluator
for the lifetime of the client. If the lookup fails or the response omits the
level, it logs a warning and caches `SESSION` as the fallback, matching the
lookup behavior used by `client.run()`.
