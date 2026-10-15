# Evaluation insight results

Some evaluators go beyond scoring and group related failures across the evaluated
sessions. Add one to `evaluator_ids` and the completed result carries them in
`insight_results`:

```python
from bedrock_agentcore.evaluation import (
    BatchEvaluationRunConfig,
    BatchEvaluationRunner,
    BatchEvaluatorConfig,
    CloudWatchDataSourceConfig,
    FileDatasetProvider,
)

config = BatchEvaluationRunConfig(
    batch_evaluation_name="my-failure-analysis",
    evaluator_config=BatchEvaluatorConfig(
        evaluator_ids=["Builtin.FailureAnalysis", "Builtin.GoalSuccessRate"],
    ),
    data_source=CloudWatchDataSourceConfig(
        service_names=["MyAgent.DEFAULT"],
        log_group_names=["/aws/bedrock-agentcore/runtimes/MyAgent-abc123-DEFAULT"],
    ),
)

runner = BatchEvaluationRunner(region="us-west-2")
result = runner.run_dataset_evaluation(
    agent_invoker=agent_invoker,
    dataset=FileDatasetProvider("dataset.json").get_dataset(),
    config=config,
)

for insight in result.insight_results or []:
    print(f"{insight.name} ({insight.affected_session_count} sessions)")
    for cause in insight.root_causes or []:
        print(f"  {cause.name}: {cause.suggested_fix}")
```

`run_dataset_evaluation` polls until the job reaches a terminal state, so the
insights are already on the result it returns.

An empty list and `None` mean different things. An empty list means the analysis
ran and found no failures, while `None` means no analysis evaluator was requested
or its results could not be read, in which case `error_details` names the
evaluator. Branch on the two separately rather than treating both as "no
insights":

```python
if result.insight_results is None:
    print("no analysis ran, or its results were unavailable:", result.error_details)
elif not result.insight_results:
    print("analysis ran and found no failures")
```

The models take the response as it arrives on the wire and expose it as
snake_case attributes. Fields the response omits come back as `None`:

```python
from bedrock_agentcore.evaluation import InsightResult

insight = InsightResult.model_validate(
    {
        "type": "FAILURE",
        "name": "Tool calls fail on missing permissions",
        "description": "Tool calls that change service state fail for lack of permissions.",
        "createdAt": "2026-10-15T21:21:11Z",
        "affectedSessionCount": 2,
        "rootCauses": [
            {
                "name": "Execution role lacks the actions the tools call",
                "explanation": "The role is missing the actions the restart tool calls.",
                "suggestedFix": "Grant the role those actions.",
                "sampleSessions": [
                    {
                        "sessionId": "session-0001",
                        "findings": [
                            {
                                "description": "restart_service failed with PermissionError.",
                                "targetId": "1a027c6a3f598862",
                                "targetType": "SPAN_ID",
                            }
                        ],
                    }
                ],
            }
        ],
    }
)

assert insight.affected_session_count == 2
assert insight.created_at.year == 2026

finding = insight.root_causes[0].sample_sessions[0].findings[0]
assert finding.target_type == "SPAN_ID"
assert finding.confidence is None
assert insight.root_causes[0].sample_sessions[0].last_seen_at is None
```

Sample sessions normally sit on each root cause. An insight carries its own
`sample_sessions` when it has no root causes.
