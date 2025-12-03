# Strands AgentCore Evaluation Integration

This integration enables you to use Amazon Bedrock AgentCore Evaluation API through the Strands Evals framework. Evaluate your Strands agents using built-in or custom evaluators without changing your existing evaluation workflow.

**Two evaluation modes:**
1. **Local agents** - Evaluate Strands agents running locally with in-memory telemetry
2. **Runtime agents** - Evaluate agents deployed to AgentCore Runtime using CloudWatch spans

## Quick Setup

```bash
pip install 'bedrock-agentcore[strands-agents-evals]'
```

Or to develop locally:
```bash
git clone https://github.com/aws/bedrock-agentcore-sdk-python.git
cd bedrock-agentcore-sdk-python
uv sync
source .venv/bin/activate
```

## Local Development with In-Memory Spans

Evaluate Strands agents during local development and testing. The integration captures OpenTelemetry spans from Strands' instrumentation and automatically converts them to ADOT format for evaluation.

### Setup Agent and Telemetry

```python
from strands import Agent, tool
from strands_evals import Experiment, Case
from strands_evals.telemetry import StrandsEvalsTelemetry
from bedrock_agentcore.evaluation import create_strands_evaluator

# Define your tools
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression."""
    return str(eval(expression))

# Setup telemetry to capture spans
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()

# Create your agent
agent = Agent(
    tools=[calculator],
    system_prompt="You are a helpful math assistant."
)
```

### Define Task Function

The task function runs your agent and returns raw OpenTelemetry spans:

```python
def task_fn(case):
    # Run the agent
    agent_response = agent(case.input)

    # Get raw spans from telemetry exporter
    # Note: Convert tuple to list to avoid Pydantic serialization warnings
    raw_spans = list(telemetry.in_memory_exporter.get_finished_spans())

    return {
        "output": str(agent_response),
        "trajectory": raw_spans  # Raw OTel spans - automatically converted to ADOT
    }
```

> **Note:** `get_finished_spans()` returns a tuple. Converting to list with `list()` avoids a harmless Pydantic serialization warning.

### Run Evaluation

```python
# Create test cases
cases = [
    Case(input="What is 5 + 3?", expected_output="8"),
    Case(input="Calculate 10 + 7", expected_output="17"),
]

# Create evaluator
evaluator = create_strands_evaluator("Builtin.Helpfulness")

# Run evaluations
experiment = Experiment(cases=cases, evaluators=[evaluator])
reports = experiment.run_evaluations(task_fn)
report = reports[0]

# View results
print(f"Overall score: {report.overall_score:.2f}")
print(f"Pass rate: {sum(report.test_passes) / len(report.test_passes):.1%}")
```

## Production Evaluation with CloudWatch Spans

Evaluate agents using ADOT spans collected in CloudWatch. Works for both AgentCore Runtime agents and custom agents that upload spans to CloudWatch.

### Prerequisites

- ADOT instrumentation configured
- Spans uploaded to CloudWatch (aws/spans for ADOT spans, configurable log group for events)
- AWS credentials with CloudWatch Logs access

### Fetch Spans from CloudWatch

ADOT spans are written to CloudWatch and typically appear 3-5 minutes after agent invocation. Use `fetch_spans_from_cloudwatch` to retrieve them:

```python
from bedrock_agentcore.evaluation import fetch_spans_from_cloudwatch
from datetime import datetime, timedelta, timezone

start_time = datetime.now(timezone.utc) - timedelta(minutes=10)

# For AgentCore Runtime agents
spans = fetch_spans_from_cloudwatch(
    session_id="your-session-id",
    event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC123-DEFAULT",
    start_time=start_time
)

# For custom agents with configurable log groups
spans = fetch_spans_from_cloudwatch(
    session_id="your-session-id",
    event_log_group="/my-app/agent-events",  # Your custom log group
    start_time=start_time
)
```

### Evaluation Workflow

```python
from strands_evals import Case, Experiment
from bedrock_agentcore.evaluation import create_strands_evaluator, fetch_spans_from_cloudwatch
import time

# 1. Invoke your agent and capture response
agent_core_client = boto3.client("bedrock-agentcore", region_name="us-west-2")
test_input = "What is 2+2?"

response = agent_core_client.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-agent-ABC123",
    payload=json.dumps({"input": test_input}).encode()
)

# Extract session ID and response from invocation
baggage = response.get("baggage", "")
session_id = None
for item in baggage.split(","):
    if item.strip().startswith("session.id="):
        session_id = item.split("=", 1)[1]
        break

agent_output = response["payload"].read().decode("utf-8")

# 2. Wait for spans to reach CloudWatch (3-5 minutes)
print("Waiting for spans to reach CloudWatch...")
time.sleep(300)

# 3. Fetch ADOT spans from CloudWatch
start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
spans = fetch_spans_from_cloudwatch(
    session_id=session_id,
    event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC123-DEFAULT",
    start_time=start_time
)

# 4. Evaluate with fetched spans
cases = [Case(input=test_input, expected_output="4")]

def task_fn(case):
    return {
        "output": agent_output,  # Response from agent invocation
        "trajectory": spans  # ADOT spans from CloudWatch
    }

evaluator = create_strands_evaluator("Builtin.Helpfulness")
experiment = Experiment(cases=cases, evaluators=[evaluator])
reports = experiment.run_evaluations(task_fn)
report = reports[0]

print(f"Overall score: {report.overall_score:.2f}")
```

## Available Evaluators

### Built-in Evaluators

AgentCore provides several built-in evaluators:

- `Builtin.Helpfulness` - Evaluates how helpful the agent's response is
- `Builtin.Accuracy` - Evaluates factual accuracy of responses
- `Builtin.Harmfulness` - Detects potentially harmful content
- `Builtin.Relevance` - Evaluates response relevance to the query

### Custom Evaluators

You can also use custom evaluator ARNs:

```python
evaluator = create_strands_evaluator(
    "arn:aws:bedrock:us-west-2:123456789012:evaluator/my-custom-evaluator"
)
```

## Configuration Options

### Region

Specify AWS region (default: from `AWS_REGION` environment variable or `us-west-2`):

```python
evaluator = create_strands_evaluator(
    "Builtin.Helpfulness",
    region="us-east-1"
)
```

### Test Pass Score

Set minimum score threshold for tests to pass (default: `0.7`):

```python
evaluator = create_strands_evaluator(
    "Builtin.Helpfulness",
    test_pass_score=0.8  # 80% threshold
)
```

## Error Handling

The evaluator handles common errors gracefully:

- **Empty trajectory**: Returns score 0.0 if agent fails to execute
- **Invalid spans**: Returns score 0.0 if span objects are malformed
- **API errors**: Returns score 0.0 with error message

## Troubleshooting

### "No trajectory data available"

**For local agents:** Ensure you're capturing spans correctly:
```python
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
# ... run agent ...
spans = telemetry.in_memory_exporter.get_finished_spans()
```

**For Runtime agents:** Verify spans exist in CloudWatch and you've waited 3-5 minutes after invocation. Check that you're using the correct log group format: `/aws/bedrock-agentcore/runtimes/{agent_id}-{endpoint}`

**For custom agents:** Verify your agent is uploading spans to CloudWatch and you're using the correct log group name.

### "Invalid span objects"

**For local agents:** Verify you're passing raw Span objects, not serialized data:
```python
# Recommended - avoids Pydantic warning
return {"trajectory": list(telemetry.in_memory_exporter.get_finished_spans())}

# Also works - but triggers harmless Pydantic warning
return {"trajectory": telemetry.in_memory_exporter.get_finished_spans()}

# Invalid - don't serialize spans
return {"trajectory": json.dumps(spans)}
```

**For Runtime agents:** Ensure you're filtering for valid ADOT documents with required fields (`scope`, `traceId`, `spanId`).

### Pydantic Serialization Warning

If you see:
```
UserWarning: Pydantic serializer warnings:
  PydanticSerializationUnexpectedValue(Expected `list[any]` - serialized value may not be as expected [field_name='actual_trajectory', input_value=(<opentelemetry.sdk.trace... object at 0x...>), input_type=tuple])
```

**Cause:** OpenTelemetry's `get_finished_spans()` returns a tuple, but Strands Evals expects a list.

**Solution:** Convert to list in your task function:
```python
raw_spans = list(telemetry.in_memory_exporter.get_finished_spans())
```

This warning is cosmetic and doesn't affect evaluation scores, but converting to list eliminates it.

### AWS Credentials

Ensure you have valid AWS credentials configured:
```bash
aws configure
# or
export AWS_PROFILE=your-profile
```

## API Reference

### `create_strands_evaluator(evaluator_id, **kwargs)`

Creates a Strands-compatible evaluator backed by AgentCore Evaluation API.

**Parameters:**
- `evaluator_id` (str): Built-in evaluator name (e.g., "Builtin.Helpfulness") or custom evaluator ARN
- `region` (str, optional): AWS region. Default: from `AWS_REGION` environment variable or `us-west-2`
- `test_pass_score` (float, optional): Minimum score for test to pass (0.0-1.0). Default: 0.7

**Returns:**
- `StrandsEvalsAgentCoreEvaluator`: Evaluator instance compatible with Strands Evals

**Example:**
```python
evaluator = create_strands_evaluator(
    "Builtin.Helpfulness",
    region="us-east-1",
    test_pass_score=0.8
)
```

### `fetch_spans_from_cloudwatch(session_id, event_log_group, start_time, **kwargs)`

Fetches ADOT spans from CloudWatch for any agent with configurable event log group.

**Parameters:**
- `session_id` (str): Session ID from agent execution
- `event_log_group` (str): CloudWatch log group name for event logs
  - For Runtime agents: `/aws/bedrock-agentcore/runtimes/{agent_id}-{endpoint}`
  - For custom agents: Any log group you configured (e.g., `/my-app/agent-events`)
- `start_time` (datetime): Start time for log query
- `region` (str, optional): AWS region. Default: from `AWS_REGION` environment variable or `us-west-2`

**Returns:**
- `List[dict]`: ADOT span and log record dictionaries

**Note:** Always queries `aws/spans` for ADOT spans and the specified `event_log_group` for event logs.

**Example (Runtime agent):**
```python
from bedrock_agentcore.evaluation import fetch_spans_from_cloudwatch
from datetime import datetime, timedelta, timezone

start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
spans = fetch_spans_from_cloudwatch(
    session_id="abc-123",
    event_log_group="/aws/bedrock-agentcore/runtimes/my-agent-ABC123-DEFAULT",
    start_time=start_time
)
```

**Example (Custom agent):**
```python
spans = fetch_spans_from_cloudwatch(
    session_id="abc-123",
    event_log_group="/my-app/agent-events",
    start_time=start_time
)
```

### `convert_strands_to_adot(raw_spans)`

Converts Strands OTel spans to ADOT format (used internally by the evaluator).

**Parameters:**
- `raw_spans` (List[Span]): List of OpenTelemetry Span objects

**Returns:**
- `List[dict]`: ADOT-formatted documents (spans and log records)

**Note:** You typically don't need to call this directly - the evaluator handles conversion automatically.

**Example:**
```python
from strands_evals.telemetry import StrandsEvalsTelemetry
from bedrock_agentcore.evaluation import convert_strands_to_adot

telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
# ... run agent ...
raw_spans = telemetry.in_memory_exporter.get_finished_spans()
adot_docs = convert_strands_to_adot(raw_spans)
```

## Learn More

- [AgentCore Evaluation API Documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluation.html)
- [Strands Evals Documentation](https://github.com/strands-agents/evals)
- [Built-in Evaluators Guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluation-builtin.html)
- [AgentCore Observability Guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html)
