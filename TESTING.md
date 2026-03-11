# Integration Tests

See [CONTRIBUTING.md](CONTRIBUTING.md) for initial setup (Python, `uv sync`, etc).

All integration tests require AWS credentials with `bedrock-agentcore:*` and `bedrock-agentcore-control:*` permissions.

```bash
export AWS_PROFILE=<your-profile>
export BEDROCK_TEST_REGION=us-west-2
```

## Runtime

```bash
uv run pytest tests_integ/runtime -xvs --log-cli-level=INFO
```

## Memory

```bash
# Required env vars for all memory tests
export MEMORY_KINESIS_ARN=<Kinesis Data Stream ARN>
export MEMORY_ROLE_ARN=<IAM role ARN trusted by bedrock-agentcore.amazonaws.com with kinesis:PutRecord/PutRecords permissions>
export MEMORY_PREPOPULATED_ID=<ID of a memory with pre-extracted data for retrieval tests>

# Control plane (CRUD on memories/strategies, 5-10 min due to provisioning)
uv run pytest tests_integ/memory/test_controlplane.py -xvs -k "integration"

# Client (data plane, retrieval, stream delivery, lifecycle)
uv run pytest tests_integ/memory/test_memory_client.py -xvs
```

### Pre-populated memory setup

Retrieval tests require a memory with data already extracted into `/facts/`, `/preferences/`, and `/summaries/` namespaces. Create one once per account:

```python
from bedrock_agentcore.memory import MemoryClient

client = MemoryClient(region_name="us-west-2")
memory = client.create_memory_and_wait(
    name="integ_test_prepopulated",
    strategies=[
        {"semanticMemoryStrategy": {"name": "Semantic", "namespaces": ["/facts/{actorId}/"]}},
        {"userPreferenceMemoryStrategy": {"name": "Preferences", "namespaces": ["/preferences/{actorId}/"]}},
        {"summaryMemoryStrategy": {"name": "Summary", "namespaces": ["/summaries/{actorId}/{sessionId}/"]}},
    ],
    event_expiry_days=90,
)
client.save_conversation(
    memory_id=memory["memoryId"],
    actor_id="integ-test-actor",
    session_id="integ-test-session",
    messages=[
        ("I prefer dark mode, Python over Java, and vim over emacs", "USER"),
        ("Noted! Those are solid developer preferences.", "ASSISTANT"),
        ("I work on distributed systems at a startup in Seattle", "USER"),
        ("Interesting! Distributed systems is a great field.", "ASSISTANT"),
    ],
)
# Wait a few minutes for extraction, then verify:
# client.retrieve_memories(memory_id=memory["memoryId"], namespace="/facts/integ-test-actor/", query="developer")
print(f"MEMORY_PREPOPULATED_ID={memory['memoryId']}")
```

## Identity

```bash
uv run pytest tests_integ/identity -xvs
```

## Tools (Code Interpreter, Browser)

```bash
uv run pytest tests_integ/tools -xvs
```

## CI

Integration tests run via `integration-testing.yml` on PRs to main. The workflow runs runtime, memory (stream delivery only), and evaluation tests in parallel.

Stream delivery tests fail in CI unless `MEMORY_KINESIS_ARN` and `MEMORY_ROLE_ARN` secrets are configured on the repo. Other memory integration tests are not yet run in CI due to provisioning times and flaky LLM-dependent assertions — CI support is planned once test stability is addressed.

## Conventions

- Each test file should map 1:1 to a source code file when possible.
- Each test class should correspond to the object under test (e.g. `TestMemoryClient`).
- Expensive resources (e.g. memories) should be created the minimum number of times, ideally once in `setup_class`.
- All resources created during tests must be cleaned up in `teardown_class`.
