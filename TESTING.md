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
# Control plane (CRUD on memories/strategies, 5-10 min due to provisioning)
uv run pytest tests_integ/memory/test_controlplane.py -xvs -k "integration"

# Stream delivery (requires env vars below, fails if unset)
export MEMORY_KINESIS_ARN=<Kinesis Data Stream ARN>
export MEMORY_ROLE_ARN=<IAM role ARN trusted by bedrock-agentcore.amazonaws.com with kinesis:PutRecord/PutRecords permissions>
uv run pytest tests_integ/memory/test_memory_client.py -xvs

# Client and developer experience
uv run pytest tests_integ/memory/test_memory_client.py -xvs
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
