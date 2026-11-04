# Agents

Guidance for coding agents working on `bedrock-agentcore`, the Amazon Bedrock AgentCore SDK for Python.
Setup and hooks are in [CONTRIBUTING.md](CONTRIBUTING.md); integration test setup is in [TESTING.md](TESTING.md).
This file lists what is enforced and the conventions reviewers ask for repeatedly.

## Commands

| Task                         | Command                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| Install                      | `uv sync`                                                     |
| Lint, format, hygiene        | `uv run pre-commit run --all-files`                           |
| Unit tests with coverage     | `uv run pytest tests/ --cov=src`                              |
| One test file                | `uv run pytest tests/bedrock_agentcore/memory/test_client.py` |
| Integration tests (live AWS) | `uv run pytest tests_integ/<module> -xvs`                     |
| Build                        | `uv build`                                                    |

## What is enforced

- **pre-commit stage** (`.pre-commit-config.yaml`): `uv-lock`, ruff lint with `--fix`, ruff format, file
  hygiene, and the local `no-hardcoded-endpoint-suffix` check. The `uv-lock` hook fails if `uv.lock` is
  out of date, so regenerate it (`uv lock`) whenever `pyproject.toml` dependencies change.
- **pre-push stage**: bandit over `src/` and the full unit suite with coverage.
- **CI** (`ci.yml`): pre-commit on Python 3.10, unit tests on 3.10, 3.11, 3.12 and 3.13, then build,
  `twine check`, and an import of the built wheel on each version. Code must run on Python 3.10.
- **Coverage**: `fail_under = 90` in `pyproject.toml`.
- **Ruff** (`pyproject.toml`): line length 120; rules B, D, E, F, G, I, LOG; Google-style docstrings
  (`D` applies to `src/` only).
- **Breaking-change check** (`breaking-change-check.yml`): griffe compares the public API against `main`.
  It flags any public signature change, including a positional parameter becoming keyword-only. A
  `str` to `Optional[str]` widening is reported but is not breaking; say so in the PR rather than
  reverting.
- **PR title** (`pr-size-title.yml`): a conventional-commit type (`feat`, `fix`, `docs`, `refactor`,
  `test`, `ci`, `chore`, ...), optional scope, and a lowercase subject: `fix(memory): handle empty
  content`. A single-commit PR's commit message must match the title. `scripts/bump_version.py` builds
  the changelog from these prefixes.
- **Integration tests** (`integration-testing.yml`): one job per module (runtime, memory, evaluation,
  services, policy, gateway, identity, payments, tools) in `us-west-2`. PRs from forks wait for maintainer
  approval before the jobs run.

mypy is configured in `pyproject.toml` but not run in CI, and `src/` does not pass it yet. Type every
public function anyway.

## Layout

- `src/bedrock_agentcore/<module>/`: one directory per AgentCore primitive (runtime, memory, identity,
  gateway, tools, evaluation, payments, policy, knowledge_base, ...).
- `src/bedrock_agentcore/<module>/integrations/<framework>/`: framework integrations (Strands, LangGraph).
  Every package directory needs an `__init__.py`; the wheel is built from `src/bedrock_agentcore`, so a
  directory without one is missing from the installed package.
- `src/bedrock_agentcore/_utils/`: shared helpers. Check here before writing one: `endpoints.py`
  (`get_data_plane_endpoint`, `get_control_plane_endpoint`, `get_gateway_mcp_endpoint`, `validate_region`),
  `polling.py` (`wait_until`, `wait_until_deleted`), `config.py` (`WaitConfig`), `snake_case.py`.
- `tests/`: unit tests. Most mirror `src/` under `tests/bedrock_agentcore/`; put a new test beside the
  existing tests for that module.
- `tests_integ/`: integration tests against live AWS.
- `docs/examples/`: usage examples. A new feature needs one.

## Conventions

### Public API

- Renaming or reordering a public parameter breaks callers. Add new parameters as keyword-only (after
  `*`) with defaults, and put required parameters before optional ones, in the same order as sibling
  methods.
- To retire a method or field, keep it working and emit a `DeprecationWarning`; don't delete it.
- Primitive clients expose boto3 operations through `__getattr__` and an `_ALLOWED_*_METHODS` set. Add an
  operation to the allowlist rather than writing a wrapper that only forwards arguments. An explicit
  method with the same name takes precedence, so don't list it in both places.
- Don't re-implement validation the service already performs (enum values, lengths, ARN shape). Validate
  on the client only for what the service cannot check, or where a bad value would otherwise surface
  later as a confusing error.
- Optional dependencies belong to an extra in `pyproject.toml`, with an upper bound (`>=1.0.0,<2.0.0`).
  Declare every package you import directly, even if another dependency pulls it in. Import optional
  extras lazily (the runtime package uses a module-level `__getattr__` for A2A) so the base install
  still imports.

### Partitions and endpoints

The SDK runs in every AWS partition (`aws`, `aws-cn`, `aws-us-gov`). Create boto3 clients without
`endpoint_url` so botocore resolves the partition's endpoint, and pass `endpoint_url` only when
`DP_ENDPOINT_OVERRIDE` or `CP_ENDPOINT_OVERRIDE` (`BEDROCK_AGENTCORE_DP_ENDPOINT`,
`BEDROCK_AGENTCORE_CP_ENDPOINT`) is set, as `config_bundle/client.py` does; dropping that branch silently
sends overridden deployments to the default endpoint. For URLs boto3 doesn't build, use the helpers in
`_utils/endpoints.py`, which derive the DNS suffix from the region's partition. The pre-commit hook rejects
`.amazonaws.com` directly after an interpolated value in `src/`. Match ARNs with a partition-tolerant
pattern (`arn:aws[a-z0-9-]*:`), and validate a region with `validate_region()` before putting it in a URL.
Pass a caller-supplied endpoint through `_validate_endpoint_url()` before signing a request to it.

### Errors

- Catch exceptions by type with `isinstance` or `except SomeError`, never by matching `type(e).__name__`.
- Don't catch broad built-ins such as `ValueError` around a large block. Narrow the `try`, or raise a
  specific subclass and catch that.
- If a docstring says a method raises `SomeError`, wrap transport and parsing failures so that is true.
- Don't swallow errors in a loop; let the call fail rather than return partial results silently.
- Branch on a service error with `e.response["Error"]["Code"]`, not a substring of `str(e)`.
- The runtime server must not return exception details to the caller; log them locally instead.

### Concurrency

- Async code must not block the event loop: run blocking work with `asyncio.to_thread()` and bound waits
  with `asyncio.wait_for()`.
- When a buffer is shared with a background flush, copy and clear it under one lock acquisition.
- Treat an `async def` callback as async: detect it (including `functools.partial` and decorated
  functions) and await it, or raise `TypeError` on the sync path. A coroutine object is truthy, so
  calling one without awaiting can silently pass a check.
- Keep sync and async variants of the same logic in one shared helper so they don't drift.

### Telemetry and logging

- Never put prompts, model responses, tool inputs or outputs, or other user content on spans or in
  logs.
- Report library output through the module logger (`logging.getLogger(__name__)`), not `print`.
- Every boto3 client the SDK creates sets its user agent with `build_user_agent_suffix` from
  `_utils/user_agent.py`.

### Code style

- Imports go at the top of the module, including in tests. Import inside a function only to defer an
  optional extra.
- Name magic numbers as module constants.
- Comments explain why, not what. Reference classes and functions by name in comments and Markdown,
  not file paths, which go stale. Every comment must make sense to a reader of this public repository,
  so don't cite tickets or trackers they can't open.
- A docstring is a contract: if it says a value is limited to 1-20 or that a method raises an error, the
  code must enforce it.
- Write idiomatic Python when porting from the TypeScript SDK. Match its behavior, not its helper
  functions.
- Add only what is needed now; a converter, option or mode with no caller gets removed in review.

## Testing

- Unit tests must not hit AWS. Patch the method the code actually calls; a mock on a wrapper the code
  no longer goes through passes without testing anything.
- Test behavior that can break: error paths and negative cases, not only the happy path. Don't test
  boto3 or botocore behavior, and don't assert the same request twice in separate tests.
- Assert the effect, not just the call: check what was sent or written, not only that a method ran.
  An attribute on a `MagicMock` exists before anything sets it, so `assert event.retry is not True`
  passes without testing anything; assert the exact value instead.
- Shared fixtures go in `conftest.py`, not copied between test files.
- Never commit account IDs, user names or other real identifiers in tests; read them from environment
  variables, as the integration workflows do.
- Integration test conventions (region, setup and cleanup, waiting, skips) are in
  [TESTING.md](TESTING.md#conventions). Don't add a skip that lets CI pass without showing the test never
  ran.

## Pull requests

- Describe the problem, the fix, and how you verified it.
- Update docstrings, `README.md` files and `docs/examples/` in the same PR when a signature or behavior
  changes.
- Keep a feature and an unrelated fix in separate PRs.
