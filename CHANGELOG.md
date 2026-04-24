# Changelog

## [1.6.4] - 2026-04-23

### Added
- feat: add utility methods for primitive clients (#424) (d3537c6)

### Fixed
- fix: use correct score field for relevance filtering in retrieve_customer_context (#415) (e632e9d)
- fix: implement update_message() for guardrail redaction support (#388) (9ba4512)

### Other Changes
- fix(converters): prepend reasoningContent blocks in _openai_to_bedrock() (#419) (667ef55)

## [1.6.3] - 2026-04-16

### Fixed
- fix: validate region parameter to prevent SSRF request redirection (#417) (640b3ad)

## [1.6.2] - 2026-04-13

### Fixed
- fix: make agentcore-worker-loop compatible with OTEL threading instrumentation (#405) (1235897)

### Other Changes
- fix(ci): increase memory integration test timeout to 15 minutes (#401) (180a7c5)

## [1.6.1] - 2026-04-10

### Added
- feat: add read_only flag to AgentCoreMemorySessionManager to disable ACM persistence (#389) (215b5bd)

### Fixed
- fix: replace blocklist with allowlist for install_packages() package validation (#403) (ed953b5)
- fix: skip integration tests for Dependabot and remove missing label (#382) (4ebfdcb)
- fix: pin griffe version and fix ExplanationStyle attribute error (#381) (2bdb9f1)

### Other Changes
- chore(deps): bump boto3 and botocore minimum to 1.42.86 (#399) (df9a21d)
- fix(ci): add pytest-rerunfailures to integration test dependencies (#400) (5ccb283)

## [1.6.0] - 2026-03-31

### Added
- feat: add custom code-based evaluator decorator and typed models (#383) (09f45f3)

## [1.5.1] - 2026-03-31

### Other Changes
- Revert "feat: Emit OTEL attributes for AgentCore Evaluation support (#368)" (#380) (a516260)

## [1.5.0] - 2026-03-30

### Added
- feat: add ground truth support to EvaluationClient and OnDemandEvaluationDatasetRunner (#376) (29b0115)
- feat: Add certificates to create_code_interpreter() (#373) (bc95735)
- feat: add support for policies and certificates (#371) (88c5101)

### Other Changes
- chore: Modify arn parsing in runtime client to allow for different (#362) (a10c0fd)
- ci: add breaking change detection workflow for pull requests (#358) (9341468)
- ci: add backward compatibility integration test workflow (#357) (50654c1)
- ci: add Dependabot auto-merge workflow (#361) (38425ec)

## [1.4.8] - 2026-03-26

### Added
- feat: Emit OTEL attributes for AgentCore Evaluation support (#368) (8bae410)

### Fixed
- fix: lazy import strands-agents-evals to avoid ImportError when not installed (#367) (2c4911d)

### Other Changes
- ci: migrate PyPI publishing to OIDC Trusted Publishing (#363) (5bdf009)
- ci(deps): bump trufflesecurity/trufflehog from 3.93.8 to 3.94.0 (#360) (cd86ebd)
- ci(deps): bump actions/github-script from 7 to 8 (#359) (64db2f0)
- Skip LTM retrieval when content has no text (#299) (a49db7d)

## [1.4.7] - 2026-03-18

### Added
- feat: add AG-UI protocol support via serve_ag_ui and AGUIApp (#350) (e799792)
- feat: add missing data plane passthroughs and integration tests (#352) (530f203)
- feat: add A2A protocol support via serve_a2a (#349) (be1be55)
- feat: add ResourcePolicyClient for resource-based policy management (#328) (51e26c7)
- feat: add data plane, extraction, and lifecycle integ tests to TestMemoryClient (#334) (62fdc9a)

### Fixed
- fix: normalize snake_case/camelCase in passthrough methods for consistent SDK API (#348) (137479c)
- fix: add retries for flaky integ tests that depend on LLM responses (#351) (8424c7a)
- fix: add missing agentId metadata to batched agent state flush (#331) (11e2ac5)
- fix: use separate ports for runtime integ tests to avoid parallel conflicts (#332) (03b599a)

### Other Changes
- ci(deps): bump actions/github-script from 7 to 8 (#80) (0dd49c0)
- chore(deps): bump pyasn1 from 0.6.2 to 0.6.3 (#353) (d3213c0)
- ci(deps): bump actions/checkout from 5 to 6 (#165) (82869c8)
- Revise agent deployment instructions in README (#130) (bb49c2e)
- ci(deps): bump aws-actions/configure-aws-credentials from 5 to 6 (#345) (d508c24)
- feat(strands-memory): add event metadata support to AgentCoreMemorySessionManager (#339) (cd2f2a0)
- chore(deps): bump pyjwt from 2.10.1 to 2.12.0 (#341) (2f4f297)
- ci(deps): bump slackapi/slack-github-action from 2.0.0 to 3.0.1 (#344) (f8710fa)
- ci(deps): bump trufflesecurity/trufflehog from 3.90.6 to 3.93.8 (#343) (d4d1892)
- chore: remove manual_test_memory_client.py (#337) (3dbb793)

## [1.4.6] - 2026-03-12

### Added
- feat: rewrite controlplane integration tests with pytest (#323) (7d7fa48)
- feat: add `name` parameter to `browser_session()` and `SessionConfiguration` (#326) (1ec1a62)

### Fixed
- fix: replace hardcoded sleeps with polling in session manager tests (#324) (18b428a)
- fix: use memoryStrategyId instead of strategyId in search_long_term_memories (#314) (a30f8da)
- fix: return 400 for UnicodeDecodeError in invocation handler (#313) (4730894)

### Other Changes
- feat(memory): add boto3_session parameter to MemoryClient (#330) (66c8488)
- chore: remove test_devex.py, update TESTING.md (#325) (a01ce71)
- fix(memory): pass through unknown config keys in _wrap_configuration (#322) (2afa155)
- test: add stream delivery integ tests and TESTING.md (#317) (32aa019)
- fix(memory): handle SELF_MANAGED override type in _wrap_configuration (#290) (0206ae4)

## [1.4.5] - 2026-03-11

### Fixed
- fix: apply ruff formatting to openai converter files (#312) (98871e9)
- fix(strands-memory): restore positional arg order in AgentCoreMemorySessionManager.__init__ (#318) (98100d7)

### Other Changes
- chore: remove deprecated legacy-release workflow (#315) (84d2916)
- ci: temporarily disable memory integration tests (#319) (ae4c15c)
- chore(deps-dev): bump wheel from 0.45.1 to 0.46.2 (#221) (a77f13a)
- chore(deps): bump cryptography from 45.0.5 to 46.0.5 (#306) (9a21bdc)
- chore(deps): bump starlette from 0.47.1 to 0.49.1 (#307) (538c56f)
- chore(deps): bump werkzeug from 3.1.5 to 3.1.6 (#308) (3f8424c)
- chore(deps): bump pillow from 11.3.0 to 12.1.1 (#309) (85d1465)
- chore(deps): bump mcp from 1.12.2 to 1.23.0 (#310) (db3fcba)
- ci(deps): bump actions/download-artifact from 5 to 6 (#139) (ff66b49)
- ci(deps): bump actions/upload-artifact from 4 to 5 (#140) (fc68025)
- Add daily Slack notification for open PRs (#304) (fda82da)
- chore(deps): bump python-multipart from 0.0.20 to 0.0.22 (#224) (8b8c6fd)
- chore(deps): bump werkzeug from 3.1.3 to 3.1.5 (#228) (7b2bb45)
- chore(deps): bump virtualenv from 20.31.2 to 20.36.1 (#229) (d92ad87)
- chore(deps): bump urllib3 from 2.5.0 to 2.6.3 (#230) (6105816)
- chore(deps): bump filelock from 3.18.0 to 3.20.3 (#231) (6d9dbde)
- chore(deps): bump aiohttp from 3.13.2 to 3.13.3 (#232) (5c7559a)
- ci: parallelize integration tests into matrix jobs (#269) (bd3b7b0)

## [1.4.4] - 2026-03-10

### Added
- feat: add streamDeliveryResources support to memory SDK (#302) (907f816)
- feat: split release workflow into prepare and publish (#301) (17e1357)
- feat: add EvaluationClient with run() for on-demand session evaluation (#300) (102ba0d)

### Fixed
- fix: Session manager batching improvements (#298) (328acba)
- fix: allow custom HTTP status codes from entrypoint handlers (#284) (#296) (2371461)

### Other Changes
- feat(strands-memory): add converter injection and optional restored-tool filtering (#288) (6cda0a3)
- chore: bump version to 1.4.3 (#297) (a29aeec)

## [1.4.3] - 2026-03-04

### Added
- feat: add buffering for agent state events (#295) (9e865da)

### Other Changes
- chore: bump version to 1.4.2 (#294) (bbc00a7)

## [1.4.2] - 2026-03-03

### Added
- feat: Add automatic flush for batched messages, on AfterInvocationEvent hook and interval-based periodical flush (#291) (bb4a1b7)

### Fixed
- fix: fix npe in memory session manager when messages have no text content (#293) (2b6736e)
- fix: AgentCoreMemorySessionManager - Cache agent timestamps to eliminate redundant list_events calls (#289) (1dd896e)

### Other Changes
- chore: bump version to 1.4.1 (#287) (ef448dc)

## [1.4.1] - 2026-02-27

### Other Changes
- chore: bump version to 1.4.0 (#281) (813c2c1)

## [1.4.0] - 2026-02-24

### Added
- feat: add SessionConfiguration with proxy, extensions, and profile support (#274) (ca3c322)

### Other Changes
- chore: bump version to 1.3.2 (#280) (a637826)

## [Unreleased]

### Added
- feat: add SessionConfiguration with proxy, extensions, and profile support for browser sessions (#274)

## [1.3.2] - 2026-02-23

### Added
- feat: configurable context_tag with user_context default (#279) (33f09f7)

### Fixed
- fix: insert retrieved LTM before last user message to avoid prefill error on Claude 4.6+ (#271) (232d05c)

### Other Changes
- test: add thinking-mode compatibility tests for LTM retrieval (#272) (1bd22b7)
- chore: bump version to 1.3.1 (#270) (8d7405c)

## [1.3.1] - 2026-02-17

### Fixed
- fix: use correct boto3 service name for evaluation client (#267) (1e2be1b)

### Documentation
- docs: update memory READMEs with metadata types and message batching (#264) (efea9d4)

### Other Changes
- chore: bump version to 1.3.0 (#263) (208cc14)

## [1.3.0] - 2026-02-11

### Fixed
- fix: download_file/download_files crash on binary content with UnicodeDecodeError (#257) (e8b63be)
- fix: remove deprecated save_turn() and process_turn() methods (#241) (9bd2623)

### Other Changes
- feat(memory): event metadata state identification, message batching, and redundant sync elimination (#244) (fbce2fc)
- fix(identity): update endpoint for Create/UpdateWorkloadIdentity (#249) (3fa9afe)
- chore: bump version to 1.2.1 (#250) (cb44b79)

## [1.2.1] - 2026-02-03

### Fixed
- fix: escape special characters in Slack notification payload (#239) (bcd312f)

### Other Changes
- Add trailing slash to namespace strings (#238) (1de940d)
- feat(memory): add metadata support to MemoryClient events (#236) (53a1baa)
- temp: add Slack notification workflow for new issues (#226) (a48944a)
- chore: bump version to 1.2.0 (#213) (52bc194)

## [1.2.0] - 2026-01-13

### Fixed
- fix: apply relevance_score filtering in Strands integration (#190) (#211) (952b018)

### Other Changes
- fix(memory): Improve pagination behavior in get_last_k_turns() and list_messages() (#209) (2b047ff)
- Add integration_source parameter for framework attribution telemetry (#210) (43c6c3c)
- feat(memory): add episodic memory strategy support (#208) (0df9757)
- chore: bump version to 1.1.4 (#207) (b3e4b4b)

## [1.1.4] - 2026-01-08

### Fixed
- fix: encode bytes before filtering empty text in message_to_payload (#199) (3f01653)

### Other Changes
- test: add unit test for bytes serialization fix in message_to_payload (#205) (a9745ce)
- Release v1.1.3 (#204) (2ec6639)

## [1.1.3] - 2026-01-07

- feat(code-interpreter): Add convenience methods for file operations and package management (#202) (bcdc6eb)

## [1.1.2] - 2025-12-26

### Fixed
- fix: Removed pre-commit from dependencies (#195) (4f8c625)
- fix: dont save empty text messages (breaks Converse API) (#185) (049ccdc)

### Other Changes
- feat(runtime): Add session_id support to WebSocket connection methods (#186) (62d297d)
- chore: bump version to 1.1.1 (#184) (92272e7)

## [1.1.1] - 2025-12-03

### Other Changes
- feat(identity):  Add @requires_iam_access_token decorator for AWS STS JWT tokens (#179) (4ab6072)
- Add Strands AgentCore Evaluation integration (#183) (f242836)
- chore: bump version to 1.1.0 (#182) (042d4bf)

## [1.1.0] - 2025-12-02

### Added
- feat: add websockets as main dependency for @app.websocket decorator (#181) (9146d3e)

### Other Changes
- Feature/bidirectional streaming (#180) (535faa5)
- feat(runtime): Add middleware data support to request context (#178) (95bbfa4)
- chore: bump version to 1.0.7 (#173) (18a78b9)

## [1.0.7] - 2025-11-25

### Added
- feat: parallelize retrieve memories API calls for multiple namespaces to improve latency (#163) (df5a2c9)
- feat: add documentation for metadata support in STM (#156) (67563f1)

### Fixed
- fix: metadata-workflow readme link (#171) (a8536df)

### Other Changes
- chore: bump strands-agents version (#172) (cb98125)
- Allow passing custom parameters to the GetResourceOauth2Token API via SDK decorator (#157) (988ca8f)
- chore: bump version to 1.0.6 (#155) (d1953e8)

## [1.0.6] - 2025-11-10

### Added
- feat: Add control plane CRUD operations and config helpers for browser and code interpreter (#152) (81faca1)
- feat: adding function to delete all memory records in namespace (#148) (72a16be)

### Fixed
- fix: list_events having branch & eventMetadata filter (#153) (70e138d)
- fix: correct workflow output reference for external PR tests (#141) (90f04bf)

### Other Changes
- chore: bump version to 1.0.5 (#144) (1456d03)

## [1.0.5] - 2025-10-29

### Documentation
- docs: update quickstart links to AWS documentation (#138) (b3d49f8)

### Other Changes
- fix(memory): resolve AWS_REGION env var (#143) (7a9a855)
- Chore/workflow improvements (#137) (091dab1)
- chore: enabling batch api pass through to boto3 client methods (#135) (245f3c1)
- chore: bump version to 1.0.4 (#134) (ecba82d)

## [1.0.4] - 2025-10-22

### Added
- feat: support for async llm callback (#131) (1e3fd0c)

### Other Changes
- chore(memory): fix linter issues (#132) (36ea477)
- Add middleware (#121) (f30e281)
- Update Outbound Oauth error message (#119) (a9ad13a)
- Update README.md (#128) (c744ba3)
- chore: bump version to 1.0.3 (#127) (d14d80e)

## [1.0.3] - 2025-10-16

### Fixed
- fix: remove NotRequried as it is supported only in python 3.11 (#125) (806ee26)

### Other Changes
- chore: bump version to 1.0.2 (#126) (11b761a)

## [1.0.2] - 2025-10-16

### Fixed
- fix: remove NotRequried as it is supported only in python 3.11 (#125) (806ee26)

## [1.0.0] - 2025-10-15

### Fixed
- fix: rename list_events parameter include_parent_events to include_parent_branches to match the boto3 parameter (#108) (ee35ade)
- fix: add the include_parent_events parameter to the get_last_k_turns method (#107) (eee67da)
- fix: fix session name typo in get_last_k_turns (#104) (1ba3e1c)

### Documentation
- docs: remove preview verbiage following Bedrock AgentCore GA release (#113) (9d496aa)

### Other Changes
- fix(deps): restrict pydantic to versions below 2.41.3 (#115) (b4a49b9)
- feat(browser): Add viewport configuration support to BrowserClient (#112) (014a6b8)
- chore: bump version to 0.1.7 (#103) (d572d68)

## [0.1.7] - 2025-10-01

### Fixed
- fix: fix validation exception which occurs if the default aws region mismatches with the user's region_name (#102) (207e3e0)

### Other Changes
- chore: bump version to 0.1.6 (#101) (5d5271d)

## [0.1.6] - 2025-10-01

### Added
- feat: Initial commit for Session Manager, Session and Actor constructs (#87) (72e37df)

### Fixed
- fix: swap event_timestamp with branch in add_turns (#99) (0027298)

### Other Changes
- chore: Add README for MemorySessionManager (#100) (9b274a0)
- Feature/boto client config (#98) (107fd53)
- Update README.md (#95) (0c65811)
- Release v0.1.5 (#96) (7948d26)

## [0.1.5] - 2025-09-24

### Other Changes
- Added request header allowlist support (#93) (7377187)
- Remove TestPyPI publishing step from release workflow (#89) (8f9bbf5)
- feat(runtime): add kwargs support to run method (#79) (c61edef)

## [0.1.4] - 2025-09-17

### Other Changes
- feat(runtime): add kwargs support to run method (#79) (c61edef)

## [0.1.3] - 2025-09-05

### Added
- fix/observability logs improvement (#67) (78a5eee)
- feat: add AgentCore Memory Session Manager with Strands Agents (#65) (7f866d9)
- feat: add validation for browser live view URL expiry timeout (#57) (9653a1f)

### Other Changes
- feat(memory): Add passthrough for gmdp and gmcp operations for Memory (#66) (1a85ebe)
- Improve serialization (#60) (00cc7ed)
- feat(memory): add functionality to memory client (#61) (3093768)
- add automated release workflows (#36) (045c34a)
- chore: remove concurrency checks and simplify thread pool handling (#46) (824f43b)
- fix(memory): fix last_k_turns (#62) (970317e)
- use json to manage local workload identity and user id (#37) (5d2fa11)
- fail github actions when coverage threshold is not met (#35) (a15ecb8)

## [0.1.2] - 2025-08-11

### Fixed
- Remove concurrency checks and simplify thread pool handling (#46)

## [0.1.1] - 2025-07-23

### Fixed
- **Identity OAuth2 parameter name** - Fixed incorrect parameter name in GetResourceOauth2Token
  - Changed `callBackUrl` to `resourceOauth2ReturnUrl` for correct API compatibility
  - Ensures proper OAuth2 token retrieval for identity authentication flows

- **Memory client region detection** - Improved region handling in MemoryClient initialization
  - Now follows standard AWS SDK region detection precedence
  - Uses explicit `region_name` parameter when provided
  - Falls back to `boto3.Session().region_name` if not specified
  - Defaults to 'us-west-2' only as last resort

- **JSON response double wrapping** - Fixed duplicate JSONResponse wrapping issue
  - Resolved issue when semaphore acquired limit is reached
  - Prevents malformed responses in high-concurrency scenarios

### Improved
- **JSON serialization consistency** - Enhanced serialization for streaming and non-streaming responses
  - Added new `_safe_serialize_to_json_string` method with progressive fallbacks
  - Handles datetime, Decimal, sets, and Unicode characters consistently
  - Ensures both streaming (SSE) and regular responses use identical serialization logic
  - Improved error handling for non-serializable objects

## [0.1.0] - 2025-07-16

### Added
- Initial release of Bedrock AgentCore Python SDK
- Runtime framework for building AI agents
- Memory client for conversation management
- Authentication decorators for OAuth2 and API keys
- Browser and Code Interpreter tool integrations
- Comprehensive documentation and examples

### Security
- TLS 1.2+ enforcement for all communications
- AWS SigV4 signing for API authentication
- Secure credential handling via AWS credential chain
