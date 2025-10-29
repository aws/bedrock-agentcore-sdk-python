# Changelog

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
