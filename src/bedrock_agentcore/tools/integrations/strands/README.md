# AgentCore Web Search for Strands Agents

`AgentCoreWebSearch` exposes the Amazon Bedrock AgentCore Web Search tool as a
[Strands Agents](https://strandsagents.com) tool, so an agent can answer questions about
recent events and cite its sources.

## Overview

- **No search API key** — calls are authenticated with SigV4 from the ambient AWS credentials, so there is no separate search provider account, key rotation or per-key billing
- **Source attribution** — every result carries a title, a URL, a publication date when the index reports one, and an extract, and the tool description asks the model to cite the URLs it used
- **Server-side filtering** — domain include and exclude lists and publication date bounds are applied by the service, not by trimming results afterwards
- **One tool, not a toolkit** — `instance.web_search` is a single Strands tool, so it drops into an existing `Agent(tools=[...])` list

## How it works

Web search reaches the service through an AgentCore Gateway connector target:

```
┌─────────┐     ┌──────────────────┐     ┌──────────────┐     ┌────────────────┐
│  Agent  │────▶│ AgentCoreWebSearch│────▶│   Gateway    │────▶│  web-search    │
│         │     │  (SigV4, MCP)     │     │ connector    │     │  connector     │
│         │◀────│  formats results  │◀────│   target     │◀────│  (AWS managed) │
└─────────┘     └──────────────────┘     └──────────────┘     └────────────────┘
     caller's credentials                  gateway execution role
```

The caller's credentials sign the request to the gateway. The gateway then uses its own
execution role to call the connector, so two different principals need permissions. See
[IAM](#iam) below.

Which transport is used is decided inside `WebSearchClient`, not here. Every gateway
argument on this class is optional and only the ones actually supplied are forwarded, so a
later direct API needing none of them works through the same call.

## Installation

```bash
pip install 'bedrock-agentcore[strands-agents]'
```

## Prerequisites

The web search connector is enabled per account and is offered in a subset of regions. A
gateway with a web search target has to exist before the tool can be used; creating one is
three calls and is done once per account:

```python
from bedrock_agentcore.gateway.client import GatewayClient

client = GatewayClient(region_name="us-east-1")

gateway = client.create_gateway_and_wait(
    name="my-web-search-gateway",
    roleArn="arn:aws:iam::111122223333:role/MyGatewayExecutionRole",
    authorizerType="AWS_IAM",
    protocolType="MCP",
)

client.create_web_search_target(gateway_identifier=gateway["gatewayId"])
```

For the regions the connector is currently offered in, see
[Web Search Tool](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html)
in the AgentCore developer guide.

## Quick start

```python
from strands import Agent

from bedrock_agentcore.tools.integrations.strands import AgentCoreWebSearch

with AgentCoreWebSearch(region="us-east-1", gateway_id="my-web-search-gateway-abc123") as search:
    agent = Agent(tools=[search.web_search])
    agent("What changed in the most recent boto3 release?")
```

Without the context manager, call `close()` when the agent is done:

```python
search = AgentCoreWebSearch(gateway_id="my-web-search-gateway-abc123")
agent = Agent(tools=[search.web_search])
agent("Who maintains urllib3?")
search.close()
```

## Configuration

| Argument | Description |
|---|---|
| `region` | AWS region to call. Defaults to `us-east-1`, except when `gateway_arn` is given, in which case the ARN's region is used. |
| `gateway_id` | ID of a gateway carrying a web search connector target. |
| `gateway_arn` | ARN of that gateway. The ID and the region are read from it. |
| `gateway_endpoint` | A gateway MCP endpoint URL, if one is already known. |
| `target_name` | Name the connector target was created under. Supplying it avoids a tool discovery round trip on the first search. |
| `tool_name` | Fully qualified name of the gateway tool, if already known. |
| `boto3_session` | Session to take credentials from. |
| `client` | A `WebSearchClient` to use as is. The caller keeps ownership of it, so `close()` leaves it open. |

Exactly one of `gateway_id`, `gateway_arn`, `gateway_endpoint` or `client` says where the
search goes.

`region` defaults to `us-east-1` rather than to the session's region, because web search is
not offered everywhere. It is filled in only when no region and no gateway ARN were given:
an explicit region takes precedence over the one in a gateway ARN, so defaulting it
unconditionally would send a `eu-west-1` gateway's traffic to `us-east-1`.

## Tool arguments

The model sees one tool, `web_search`. Only `query` is required.

| Argument | Description |
|---|---|
| `query` | What to search for, as a natural language query. 200 characters or fewer, checked locally. |
| `max_results` | How many results to return, between 1 and 25. Unset means the service default of 10. |
| `include_domains` | Only return results from these domains, up to 100. A root domain also matches its subdomains. |
| `exclude_domains` | Drop results from these domains, up to 100. |
| `published_after` | Only return pages published on or after this ISO-8601 UTC timestamp, inclusive. Applies to web results only. |
| `published_before` | Only return pages published on or before this one, inclusive. Applies to web results only. |

Request filters compose with the target's own domain rules and can never widen them. A
domain is returned only if it appears on every include list that is set, so when the target
was created with an include list, a request-level `include_domains` narrows to the
intersection of the two. If the two share no domains the search returns nothing, and that is
a silent empty result rather than an error.

Target-level lists are applied server side and are not visible to the model, so a model
asking repeatedly for a domain the target excludes gets nothing back and cannot tell why.
Keeping the restriction in the tool arguments instead is what lets it see the boundary it is
working inside.

Request-level filter arguments need connector version 1.2.0 or later on the target. On an
earlier version the tool accepts only `query` and `max_results`.

## IAM

Two permissions on two different principals, and getting this wrong is the most common
setup failure.

The caller's credentials need `bedrock-agentcore:InvokeGateway` on the gateway:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "bedrock-agentcore:InvokeGateway",
    "Resource": "arn:aws:bedrock-agentcore:us-east-1:111122223333:gateway/my-gateway-abc123"
  }]
}
```

The gateway's execution role needs `bedrock-agentcore:InvokeWebSearch` on the connector, and
nothing else:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "bedrock-agentcore:InvokeWebSearch",
    "Resource": "arn:aws:bedrock-agentcore:us-east-1:aws:tool/web-search.v1"
  }]
}
```

Note the `aws` account field in that resource ARN: the connector is service owned, not
account owned.

If the execution role's trust policy narrows `aws:SourceArn` to a gateway in a different
region than the gateway actually created, `CreateGateway` still succeeds and the first
search fails with `Failed to obtain execution role credentials`, which reads like an
entitlement problem rather than an IAM one.

## Errors

A failed search raises rather than returning the message as a result, because a string
saying the search failed is indistinguishable to the model from a search that found nothing:

- `WebSearchError` when the search itself fails, including when the account is not entitled to the connector (`not available for this account`). An unsupported region reports the same message, so check the region before concluding the account lacks the entitlement.
- `ValueError` when an argument is outside the documented limits, or the tool has already been closed.

Neither reaches the caller of `agent(...)`. The Strands tool executor turns a tool
exception into a tool result with `status` of `error` carrying the message, so the agent
keeps running and the model sees that its search failed rather than that the web is empty.
The distinction is the point: an error result is something a model can react to, a
successful result reading "search failed" is not.

An empty result set is not an error. It returns a sentence saying so and naming an
over-narrow filter as the likely cause, so the model can widen the query itself. When a
filtered search keeps coming back empty, check the target's own domain configuration before
the query: a request include list that shares no domains with the target's include list
returns nothing, silently.

## Telemetry

Calls are attributed to Strands through `integration_source="strands"` on the user agent, so
web search usage from Strands agents is distinguishable from usage through the raw SDK.

## Related

- [`WebSearchClient`](../../web_search_client.py) — the underlying client, usable without a framework
- [Web Search Tool](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-connector-web-search-tool.html) — the service documentation
