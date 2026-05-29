"""MCP-compatible Lambda handler for AgentCore Gateway integration tests.

This Lambda implements the MCP JSON-RPC protocol over HTTP, responding to:
- initialize: Returns server capabilities
- tools/list: Returns available tool definitions
- tools/call: Executes a tool and returns results

Deploy with Python 3.10+ runtime, handler: lambda_function.lambda_handler
"""

import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]


def lambda_handler(event, context):
    """Handle MCP JSON-RPC requests from AgentCore Gateway."""
    logger.info("Received event: %s", json.dumps(event))

    body = event.get("body", "{}")
    if isinstance(body, str):
        body = json.loads(body)

    method = body.get("method", "")
    request_id = body.get("id")
    params = body.get("params", {})

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": "integ-test-mcp-server",
                "version": "1.0.0",
            },
        }
    elif method == "notifications/initialized":
        # Client acknowledgment, no response needed
        return {"statusCode": 200, "body": ""}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = _handle_tool_call(tool_name, arguments)
    else:
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
            ),
        }

    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }

    return {
        "statusCode": 200,
        "body": json.dumps(response),
    }


def _handle_tool_call(tool_name, arguments):
    """Execute a tool and return MCP-formatted result."""
    if tool_name == "get_weather":
        city = arguments.get("city", "unknown")
        return {"content": [{"type": "text", "text": f"Weather in {city}: 72°F, sunny with light clouds."}]}
    elif tool_name == "send_email":
        to = arguments.get("to", "")
        subject = arguments.get("subject", "")
        return {"content": [{"type": "text", "text": f"Email sent to {to} with subject: {subject}"}]}
    else:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            "isError": True,
        }
