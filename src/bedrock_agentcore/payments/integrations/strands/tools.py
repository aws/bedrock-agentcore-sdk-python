"""Shared utilities for payment instrument tools."""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def validate_required_params(
    params: dict[str, Any],
    required: list[str],
    optional: Optional[list[str]] = None,
) -> Optional[dict[str, str]]:
    """Validate required and optional parameters.

    Args:
        params: Dictionary of parameters to validate
        required: List of required parameter names
        optional: List of optional parameter names (if provided, will be validated)

    Returns:
        Error dict if validation fails, None if valid
    """
    # Check required parameters
    for param in required:
        if param not in params:
            error_dict = {
                "error": "ValidationError",
                "message": f"Missing required parameter: {param}",
            }
            logger.warning("Validation error: %s", error_dict["message"])
            return error_dict

        if isinstance(params[param], str) and not params[param].strip():
            error_dict = {
                "error": "ValidationError",
                "message": f"Parameter cannot be empty: {param}",
            }
            logger.warning("Validation error: %s", error_dict["message"])
            return error_dict

    # Check optional parameters if provided
    if optional:
        for param in optional:
            if param in params and isinstance(params[param], str):
                if not params[param].strip():
                    error_dict = {
                        "error": "ValidationError",
                        "message": f"Parameter cannot be empty: {param}",
                    }
                    logger.warning("Validation error: %s", error_dict["message"])
                    return error_dict

    return None


def format_error_response(tool_use_id: str, exception: Exception) -> dict[str, Any]:
    """Format exception as error response.

    Args:
        tool_use_id: Tool use ID from Strands
        exception: Exception to format

    Returns:
        ToolResult dict with error status
    """
    error_dict = {
        "error": exception.__class__.__name__,
        "message": str(exception),
    }
    logger.error("Tool error: %s - %s", error_dict["error"], error_dict["message"])
    return {
        "toolUseId": tool_use_id,
        "status": "error",
        "content": [{"text": json.dumps(error_dict)}],
    }


def format_success_response(tool_use_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Format data as success response.

    Args:
        tool_use_id: Tool use ID from Strands
        data: Data to return in response

    Returns:
        ToolResult dict with success status
    """
    logger.info("Tool execution successful for tool_use_id: %s", tool_use_id)
    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": json.dumps(data)}],
    }
