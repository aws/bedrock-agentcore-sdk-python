#!/usr/bin/env python3
"""
Example demonstrating the async status functionality in Bedrock AgentCore SDK.

This example shows how to:
1. Use @app.async_task decorator for automatic status tracking
2. Use @app.ping decorator for custom ping status logic
3. Use debug actions to query and control ping status (debug=True enabled)
4. Use utility functions to inspect and control task status

"""

import asyncio

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime.models import PingStatus

app = BedrockAgentCoreApp(debug=True)


# Example 1: Async task that will automatically set status to "HealthyBusy"
@app.async_task
async def background_data_processing():
    """Simulate a long-running background task."""
    print("Starting background data processing...")
    await asyncio.sleep(200)  # Simulate work
    print("Background data processing completed")


@app.async_task
async def database_cleanup():
    """Simulate database cleanup task."""
    print("Starting database cleanup...")
    await asyncio.sleep(100)  # Simulate work
    print("Database cleanup completed")


# Main entrypoint
@app.entrypoint
async def handler(event):
    """Main handler that demonstrates various features.

    Note: Debug actions (_agent_core_app_action) are handled automatically
    by the framework and never reach this handler function.
    """

    # Regular business logic
    action = event.get("action", "info")

    if action == "start_background_task":
        # Start a background task - ping status will automatically become "HealthyBusy"
        asyncio.create_task(background_data_processing())
        return {"message": "Background task started", "status": "task_started"}

    elif action == "start_multiple_tasks":
        # Start multiple background tasks
        asyncio.create_task(background_data_processing())
        asyncio.create_task(database_cleanup())
        return {"message": "Multiple background tasks started", "status": "tasks_started"}

    elif action == "get_task_info":
        # Use app method to get task information
        task_info = app.get_async_task_info()
        return {"message": "Current task information", "task_info": task_info}

    elif action == "force_status":
        # Demonstrate forcing ping status
        status = event.get("ping_status", "Healthy")
        if status == "Healthy":
            app.force_ping_status(PingStatus.HEALTHY)
        elif status == "HealthyBusy":
            app.force_ping_status(PingStatus.HEALTHY_BUSY)

        return {"message": f"Ping status forced to {status}"}

    else:
        return {
            "message": "BedrockAgentCore Async Status Demo",
            "available_actions": ["start_background_task", "start_multiple_tasks", "get_task_info", "force_status"],
            "debug_actions": ["ping_status", "job_status", "force_healthy", "force_busy", "clear_forced_status"],
        }


if __name__ == "__main__":
    # For local testing
    print("Starting BedrockAgentCore app with async status functionality...")
    print("Available endpoints:")
    print("  GET /ping - Check current ping status")
    print("  POST /invocations - Main handler")
    print("")
    print("Example debug action calls (debug=True is enabled):")
    print("  {'_agent_core_app_action': 'ping_status'}")
    print("  {'_agent_core_app_action': 'job_status'}")
    print("  {'_agent_core_app_action': 'force_healthy'}")
    print("  {'_agent_core_app_action': 'force_busy'}")
    print("  {'_agent_core_app_action': 'clear_forced_status'}")
    print("")
    print("Example regular calls:")
    print("  {'action': 'start_background_task'}")
    print("  {'action': 'get_task_info'}")
    print("  {'action': 'force_status', 'ping_status': 'HealthyBusy'}")

    app.run()
