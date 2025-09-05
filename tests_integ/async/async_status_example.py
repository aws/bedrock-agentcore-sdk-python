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
    app.logger.info("Starting background data processing...")
    await asyncio.sleep(200)  # Simulate work
    app.logger.info("Background data processing completed")


@app.async_task
async def database_cleanup():
    """Simulate database cleanup task."""
    app.logger.info("Starting database cleanup...")
    await asyncio.sleep(100)  # Simulate work
    app.logger.info("Database cleanup completed")


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
    app.logger.info("Starting BedrockAgentCore app with async status functionality...")
    app.logger.info("Available endpoints:")
    app.logger.info("  GET /ping - Check current ping status")
    app.logger.info("  POST /invocations - Main handler")
    app.logger.info("")
    app.logger.info("Example debug action calls (debug=True is enabled):")
    app.logger.info("  {'_agent_core_app_action': 'ping_status'}")
    app.logger.info("  {'_agent_core_app_action': 'job_status'}")
    app.logger.info("  {'_agent_core_app_action': 'force_healthy'}")
    app.logger.info("  {'_agent_core_app_action': 'force_busy'}")
    app.logger.info("  {'_agent_core_app_action': 'clear_forced_status'}")
    app.logger.info("")
    app.logger.info("Example regular calls:")
    app.logger.info("  {'action': 'start_background_task'}")
    app.logger.info("  {'action': 'get_task_info'}")
    app.logger.info("  {'action': 'force_status', 'ping_status': 'HealthyBusy'}")

    app.run()
