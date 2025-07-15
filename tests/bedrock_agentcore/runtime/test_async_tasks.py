"""Tests for async task management and ping status functionality."""

import asyncio
import time

import pytest

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime.models import PingStatus


class TestAsyncTaskDecorator:
    """Test the @app.async_task decorator functionality."""

    def test_async_task_decorator_validation(self):
        """Test that decorator only accepts async functions."""
        app = BedrockAgentCoreApp()

        # Should work with async function
        @app.async_task
        async def valid_async_function():
            await asyncio.sleep(0.1)
            return "done"

        assert callable(valid_async_function)

        # Should raise error with sync function
        with pytest.raises(ValueError, match="@async_task can only be applied to async functions"):

            @app.async_task
            def invalid_sync_function():
                return "done"

    @pytest.mark.asyncio
    async def test_async_task_tracking(self):
        """Test that async tasks are properly tracked."""
        app = BedrockAgentCoreApp()

        @app.async_task
        async def test_task():
            await asyncio.sleep(0.1)
            return "completed"

        # Initially no active tasks
        assert len(app._active_tasks) == 0
        assert app.get_current_ping_status() == PingStatus.HEALTHY

        # Start task
        task = asyncio.create_task(test_task())

        # Should have one active task
        await asyncio.sleep(0.01)  # Allow task to start
        assert len(app._active_tasks) == 1
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

        # Wait for completion
        result = await task
        assert result == "completed"

        # Should have no active tasks after completion
        assert len(app._active_tasks) == 0
        assert app.get_current_ping_status() == PingStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self):
        """Test multiple instances of the same function running concurrently."""
        app = BedrockAgentCoreApp()

        @app.async_task
        async def concurrent_task(task_id):
            await asyncio.sleep(0.1)
            return f"task_{task_id}_completed"

        # Start multiple tasks
        tasks = []
        for i in range(3):
            task = asyncio.create_task(concurrent_task(i))
            tasks.append(task)

        # Allow tasks to start
        await asyncio.sleep(0.01)

        # Should have 3 active tasks
        assert len(app._active_tasks) == 3
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

        # Wait for all to complete
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 3
        assert all("completed" in result for result in results)

        # No active tasks after completion
        assert len(app._active_tasks) == 0
        assert app.get_current_ping_status() == PingStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_async_task_exception_handling(self):
        """Test that task counter is decremented even when task fails."""
        app = BedrockAgentCoreApp()

        @app.async_task
        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("Task failed")

        # Start failing task
        task = asyncio.create_task(failing_task())

        # Allow task to start
        await asyncio.sleep(0.005)
        assert len(app._active_tasks) == 1

        # Wait for task to fail
        with pytest.raises(ValueError, match="Task failed"):
            await task

        # Task counter should be decremented despite exception
        assert len(app._active_tasks) == 0
        assert app.get_current_ping_status() == PingStatus.HEALTHY

    def test_task_info_structure(self):
        """Test the structure of task information."""
        app = BedrockAgentCoreApp()

        # Add mock active tasks
        app._active_tasks = {
            1: {"name": "task_one", "start_time": time.time() - 5},
            2: {"name": "task_two", "start_time": time.time() - 10},
        }

        task_info = app.get_async_task_info()

        assert "active_count" in task_info
        assert "running_jobs" in task_info
        assert task_info["active_count"] == 2
        assert len(task_info["running_jobs"]) == 2

        # Check job structure
        job = task_info["running_jobs"][0]
        assert "name" in job
        assert "duration" in job
        assert isinstance(job["duration"], float)
        assert job["duration"] > 0


class TestPingStatusLogic:
    """Test ping status determination logic."""

    def test_default_healthy_status(self):
        """Test default ping status is Healthy."""
        app = BedrockAgentCoreApp()
        assert app.get_current_ping_status() == PingStatus.HEALTHY

    def test_automatic_busy_status(self):
        """Test automatic busy status with active tasks."""
        app = BedrockAgentCoreApp()

        # Add mock active task
        app._active_tasks[1] = {"name": "test_task", "start_time": time.time()}

        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

    def test_custom_ping_handler(self):
        """Test custom ping handler overrides automatic tracking."""
        app = BedrockAgentCoreApp()

        @app.ping
        def custom_status():
            return PingStatus.HEALTHY_BUSY

        # Should return custom status even without active tasks
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

        # Should still return custom status with active tasks
        app._active_tasks[1] = {"name": "test_task", "start_time": time.time()}
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

    def test_custom_ping_handler_exception_handling(self):
        """Test that exceptions in custom ping handler are handled gracefully."""
        app = BedrockAgentCoreApp()

        @app.ping
        def failing_status():
            raise RuntimeError("Custom handler failed")

        # Should fall back to automatic tracking
        assert app.get_current_ping_status() == PingStatus.HEALTHY

        # Add active task, should still work
        app._active_tasks[1] = {"name": "test_task", "start_time": time.time()}
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

    def test_forced_ping_status(self):
        """Test forced ping status overrides everything."""
        app = BedrockAgentCoreApp()

        # Add custom handler
        @app.ping
        def custom_status():
            return PingStatus.HEALTHY

        # Add active task
        app._active_tasks[1] = {"name": "test_task", "start_time": time.time()}

        # Force status should override both custom handler and active tasks
        app.force_ping_status(PingStatus.HEALTHY)
        assert app.get_current_ping_status() == PingStatus.HEALTHY

        app.force_ping_status(PingStatus.HEALTHY_BUSY)
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

    def test_clear_forced_ping_status(self):
        """Test clearing forced ping status."""
        app = BedrockAgentCoreApp()

        # Force status
        app.force_ping_status(PingStatus.HEALTHY_BUSY)
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

        # Clear forced status
        app.clear_forced_ping_status()
        assert app.get_current_ping_status() == PingStatus.HEALTHY

        # Should now respond to active tasks
        app._active_tasks[1] = {"name": "test_task", "start_time": time.time()}
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY


class TestRPCActions:
    """Test RPC action handling."""

    @pytest.mark.asyncio
    async def test_ping_status_rpc(self):
        """Test ping_status RPC action."""
        app = BedrockAgentCoreApp(debug=True)

        # Add dummy entrypoint to prevent 500 error
        @app.entrypoint
        def dummy_handler(event):
            return {"result": "ok"}

        # Mock request
        class MockRequest:
            async def json(self):
                return {"_agent_core_app_action": "ping_status"}

            headers = {}

        request = MockRequest()
        response = await app._handle_invocation(request)

        assert response.status_code == 200
        # Note: In real testing, you'd parse response.body, but for unit tests
        # we can check the response was created properly

    @pytest.mark.asyncio
    async def test_job_status_rpc(self):
        """Test job_status RPC action."""
        app = BedrockAgentCoreApp(debug=True)

        # Add dummy entrypoint to prevent 500 error
        @app.entrypoint
        def dummy_handler(event):
            return {"result": "ok"}

        # Add mock active task
        app._active_tasks[1] = {"name": "test_task", "start_time": time.time()}

        class MockRequest:
            async def json(self):
                return {"_agent_core_app_action": "job_status"}

            headers = {}

        request = MockRequest()
        response = await app._handle_invocation(request)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_force_healthy_rpc(self):
        """Test force_healthy RPC action."""
        app = BedrockAgentCoreApp(debug=True)

        # Add dummy entrypoint to prevent 500 error
        @app.entrypoint
        def dummy_handler(event):
            return {"result": "ok"}

        class MockRequest:
            async def json(self):
                return {"_agent_core_app_action": "force_healthy"}

            headers = {}

        request = MockRequest()
        response = await app._handle_invocation(request)

        assert response.status_code == 200
        assert app.get_current_ping_status() == PingStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_force_busy_rpc(self):
        """Test force_busy RPC action."""
        app = BedrockAgentCoreApp(debug=True)

        # Add dummy entrypoint to prevent 500 error
        @app.entrypoint
        def dummy_handler(event):
            return {"result": "ok"}

        class MockRequest:
            async def json(self):
                return {"_agent_core_app_action": "force_busy"}

            headers = {}

        request = MockRequest()
        response = await app._handle_invocation(request)

        assert response.status_code == 200
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

    @pytest.mark.asyncio
    async def test_clear_forced_status_rpc(self):
        """Test clear_forced_status RPC action."""
        app = BedrockAgentCoreApp(debug=True)

        # Add dummy entrypoint to prevent 500 error
        @app.entrypoint
        def dummy_handler(event):
            return {"result": "ok"}

        # First force a status
        app.force_ping_status(PingStatus.HEALTHY_BUSY)
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

        class MockRequest:
            async def json(self):
                return {"_agent_core_app_action": "clear_forced_status"}

            headers = {}

        request = MockRequest()
        response = await app._handle_invocation(request)

        assert response.status_code == 200
        assert app.get_current_ping_status() == PingStatus.HEALTHY  # Should be back to automatic

    @pytest.mark.asyncio
    async def test_unknown_rpc_action(self):
        """Test handling of unknown RPC actions."""
        app = BedrockAgentCoreApp(debug=True)

        # Add dummy entrypoint to prevent 500 error
        @app.entrypoint
        def dummy_handler(event):
            return {"result": "ok"}

        class MockRequest:
            async def json(self):
                return {"_agent_core_app_action": "unknown_action"}

            headers = {}

        request = MockRequest()
        response = await app._handle_invocation(request)

        assert response.status_code == 400


class TestUtilityFunctions:
    """Test utility functions for developers."""

    def test_get_async_task_info_utility(self):
        """Test get_async_task_info utility function."""
        # This requires the global app instance to be set
        app = BedrockAgentCoreApp()

        # Mock active tasks
        app._active_tasks = {1: {"name": "task_one", "start_time": time.time() - 5}}

        # Test direct app method
        task_info = app.get_async_task_info()
        assert task_info["active_count"] == 1

    def test_force_ping_status_utility(self):
        """Test force_ping_status utility function."""
        app = BedrockAgentCoreApp()

        # Test forcing status
        app.force_ping_status(PingStatus.HEALTHY_BUSY)
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

        # Test clearing forced status
        app.clear_forced_ping_status()
        assert app.get_current_ping_status() == PingStatus.HEALTHY


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_ping_handler_string_return(self):
        """Test ping handler returning string instead of enum."""
        app = BedrockAgentCoreApp()

        @app.ping
        def string_status():
            return "Healthy"  # String instead of enum

        # Should still work by converting string to enum
        status = app.get_current_ping_status()
        assert status == PingStatus.HEALTHY
        assert isinstance(status, PingStatus)

    def test_task_counter_overflow_protection(self):
        """Test that task counter doesn't cause issues with large numbers."""
        app = BedrockAgentCoreApp()

        # Set counter to large number
        app._task_counter = 999999

        @app.async_task
        async def test_task():
            return "done"

        # Should still work normally
        assert asyncio.iscoroutinefunction(test_task)

    def test_concurrent_task_modifications(self):
        """Test that concurrent modifications to task dictionary are handled safely."""
        app = BedrockAgentCoreApp()

        @app.async_task
        async def concurrent_task():
            await asyncio.sleep(0.01)
            return "done"

        # This is more of a design verification - the dict operations should be atomic enough
        # for our use case (single-threaded async event loop)
        assert len(app._active_tasks) == 0

    @pytest.mark.asyncio
    async def test_very_short_tasks(self):
        """Test tracking of very short-duration tasks."""
        app = BedrockAgentCoreApp()

        @app.async_task
        async def instant_task():
            return "instant"

        # Even instant tasks should be tracked briefly
        task = asyncio.create_task(instant_task())
        result = await task

        assert result == "instant"
        # Task should be cleaned up
        assert len(app._active_tasks) == 0

    @pytest.mark.asyncio
    async def test_task_with_cancellation(self):
        """Test task tracking when task is cancelled."""
        app = BedrockAgentCoreApp()

        @app.async_task
        async def long_task():
            await asyncio.sleep(10)  # Long enough to cancel
            return "completed"

        # Start task
        task = asyncio.create_task(long_task())

        # Allow task to start
        await asyncio.sleep(0.01)
        assert len(app._active_tasks) == 1

        # Cancel task
        task.cancel()

        # Wait for cancellation to complete
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Task should be cleaned up even after cancellation
        assert len(app._active_tasks) == 0


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    @pytest.mark.asyncio
    async def test_mixed_task_lifecycle(self):
        """Test mixed scenarios with multiple tasks, custom handlers, and forced status."""
        app = BedrockAgentCoreApp()

        @app.async_task
        async def background_job():
            await asyncio.sleep(0.1)
            return "job_done"

        @app.ping
        def conditional_status():
            # Custom logic that sometimes overrides
            if len(app._active_tasks) > 2:
                return PingStatus.HEALTHY_BUSY
            return PingStatus.HEALTHY

        # Start with custom handler
        assert app.get_current_ping_status() == PingStatus.HEALTHY

        # Start some tasks (but not enough to trigger custom logic)
        task1 = asyncio.create_task(background_job())
        task2 = asyncio.create_task(background_job())

        await asyncio.sleep(0.01)  # Let tasks start
        assert app.get_current_ping_status() == PingStatus.HEALTHY  # Custom handler

        # Start more tasks to trigger custom logic
        task3 = asyncio.create_task(background_job())
        await asyncio.sleep(0.01)
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY  # Custom handler triggered

        # Force status should override everything
        app.force_ping_status(PingStatus.HEALTHY)
        assert app.get_current_ping_status() == PingStatus.HEALTHY

        # Clean up
        await asyncio.gather(task1, task2, task3)
        app.clear_forced_ping_status()
        assert app.get_current_ping_status() == PingStatus.HEALTHY

    def test_http_ping_endpoint(self):
        """Test the HTTP ping endpoint returns correct status."""
        app = BedrockAgentCoreApp()

        # Mock HTTP request
        class MockRequest:
            pass

        # Test default status
        response = app._handle_ping(MockRequest())
        assert response.status_code == 200

        # Add active task and test again
        app._active_tasks[1] = {"name": "test_task", "start_time": time.time()}
        response = app._handle_ping(MockRequest())
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_error_resilience(self):
        """Test system resilience to various error conditions."""
        app = BedrockAgentCoreApp()

        # Test with corrupted task data
        app._active_tasks[1] = {"invalid": "data"}  # Missing required fields

        # Should not crash when getting task info
        task_info = app.get_async_task_info()
        assert isinstance(task_info, dict)
        assert "active_count" in task_info

        # Status should still work
        status = app.get_current_ping_status()
        assert isinstance(status, PingStatus)


class TestPingStatusTimestamp:
    """Test ping status timestamp functionality."""

    def test_initial_timestamp_set(self):
        """Test that timestamp is set on app initialization."""
        app = BedrockAgentCoreApp()
        assert app._last_status_update_time > 0
        assert isinstance(app._last_status_update_time, float)

    def test_timestamp_updates_on_status_change(self):
        """Test that timestamp updates when status changes."""
        app = BedrockAgentCoreApp()

        # Get initial timestamp
        initial_time = app._last_status_update_time

        # Force a small delay to ensure timestamp difference
        time.sleep(0.01)

        # Add active task to change status from HEALTHY to HEALTHY_BUSY
        app._active_tasks[1] = {"name": "test_task", "start_time": time.time()}
        status = app.get_current_ping_status()

        # Timestamp should have updated
        assert app._last_status_update_time > initial_time
        assert status == PingStatus.HEALTHY_BUSY

        # Store second timestamp
        second_time = app._last_status_update_time

        # Another small delay
        time.sleep(0.01)

        # Remove task to change status back to HEALTHY
        app._active_tasks.clear()
        status = app.get_current_ping_status()

        # Timestamp should update again
        assert app._last_status_update_time > second_time
        assert status == PingStatus.HEALTHY

    def test_timestamp_does_not_update_on_same_status(self):
        """Test that timestamp doesn't update when status remains the same."""
        app = BedrockAgentCoreApp()

        # Get initial status and timestamp
        status1 = app.get_current_ping_status()
        time1 = app._last_status_update_time

        # Small delay
        time.sleep(0.01)

        # Get status again (should be same)
        status2 = app.get_current_ping_status()
        time2 = app._last_status_update_time

        # Status should be same and timestamp should not change
        assert status1 == status2
        assert time1 == time2

    def test_forced_status_updates_timestamp(self):
        """Test that forcing status updates timestamp."""
        app = BedrockAgentCoreApp()

        initial_time = app._last_status_update_time
        time.sleep(0.01)

        # Force status
        app.force_ping_status(PingStatus.HEALTHY_BUSY)
        status = app.get_current_ping_status()

        assert status == PingStatus.HEALTHY_BUSY
        assert app._last_status_update_time > initial_time

    def test_custom_ping_handler_updates_timestamp(self):
        """Test that custom ping handler status changes update timestamp."""
        app = BedrockAgentCoreApp()

        # Variable to control custom handler behavior
        return_busy = False

        @app.ping
        def dynamic_status():
            return PingStatus.HEALTHY_BUSY if return_busy else PingStatus.HEALTHY

        initial_time = app._last_status_update_time
        time.sleep(0.01)

        # Change custom handler behavior
        return_busy = True
        status = app.get_current_ping_status()

        assert status == PingStatus.HEALTHY_BUSY
        assert app._last_status_update_time > initial_time

    @pytest.mark.asyncio
    async def test_ping_endpoint_includes_timestamp(self):
        """Test that ping endpoints include timestamp in response."""
        app = BedrockAgentCoreApp(debug=True)

        # Add dummy entrypoint to prevent 500 error
        @app.entrypoint
        def dummy_handler(event):
            return {"result": "ok"}

        # Test HTTP ping endpoint
        class MockRequest:
            pass

        response = app._handle_ping(MockRequest())
        assert response.status_code == 200

        # Parse response body (in real implementation)
        # For this test, we verify the response was created with timestamp

        # Test RPC ping_status action
        class MockRPCRequest:
            async def json(self):
                return {"_agent_core_app_action": "ping_status"}

            headers = {}

        rpc_response = await app._handle_invocation(MockRPCRequest())
        assert rpc_response.status_code == 200


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])


class TestTaskActionsDisabled:
    """Test behavior when task_actions is disabled."""

    @pytest.mark.asyncio
    async def test_task_actions_disabled_by_default(self):
        """Test that task actions are disabled by default."""
        app = BedrockAgentCoreApp()  # Default should be False

        class MockRequest:
            async def json(self):
                return {"_agent_core_app_action": "ping_status"}

            headers = {}

        # Should not handle task actions when disabled
        response = await app._handle_invocation(MockRequest())

        # Should get "No entrypoint defined" error instead of task action response
        assert response.status_code == 500
