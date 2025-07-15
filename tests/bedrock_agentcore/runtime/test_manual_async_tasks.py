"""Tests for manual async task management and edge case coverage."""

import asyncio
import json
import time
from unittest.mock import Mock, patch

import pytest

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime.models import PingStatus


class TestManualAsyncTaskManagement:
    """Test manual async task management functionality."""

    def test_add_async_task_with_metadata(self):
        """Test add_async_task with metadata parameter."""
        app = BedrockAgentCoreApp()

        # Test with metadata
        metadata = {"file": "data.csv", "priority": "high"}
        task_id = app.add_async_task("file_processing", metadata)

        assert isinstance(task_id, int)
        assert len(app._active_tasks) == 1

        # Verify metadata is stored
        task_info = app._active_tasks[task_id]
        assert task_info["name"] == "file_processing"
        assert task_info["metadata"] == metadata
        assert "start_time" in task_info

    def test_add_async_task_without_metadata(self):
        """Test add_async_task without metadata parameter."""
        app = BedrockAgentCoreApp()

        task_id = app.add_async_task("simple_task")

        assert isinstance(task_id, int)
        assert len(app._active_tasks) == 1

        # Verify no metadata key when not provided
        task_info = app._active_tasks[task_id]
        assert task_info["name"] == "simple_task"
        assert "metadata" not in task_info

    def test_complete_unknown_task_id(self):
        """Test completing a task ID that doesn't exist."""
        app = BedrockAgentCoreApp()

        # Try to complete non-existent task
        result = app.complete_async_task(999999)

        assert result is False
        assert len(app._active_tasks) == 0

    def test_complete_async_task_success(self):
        """Test successful task completion."""
        app = BedrockAgentCoreApp()

        task_id = app.add_async_task("test_task")
        assert len(app._active_tasks) == 1

        result = app.complete_async_task(task_id)

        assert result is True
        assert len(app._active_tasks) == 0

    def test_get_async_task_info_with_corrupted_data(self):
        """Test get_async_task_info handles corrupted task data gracefully."""
        app = BedrockAgentCoreApp()

        # Add corrupted task data (missing required fields)
        app._active_tasks[1] = {"invalid": "data"}  # Missing name and start_time
        app._active_tasks[2] = {"name": "valid_task", "start_time": time.time()}
        app._active_tasks[3] = {"name": "bad_time", "start_time": "not_a_number"}

        # Should handle corrupted data gracefully
        task_info = app.get_async_task_info()

        assert isinstance(task_info, dict)
        assert "active_count" in task_info
        assert "running_jobs" in task_info
        assert task_info["active_count"] == 3  # All tasks counted

        # Only valid jobs should be in running_jobs
        valid_jobs = [job for job in task_info["running_jobs"] if "name" in job and "duration" in job]
        assert len(valid_jobs) <= 2  # At most 2 valid jobs


class TestErrorHandlingScenarios:
    """Test error handling and exception scenarios."""

    @pytest.mark.asyncio
    async def test_invocation_with_malformed_json(self):
        """Test handling of malformed JSON in invocation requests."""
        app = BedrockAgentCoreApp()

        @app.entrypoint
        def test_handler(event):
            return {"result": "ok"}

        # Mock request with invalid JSON
        class MockBadJSONRequest:
            async def json(self):
                raise json.JSONDecodeError("Invalid JSON", "test", 0)

            headers = {}

        request = MockBadJSONRequest()
        response = await app._handle_invocation(request)

        assert response.status_code == 400

    def test_ping_endpoint_exception_handling(self):
        """Test ping endpoint handles exceptions gracefully."""
        app = BedrockAgentCoreApp()

        # Mock get_current_ping_status to raise exception
        with patch.object(app, "get_current_ping_status", side_effect=RuntimeError("Ping failed")):
            response = app._handle_ping(Mock())

            assert response.status_code == 200  # Should return fallback response

    @pytest.mark.asyncio
    async def test_debug_action_exception_handling(self):
        """Test debug action exception handling."""
        app = BedrockAgentCoreApp(debug=True)

        @app.entrypoint
        def test_handler(event):
            return {"result": "ok"}

        # Mock force_ping_status to raise exception
        with patch.object(app, "force_ping_status", side_effect=RuntimeError("Force failed")):

            class MockRequest:
                async def json(self):
                    return {"_agent_core_app_action": "force_healthy"}

                headers = {}

            response = await app._handle_invocation(MockRequest())
            assert response.status_code == 500

    def test_sse_chunk_normal_serialization(self):
        """Test normal SSE chunk serialization."""
        app = BedrockAgentCoreApp()

        # Test with dict
        data = {"message": "hello", "count": 42}
        result = app._convert_to_sse(data)
        assert result == b'data: {"message": "hello", "count": 42}\n\n'

        # Test with string (now sent as plain text, not JSON-encoded)
        result = app._convert_to_sse("simple string")
        assert result == b'data: "simple string"\n\n'

    def test_custom_ping_handler_result_assignment(self):
        """Test custom ping handler result assignment."""
        app = BedrockAgentCoreApp()

        @app.ping
        def custom_handler():
            return "HealthyBusy"  # String that needs conversion

        status = app.get_current_ping_status()
        assert status == PingStatus.HEALTHY_BUSY


class TestStreamingAndAuthentication:
    """Test streaming responses and authentication handling."""

    @pytest.mark.asyncio
    async def test_streaming_generator_response(self):
        """Test streaming response with generator."""
        app = BedrockAgentCoreApp()

        def generator_handler(event):
            yield {"chunk": 1}
            yield {"chunk": 2}
            yield {"chunk": 3}

        @app.entrypoint
        def test_handler(event):
            return generator_handler(event)

        class MockRequest:
            async def json(self):
                return {"test": "data"}

            headers = {}

        response = await app._handle_invocation(MockRequest())

        # Should return StreamingResponse
        assert hasattr(response, "media_type")
        assert response.media_type == "text/event-stream"

    @pytest.mark.asyncio
    async def test_streaming_async_generator_response(self):
        """Test streaming response with async generator."""
        app = BedrockAgentCoreApp()

        async def async_generator_handler(event):
            yield {"chunk": 1}
            yield {"chunk": 2}
            yield {"chunk": 3}

        @app.entrypoint
        async def test_handler(event):
            return async_generator_handler(event)

        class MockRequest:
            async def json(self):
                return {"test": "data"}

            headers = {}

        response = await app._handle_invocation(MockRequest())

        # Should return StreamingResponse
        assert hasattr(response, "media_type")
        assert response.media_type == "text/event-stream"

    @pytest.mark.asyncio
    async def test_authentication_token_handling(self):
        """Test authentication token setting."""
        app = BedrockAgentCoreApp()

        @app.entrypoint
        def test_handler(event, context):
            # Return context to verify it was set
            return {"context_set": context is not None}

        class MockRequest:
            async def json(self):
                return {"test": "data"}

            headers = {"X-Agent-Access-Token": "test-token-123"}

        # Test that handler with context parameter gets called
        response = await app._handle_invocation(MockRequest())
        assert response.status_code == 200

        # Test authentication token extraction
        token = MockRequest().headers.get("X-Agent-Access-Token")
        assert token == "test-token-123"

    @pytest.mark.asyncio
    async def test_no_task_action_return_path(self):
        """Test task action return path when no action is present."""
        app = BedrockAgentCoreApp(debug=True)

        @app.entrypoint
        def test_handler(event):
            return {"result": "ok"}

        class MockRequest:
            async def json(self):
                return {"normal": "request"}  # No _agent_core_app_action

            headers = {}

        # Should return None from _handle_task_action and proceed normally
        response = await app._handle_invocation(MockRequest())
        assert response.status_code == 200


class TestIntegrationScenarios:
    """Test integration scenarios with multiple features."""

    def test_mixed_manual_and_decorator_tasks(self):
        """Test mixing manual task management with decorator tasks."""
        app = BedrockAgentCoreApp()

        @app.async_task
        async def decorated_task():
            await asyncio.sleep(0.01)
            return "decorated_done"

        # Add manual task
        manual_task_id = app.add_async_task("manual_task", {"type": "manual"})

        # Should have one manual task
        assert len(app._active_tasks) == 1
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

        # Complete manual task
        app.complete_async_task(manual_task_id)
        assert len(app._active_tasks) == 0
        assert app.get_current_ping_status() == PingStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_concurrent_task_management(self):
        """Test concurrent manual task operations."""
        app = BedrockAgentCoreApp()

        # Add multiple tasks concurrently (simulated)
        task_ids = []
        for i in range(5):
            task_id = app.add_async_task(f"task_{i}", {"index": i})
            task_ids.append(task_id)

        assert len(app._active_tasks) == 5
        assert app.get_current_ping_status() == PingStatus.HEALTHY_BUSY

        # Complete tasks
        for task_id in task_ids:
            result = app.complete_async_task(task_id)
            assert result is True

        assert len(app._active_tasks) == 0
        assert app.get_current_ping_status() == PingStatus.HEALTHY

    def test_task_id_uniqueness(self):
        """Test that task IDs are unique."""
        app = BedrockAgentCoreApp()

        task_ids = set()
        for i in range(100):
            task_id = app.add_async_task(f"task_{i}")
            assert task_id not in task_ids
            task_ids.add(task_id)

        # All task IDs should be unique
        assert len(task_ids) == 100
        assert len(app._active_tasks) == 100

    def test_task_lifecycle_logging(self):
        """Test that task lifecycle generates appropriate log messages."""
        app = BedrockAgentCoreApp()

        with patch.object(app.logger, "info") as mock_info:
            # Add task
            task_id = app.add_async_task("logged_task")

            # Complete task
            app.complete_async_task(task_id)

            # Verify logging calls
            assert mock_info.call_count >= 2  # At least start and complete messages

    @pytest.mark.asyncio
    async def test_error_resilience_with_active_tasks(self):
        """Test system resilience when errors occur with active tasks."""
        app = BedrockAgentCoreApp()

        # Add some tasks
        task_id1 = app.add_async_task("task1")
        task_id2 = app.add_async_task("task2")

        # Corrupt one task's data
        app._active_tasks[task_id1] = {"corrupted": "data"}

        # System should still function
        ping_status = app.get_current_ping_status()
        assert ping_status == PingStatus.HEALTHY_BUSY

        task_info = app.get_async_task_info()
        assert task_info["active_count"] == 2

        # Clean completion should still work for valid tasks
        result = app.complete_async_task(task_id2)
        assert result is True


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_task_completion_race_condition_simulation(self):
        """Test task completion under simulated race conditions."""
        app = BedrockAgentCoreApp()

        task_id = app.add_async_task("race_task")

        # Simulate race condition by completing twice
        result1 = app.complete_async_task(task_id)
        result2 = app.complete_async_task(task_id)

        assert result1 is True  # First completion succeeds
        assert result2 is False  # Second completion fails

    def test_large_metadata_handling(self):
        """Test handling of large metadata objects."""
        app = BedrockAgentCoreApp()

        # Create large metadata
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        task_id = app.add_async_task("large_meta_task", large_metadata)

        # Should handle large metadata without issues
        task_info = app._active_tasks[task_id]
        assert task_info["metadata"] == large_metadata

        # Cleanup
        app.complete_async_task(task_id)

    def test_task_duration_calculation_accuracy(self):
        """Test accuracy of task duration calculations."""
        app = BedrockAgentCoreApp()
        task_id = app.add_async_task("duration_test")

        # Wait a bit
        time.sleep(0.1)

        task_info = app.get_async_task_info()
        job = task_info["running_jobs"][0]

        expected_min_duration = 0.05  # At least 50ms
        assert job["duration"] >= expected_min_duration

        app.complete_async_task(task_id)

    @pytest.mark.asyncio
    async def test_context_parameter_detection(self):
        """Test detection of context parameter in handlers."""
        app = BedrockAgentCoreApp()

        @app.entrypoint
        def handler_with_context(event, context):
            return {"has_context": True}

        @app.entrypoint
        def handler_without_context(event):
            return {"has_context": False}

        # Test with context handler
        app.handlers["main"] = handler_with_context
        assert app._takes_context(handler_with_context) is True

        # Test without context handler
        app.handlers["main"] = handler_without_context
        assert app._takes_context(handler_without_context) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
