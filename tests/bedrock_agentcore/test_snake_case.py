"""Tests for snake_case kwargs utilities."""

from unittest.mock import MagicMock

import pytest

from bedrock_agentcore._utils.snake_case import accept_snake_case_kwargs, snake_to_camel


class TestSnakeToCamel:
    """Tests for snake_to_camel conversion."""

    def test_single_word(self):
        assert snake_to_camel("name") == "name"

    def test_two_words(self):
        assert snake_to_camel("memory_id") == "memoryId"

    def test_already_camel_case_passthrough(self):
        assert snake_to_camel("memoryId") == "memoryId"

    def test_multi_segment_snake(self):
        assert snake_to_camel("memory_execution_role_arn") == "memoryExecutionRoleArn"

    def test_empty_string(self):
        assert snake_to_camel("") == ""

    # Reject malformed snake_case early rather than silently converting it.
    # We don't want users depending on conversion quirks (e.g. "a__b" → "aB")
    # that only work by accident of the current implementation.

    def test_rejects_leading_underscore(self):
        with pytest.raises(ValueError, match="Invalid parameter name"):
            snake_to_camel("_private")

    def test_rejects_consecutive_underscores(self):
        with pytest.raises(ValueError, match="Invalid parameter name"):
            snake_to_camel("a__b")

    def test_rejects_trailing_underscore(self):
        with pytest.raises(ValueError, match="Invalid parameter name"):
            snake_to_camel("name_")

    def test_rejects_uppercase_in_snake(self):
        with pytest.raises(ValueError, match="Invalid parameter name"):
            snake_to_camel("memory_ID")


class TestAcceptSnakeCaseKwargs:
    """Tests for accept_snake_case_kwargs wrapper."""

    def setup_method(self):
        self.mock_method = MagicMock(return_value={"result": "ok"})

    def test_snake_case_converted(self):
        wrapped = accept_snake_case_kwargs(self.mock_method)
        wrapped(memory_id="mem-1", actor_id="user-1")
        self.mock_method.assert_called_once_with(memoryId="mem-1", actorId="user-1")

    def test_camel_case_passthrough(self):
        wrapped = accept_snake_case_kwargs(self.mock_method)
        wrapped(memoryId="mem-1", actorId="user-1")
        self.mock_method.assert_called_once_with(memoryId="mem-1", actorId="user-1")

    def test_mixed_snake_and_camel_different_params(self):
        wrapped = accept_snake_case_kwargs(self.mock_method)
        wrapped(memory_id="mem-1", actorId="user-1")
        self.mock_method.assert_called_once_with(memoryId="mem-1", actorId="user-1")

    def test_collision_raises_type_error(self):
        wrapped = accept_snake_case_kwargs(self.mock_method)
        with pytest.raises(TypeError, match="memoryId.*memory_id"):
            wrapped(memoryId="mem-1", memory_id="mem-2")

    def test_return_value_forwarded(self):
        wrapped = accept_snake_case_kwargs(self.mock_method)
        result = wrapped(memory_id="mem-1")
        assert result == {"result": "ok"}

    def test_positional_args_forwarded(self):
        wrapped = accept_snake_case_kwargs(self.mock_method)
        wrapped("pos1", "pos2", memory_id="mem-1")
        self.mock_method.assert_called_once_with("pos1", "pos2", memoryId="mem-1")

    def test_no_kwargs(self):
        wrapped = accept_snake_case_kwargs(self.mock_method)
        wrapped()
        self.mock_method.assert_called_once_with()

    def test_exception_propagated(self):
        self.mock_method.side_effect = ValueError("boom")
        wrapped = accept_snake_case_kwargs(self.mock_method)
        with pytest.raises(ValueError, match="boom"):
            wrapped(memory_id="mem-1")

    def test_preserves_function_name(self):
        def my_boto3_method():
            pass

        wrapped = accept_snake_case_kwargs(my_boto3_method)
        assert wrapped.__name__ == "my_boto3_method"
