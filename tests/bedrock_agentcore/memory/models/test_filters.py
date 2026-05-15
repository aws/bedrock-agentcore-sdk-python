"""Unit tests for memory record metadata filter models."""

from datetime import datetime

import pytest

from bedrock_agentcore.memory.models import (
    IndexedKey,
    MemoryMetadataFilter,
    MemoryRecordLeftExpression,
    MemoryRecordOperatorType,
    MemoryRecordRightExpression,
    MetadataValueType,
)


class TestMemoryRecordLeftExpression:
    """Test cases for MemoryRecordLeftExpression."""

    def test_build(self):
        """Test building a left expression from a key name."""
        result = MemoryRecordLeftExpression.build("priority")
        assert result == {"metadataKey": "priority"}

    def test_build_various_keys(self):
        """Test building left expressions with various key names."""
        assert MemoryRecordLeftExpression.build("agent_type") == {"metadataKey": "agent_type"}
        assert MemoryRecordLeftExpression.build("created_at") == {"metadataKey": "created_at"}
        assert MemoryRecordLeftExpression.build("tags") == {"metadataKey": "tags"}


class TestMemoryRecordRightExpression:
    """Test cases for MemoryRecordRightExpression."""

    def test_build_string(self):
        """Test building a string value right expression."""
        result = MemoryRecordRightExpression.build_string("high")
        assert result == {"metadataValue": {"stringValue": "high"}}

    def test_build_string_empty(self):
        """Test building a string value with empty string."""
        result = MemoryRecordRightExpression.build_string("")
        assert result == {"metadataValue": {"stringValue": ""}}

    def test_build_number_integer(self):
        """Test building a numeric right expression with integer-like float."""
        result = MemoryRecordRightExpression.build_number(5.0)
        assert result == {"metadataValue": {"numberValue": 5.0}}

    def test_build_number_float(self):
        """Test building a numeric right expression with decimal float."""
        result = MemoryRecordRightExpression.build_number(3.14)
        assert result == {"metadataValue": {"numberValue": 3.14}}

    def test_build_number_zero(self):
        """Test building a numeric right expression with zero."""
        result = MemoryRecordRightExpression.build_number(0.0)
        assert result == {"metadataValue": {"numberValue": 0.0}}

    def test_build_number_negative(self):
        """Test building a numeric right expression with negative value."""
        result = MemoryRecordRightExpression.build_number(-1.5)
        assert result == {"metadataValue": {"numberValue": -1.5}}

    def test_build_datetime(self):
        """Test building a datetime right expression."""
        dt = datetime(2024, 6, 15, 10, 30, 0)
        result = MemoryRecordRightExpression.build_datetime(dt)
        assert result == {"metadataValue": {"dateTimeValue": dt}}

    def test_build_string_list(self):
        """Test building a string list right expression."""
        result = MemoryRecordRightExpression.build_string_list(["tag1", "tag2", "tag3"])
        assert result == {"metadataValue": {"stringListValue": ["tag1", "tag2", "tag3"]}}

    def test_build_string_list_single_item(self):
        """Test building a string list with a single item."""
        result = MemoryRecordRightExpression.build_string_list(["only_one"])
        assert result == {"metadataValue": {"stringListValue": ["only_one"]}}

    def test_build_string_list_empty(self):
        """Test building a string list with empty list."""
        result = MemoryRecordRightExpression.build_string_list([])
        assert result == {"metadataValue": {"stringListValue": []}}


class TestMemoryRecordOperatorType:
    """Test cases for MemoryRecordOperatorType enum."""

    def test_all_operators_exist(self):
        """Test that all expected operator types are defined."""
        assert MemoryRecordOperatorType.EQUALS_TO.value == "EQUALS_TO"
        assert MemoryRecordOperatorType.EXISTS.value == "EXISTS"
        assert MemoryRecordOperatorType.NOT_EXISTS.value == "NOT_EXISTS"
        assert MemoryRecordOperatorType.BEFORE.value == "BEFORE"
        assert MemoryRecordOperatorType.AFTER.value == "AFTER"
        assert MemoryRecordOperatorType.CONTAINS.value == "CONTAINS"
        assert MemoryRecordOperatorType.GREATER_THAN.value == "GREATER_THAN"
        assert MemoryRecordOperatorType.GREATER_THAN_OR_EQUALS.value == "GREATER_THAN_OR_EQUALS"
        assert MemoryRecordOperatorType.LESS_THAN.value == "LESS_THAN"
        assert MemoryRecordOperatorType.LESS_THAN_OR_EQUALS.value == "LESS_THAN_OR_EQUALS"

    def test_operator_count(self):
        """Test that exactly 10 operators are defined."""
        assert len(MemoryRecordOperatorType) == 10


class TestMemoryMetadataFilter:
    """Test cases for MemoryMetadataFilter."""

    def test_build_expression_equals_string(self):
        """Test building an EQUALS_TO filter with a string value."""
        result = MemoryMetadataFilter.build_expression(
            MemoryRecordLeftExpression.build("agent_type"),
            MemoryRecordOperatorType.EQUALS_TO,
            MemoryRecordRightExpression.build_string("support"),
        )

        assert result == {
            "left": {"metadataKey": "agent_type"},
            "operator": "EQUALS_TO",
            "right": {"metadataValue": {"stringValue": "support"}},
        }

    def test_build_expression_greater_than_number(self):
        """Test building a GREATER_THAN filter with a numeric value."""
        result = MemoryMetadataFilter.build_expression(
            MemoryRecordLeftExpression.build("priority"),
            MemoryRecordOperatorType.GREATER_THAN,
            MemoryRecordRightExpression.build_number(3.0),
        )

        assert result == {
            "left": {"metadataKey": "priority"},
            "operator": "GREATER_THAN",
            "right": {"metadataValue": {"numberValue": 3.0}},
        }

    def test_build_expression_less_than_or_equals(self):
        """Test building a LESS_THAN_OR_EQUALS filter."""
        result = MemoryMetadataFilter.build_expression(
            MemoryRecordLeftExpression.build("score"),
            MemoryRecordOperatorType.LESS_THAN_OR_EQUALS,
            MemoryRecordRightExpression.build_number(100.0),
        )

        assert result == {
            "left": {"metadataKey": "score"},
            "operator": "LESS_THAN_OR_EQUALS",
            "right": {"metadataValue": {"numberValue": 100.0}},
        }

    def test_build_expression_before_datetime(self):
        """Test building a BEFORE filter with a datetime value."""
        dt = datetime(2024, 1, 1, 0, 0, 0)
        result = MemoryMetadataFilter.build_expression(
            MemoryRecordLeftExpression.build("created_at"),
            MemoryRecordOperatorType.BEFORE,
            MemoryRecordRightExpression.build_datetime(dt),
        )

        assert result == {
            "left": {"metadataKey": "created_at"},
            "operator": "BEFORE",
            "right": {"metadataValue": {"dateTimeValue": dt}},
        }

    def test_build_expression_after_datetime(self):
        """Test building an AFTER filter with a datetime value."""
        dt = datetime(2024, 12, 31, 23, 59, 59)
        result = MemoryMetadataFilter.build_expression(
            MemoryRecordLeftExpression.build("updated_at"),
            MemoryRecordOperatorType.AFTER,
            MemoryRecordRightExpression.build_datetime(dt),
        )

        assert result == {
            "left": {"metadataKey": "updated_at"},
            "operator": "AFTER",
            "right": {"metadataValue": {"dateTimeValue": dt}},
        }

    def test_build_expression_contains_string_list(self):
        """Test building a CONTAINS filter with a string list."""
        result = MemoryMetadataFilter.build_expression(
            MemoryRecordLeftExpression.build("tags"),
            MemoryRecordOperatorType.CONTAINS,
            MemoryRecordRightExpression.build_string_list(["urgent", "follow-up"]),
        )

        assert result == {
            "left": {"metadataKey": "tags"},
            "operator": "CONTAINS",
            "right": {"metadataValue": {"stringListValue": ["urgent", "follow-up"]}},
        }

    def test_build_expression_exists_no_right_operand(self):
        """Test building an EXISTS filter without a right operand."""
        result = MemoryMetadataFilter.build_expression(
            MemoryRecordLeftExpression.build("metadata_key"),
            MemoryRecordOperatorType.EXISTS,
        )

        assert result == {
            "left": {"metadataKey": "metadata_key"},
            "operator": "EXISTS",
        }
        assert "right" not in result

    def test_build_expression_not_exists_no_right_operand(self):
        """Test building a NOT_EXISTS filter without a right operand."""
        result = MemoryMetadataFilter.build_expression(
            MemoryRecordLeftExpression.build("optional_field"),
            MemoryRecordOperatorType.NOT_EXISTS,
        )

        assert result == {
            "left": {"metadataKey": "optional_field"},
            "operator": "NOT_EXISTS",
        }
        assert "right" not in result

    def test_build_expression_exists_with_none_right_operand(self):
        """Test building EXISTS filter with explicit None right operand."""
        result = MemoryMetadataFilter.build_expression(
            MemoryRecordLeftExpression.build("some_key"),
            MemoryRecordOperatorType.EXISTS,
            None,
        )

        assert result == {
            "left": {"metadataKey": "some_key"},
            "operator": "EXISTS",
        }
        assert "right" not in result

    def test_build_expression_rejects_right_operand_with_exists(self):
        """EXISTS rejects a right operand at build time."""
        with pytest.raises(ValueError, match="EXISTS does not accept a right operand"):
            MemoryMetadataFilter.build_expression(
                MemoryRecordLeftExpression.build("priority"),
                MemoryRecordOperatorType.EXISTS,
                MemoryRecordRightExpression.build_string("high"),
            )

    def test_build_expression_rejects_right_operand_with_not_exists(self):
        """NOT_EXISTS rejects a right operand at build time."""
        with pytest.raises(ValueError, match="NOT_EXISTS does not accept a right operand"):
            MemoryMetadataFilter.build_expression(
                MemoryRecordLeftExpression.build("priority"),
                MemoryRecordOperatorType.NOT_EXISTS,
                MemoryRecordRightExpression.build_string("high"),
            )

    def test_build_expression_requires_right_operand_for_comparison_operators(self):
        """Non-existence operators raise when right operand is missing."""
        with pytest.raises(ValueError, match="EQUALS_TO requires a right operand"):
            MemoryMetadataFilter.build_expression(
                MemoryRecordLeftExpression.build("priority"),
                MemoryRecordOperatorType.EQUALS_TO,
            )


class TestMetadataValueType:
    """Test cases for MetadataValueType enum."""

    def test_all_types_exist(self):
        """Test that all expected value types are defined."""
        assert MetadataValueType.STRING.value == "STRING"
        assert MetadataValueType.STRINGLIST.value == "STRINGLIST"
        assert MetadataValueType.NUMBER.value == "NUMBER"

    def test_type_count(self):
        """Test that exactly 3 value types are defined."""
        assert len(MetadataValueType) == 3


class TestIndexedKey:
    """Test cases for IndexedKey."""

    def test_build_string_key(self):
        """Test building an indexed key with STRING type."""
        result = IndexedKey.build("agent_type", MetadataValueType.STRING)
        assert result == {"key": "agent_type", "type": "STRING"}

    def test_build_number_key(self):
        """Test building an indexed key with NUMBER type."""
        result = IndexedKey.build("priority", MetadataValueType.NUMBER)
        assert result == {"key": "priority", "type": "NUMBER"}

    def test_build_stringlist_key(self):
        """Test building an indexed key with STRINGLIST type."""
        result = IndexedKey.build("tags", MetadataValueType.STRINGLIST)
        assert result == {"key": "tags", "type": "STRINGLIST"}

    def test_build_multiple_keys(self):
        """Test building a list of indexed keys for create_memory."""
        indexed_keys = [
            IndexedKey.build("priority", MetadataValueType.NUMBER),
            IndexedKey.build("agent_type", MetadataValueType.STRING),
            IndexedKey.build("categories", MetadataValueType.STRINGLIST),
        ]

        assert indexed_keys == [
            {"key": "priority", "type": "NUMBER"},
            {"key": "agent_type", "type": "STRING"},
            {"key": "categories", "type": "STRINGLIST"},
        ]
