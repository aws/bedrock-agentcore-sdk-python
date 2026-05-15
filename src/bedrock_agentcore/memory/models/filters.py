"""Metadata filter models for querying events and memory records."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, TypedDict, Union

from typing_extensions import NotRequired

# ============================================================================
# Event Metadata Filters (existing)
# ============================================================================


class StringValue(TypedDict):
    """Value associated with the `eventMetadata` key."""

    stringValue: str

    @staticmethod
    def build(value: str) -> "StringValue":
        """Build a StringValue from a string."""
        return {"stringValue": value}


MetadataValue = Union[StringValue]
"""
Union type representing metadata values.

Variants:
- StringValue: {"stringValue": str} - String metadata value
"""

MetadataKey = Union[str]
"""
Union type representing metadata key.
"""


class LeftExpression(TypedDict):
    """Left operand of the event metadata filter expression."""

    metadataKey: MetadataKey

    @staticmethod
    def build(key: str) -> "LeftExpression":
        """Builds the `metadataKey` for `LeftExpression`."""
        return {"metadataKey": key}


class OperatorType(Enum):
    """Operator applied to the event metadata filter expression.

    Currently supports:
    - `EQUALS_TO`
    - `EXISTS`
    - `NOT_EXISTS`
    """

    EQUALS_TO = "EQUALS_TO"
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"


class RightExpression(TypedDict):
    """Right operand of the event metadata filter expression.

    Variants:
    - StringValue: {"metadataValue": {"stringValue": str}}
    """

    metadataValue: MetadataValue

    @staticmethod
    def build(value: str) -> "RightExpression":
        """Builds the `RightExpression` for `stringValue` type."""
        return {"metadataValue": StringValue.build(value)}


class EventMetadataFilter(TypedDict):
    """Filter expression for retrieving events based on metadata associated with an event.

    Args:
        left: `LeftExpression` of the event metadata filter expression.
        operator: `OperatorType` applied to the event metadata filter expression.
        right: Optional `RightExpression` of the event metadata filter expression.
    """

    left: LeftExpression
    operator: OperatorType
    right: Optional[RightExpression]

    def build_expression(
        left_operand: LeftExpression,
        operator: OperatorType,
        right_operand: Optional[RightExpression] = None,
    ) -> "EventMetadataFilter":
        """Build the required event metadata filter expression.

        This method builds the required event metadata filter expression into the
        `EventMetadataFilterExpression` type when querying listEvents.

        Args:
            left_operand: Left operand of the event metadata filter expression
            operator: Operator applied to the event metadata filter expression
            right_operand: Optional right_operand of the event metadata filter expression.

        Example:
        ```
            left_operand = LeftExpression.build_key(key='location')
            operator = OperatorType.EQUALS_TO
            right_operand = RightExpression.build_string_value(value='NYC')
        ```

        #### Response Object:
        ```
            {
                'left': {
                    'metadataKey': 'location'
                },
                'operator': 'EQUALS_TO',
                'right': {
                    'metadataValue': {
                        'stringValue': 'NYC'
                    }
                }
            }
        ```
        """
        filter = {"left": left_operand, "operator": operator.value}

        if right_operand:
            filter["right"] = right_operand
        return filter


# ============================================================================
# Memory Record Metadata Filters (LTM)
# ============================================================================


class MemoryRecordOperatorType(Enum):
    """Operator applied to memory record metadata filter expressions.

    Each operator is paired with a specific right-operand value type. Mismatches
    are rejected by the service — pass the right operand built via the matching
    `MemoryRecordRightExpression.build_*` factory.

    | Operator                 | Right operand               | Builder                  |
    |--------------------------|-----------------------------|--------------------------|
    | `EQUALS_TO`              | string                      | `build_string`           |
    | `EXISTS`                 | (none)                      | —                        |
    | `NOT_EXISTS`             | (none)                      | —                        |
    | `BEFORE`                 | datetime                    | `build_datetime`         |
    | `AFTER`                  | datetime                    | `build_datetime`         |
    | `CONTAINS`               | string list                 | `build_string_list`      |
    | `GREATER_THAN`           | number                      | `build_number`           |
    | `GREATER_THAN_OR_EQUALS` | number                      | `build_number`           |
    | `LESS_THAN`              | number                      | `build_number`           |
    | `LESS_THAN_OR_EQUALS`    | number                      | `build_number`           |
    """

    EQUALS_TO = "EQUALS_TO"
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    CONTAINS = "CONTAINS"
    GREATER_THAN = "GREATER_THAN"
    GREATER_THAN_OR_EQUALS = "GREATER_THAN_OR_EQUALS"
    LESS_THAN = "LESS_THAN"
    LESS_THAN_OR_EQUALS = "LESS_THAN_OR_EQUALS"


class MemoryRecordLeftExpression(TypedDict):
    """Left operand of the memory record metadata filter expression."""

    metadataKey: str

    @staticmethod
    def build(key: str) -> "MemoryRecordLeftExpression":
        """Build a MemoryRecordLeftExpression from a key name."""
        return {"metadataKey": key}


class MemoryRecordRightExpression(TypedDict):
    """Right operand of the memory record metadata filter expression.

    Variants:
    - {"metadataValue": {"stringValue": str}}
    - {"metadataValue": {"numberValue": float}}
    - {"metadataValue": {"dateTimeValue": datetime}}
    - {"metadataValue": {"stringListValue": List[str]}}
    """

    metadataValue: dict

    @staticmethod
    def build_string(value: str) -> "MemoryRecordRightExpression":
        """Build a right expression with a string value."""
        return {"metadataValue": {"stringValue": value}}

    @staticmethod
    def build_number(value: Union[int, float]) -> "MemoryRecordRightExpression":
        """Build a right expression with a numeric value."""
        return {"metadataValue": {"numberValue": value}}

    @staticmethod
    def build_datetime(value: datetime) -> "MemoryRecordRightExpression":
        """Build a right expression with a datetime value."""
        return {"metadataValue": {"dateTimeValue": value}}

    @staticmethod
    def build_string_list(value: List[str]) -> "MemoryRecordRightExpression":
        """Build a right expression with a string list value."""
        return {"metadataValue": {"stringListValue": value}}


class MemoryMetadataFilter(TypedDict):
    """Filter expression for querying memory records by metadata.

    Used with `retrieve_memories()` and `list_memory_records()` to scope
    results by indexed metadata keys before semantic search runs.

    Args:
        left: `MemoryRecordLeftExpression` specifying the metadata key.
        operator: `MemoryRecordOperatorType` defining the comparison.
        right: Optional `MemoryRecordRightExpression` with the value to compare against.
               Not required for EXISTS/NOT_EXISTS operators.

    Example:
        ```python
        filter = MemoryMetadataFilter.build_expression(
            MemoryRecordLeftExpression.build("priority"),
            MemoryRecordOperatorType.EQUALS_TO,
            MemoryRecordRightExpression.build_string("high"),
        )
        ```
    """

    left: MemoryRecordLeftExpression
    # Stored as the operator's string value (e.g. "EQUALS_TO"), not the enum itself,
    # since this dict is serialized directly to the AgentCore service.
    operator: str
    right: NotRequired[MemoryRecordRightExpression]

    @staticmethod
    def build_expression(
        left_operand: "MemoryRecordLeftExpression",
        operator: MemoryRecordOperatorType,
        right_operand: Optional["MemoryRecordRightExpression"] = None,
    ) -> "MemoryMetadataFilter":
        """Build a memory metadata filter expression.

        Args:
            left_operand: The metadata key to filter on.
            operator: The comparison operator.
            right_operand: The value to compare against. Required for all operators
                          except EXISTS and NOT_EXISTS, which must NOT receive a
                          right operand.

        Raises:
            ValueError: If `right_operand` is supplied with EXISTS or NOT_EXISTS,
                or if `right_operand` is missing for any other operator.

        Example:
        ```python
            left_operand = MemoryRecordLeftExpression.build("priority")
            operator = MemoryRecordOperatorType.GREATER_THAN
            right_operand = MemoryRecordRightExpression.build_number(3.0)

            filter = MemoryMetadataFilter.build_expression(left_operand, operator, right_operand)
            # Result:
            # {
            #     "left": {"metadataKey": "priority"},
            #     "operator": "GREATER_THAN",
            #     "right": {"metadataValue": {"numberValue": 3.0}}
            # }
        ```
        """
        is_existence_op = operator in (
            MemoryRecordOperatorType.EXISTS,
            MemoryRecordOperatorType.NOT_EXISTS,
        )
        if is_existence_op and right_operand is not None:
            raise ValueError(f"{operator.value} does not accept a right operand; the service rejects this combination.")
        if not is_existence_op and right_operand is None:
            raise ValueError(f"{operator.value} requires a right operand.")

        filter = {"left": left_operand, "operator": operator.value}

        if right_operand is not None:
            filter["right"] = right_operand
        return filter


# ============================================================================
# Indexed Key Types (Control Plane)
# ============================================================================


class MetadataValueType(Enum):
    """Supported data types for indexed metadata key values."""

    STRING = "STRING"
    STRINGLIST = "STRINGLIST"
    NUMBER = "NUMBER"


class IndexedKey(TypedDict):
    r"""A metadata key indexed for filtering on memory records.

    Args:
        key: The metadata key name. 1-128 characters. May contain alphanumeric
            characters, whitespace, and the symbols `. _ : / = + @ -`. Pattern:
            `[a-zA-Z0-9\s._:/=+@-]*`.
        type: The data type of the indexed key value.

    Note:
        Indexed keys are append-only on the AgentCore service: once an
        indexed key is declared on a memory it cannot be removed. New keys
        can be added via `update_memory(addIndexedKeys=...)`.

    Example:
        ```python
        indexed_keys = [
            IndexedKey.build("priority", MetadataValueType.NUMBER),
            IndexedKey.build("agent_type", MetadataValueType.STRING),
        ]
        ```
    """

    key: str
    type: str

    @staticmethod
    def build(key: str, value_type: MetadataValueType) -> "IndexedKey":
        """Build an IndexedKey configuration.

        Args:
            key: The metadata key name.
            value_type: The MetadataValueType for this key.
        """
        return {"key": key, "type": value_type.value}
