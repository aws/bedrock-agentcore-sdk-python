"""Event metadata filter models for querying events based on metadata."""

from enum import Enum
from typing import Optional, TypedDict, Union


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
