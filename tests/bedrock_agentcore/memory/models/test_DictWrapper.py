"""Unit tests for DictWrapper class."""

import pytest

from bedrock_agentcore.memory.models.DictWrapper import DictWrapper


class TestDictWrapper:
    """Test cases for DictWrapper class."""

    def test_dict_wrapper_initialization(self):
        """Test DictWrapper initialization."""
        data = {"key1": "value1", "key2": "value2", "nested": {"inner": "value"}}
        wrapper = DictWrapper(data)

        assert wrapper._data == data

    def test_getattr_existing_key(self):
        """Test __getattr__ with existing key."""
        data = {"name": "test", "value": 123}
        wrapper = DictWrapper(data)

        assert wrapper.name == "test"
        assert wrapper.value == 123

    def test_getattr_missing_key(self):
        """Test __getattr__ with missing key returns None."""
        data = {"existing": "value"}
        wrapper = DictWrapper(data)

        assert wrapper.missing is None

    def test_getitem_existing_key(self):
        """Test __getitem__ with existing key."""
        data = {"key1": "value1", "key2": 42}
        wrapper = DictWrapper(data)

        assert wrapper["key1"] == "value1"
        assert wrapper["key2"] == 42

    def test_getitem_missing_key(self):
        """Test __getitem__ with missing key raises KeyError."""
        data = {"existing": "value"}
        wrapper = DictWrapper(data)

        with pytest.raises(KeyError):
            _ = wrapper["missing"]

    def test_get_existing_key(self):
        """Test get() with existing key."""
        data = {"key1": "value1", "key2": None}
        wrapper = DictWrapper(data)

        assert wrapper.get("key1") == "value1"
        assert wrapper.get("key2") is None

    def test_get_missing_key_default_none(self):
        """Test get() with missing key returns None by default."""
        data = {"existing": "value"}
        wrapper = DictWrapper(data)

        assert wrapper.get("missing") is None

    def test_get_missing_key_custom_default(self):
        """Test get() with missing key returns custom default."""
        data = {"existing": "value"}
        wrapper = DictWrapper(data)

        assert wrapper.get("missing", "default") == "default"
        assert wrapper.get("missing", 42) == 42

    def test_contains_existing_key(self):
        """Test __contains__ with existing key."""
        data = {"key1": "value1", "key2": None}
        wrapper = DictWrapper(data)

        assert "key1" in wrapper
        assert "key2" in wrapper

    def test_contains_missing_key(self):
        """Test __contains__ with missing key."""
        data = {"existing": "value"}
        wrapper = DictWrapper(data)

        assert "missing" not in wrapper

    def test_keys(self):
        """Test keys() method."""
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        wrapper = DictWrapper(data)

        keys = wrapper.keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_values(self):
        """Test values() method."""
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        wrapper = DictWrapper(data)

        values = wrapper.values()
        assert set(values) == {"value1", "value2", "value3"}

    def test_items(self):
        """Test items() method."""
        data = {"key1": "value1", "key2": "value2"}
        wrapper = DictWrapper(data)

        items = wrapper.items()
        assert set(items) == {("key1", "value1"), ("key2", "value2")}

    def test_dir(self):
        """Test __dir__ method for tab completion."""
        data = {"key1": "value1", "key2": "value2", "method_name": "value"}
        wrapper = DictWrapper(data)

        dir_result = wrapper.__dir__()
        assert "key1" in dir_result
        assert "key2" in dir_result
        assert "method_name" in dir_result
        assert "get" in dir_result

    def test_repr(self):
        """Test __repr__ method."""
        data = {"key1": "value1", "key2": 42}
        wrapper = DictWrapper(data)

        repr_result = wrapper.__repr__()
        assert repr_result == str(data)

    def test_str(self):
        """Test __str__ method."""
        data = {"key1": "value1", "key2": 42}
        wrapper = DictWrapper(data)

        str_result = wrapper.__str__()
        assert str_result == str(data)
        assert str_result == wrapper.__repr__()

    def test_complex_nested_data(self):
        """Test DictWrapper with complex nested data."""
        data = {
            "simple": "value",
            "nested": {"inner": {"deep": "value"}},
            "list": [1, 2, 3],
            "mixed": {"list": [{"key": "value"}], "number": 42},
        }
        wrapper = DictWrapper(data)

        assert wrapper.simple == "value"
        assert wrapper.nested == {"inner": {"deep": "value"}}
        assert wrapper.list == [1, 2, 3]
        assert wrapper.mixed["number"] == 42

    def test_empty_data(self):
        """Test DictWrapper with empty data."""
        wrapper = DictWrapper({})

        assert wrapper.any_key is None
        assert wrapper.get("any_key") is None
        assert "any_key" not in wrapper
        assert list(wrapper.keys()) == []
        assert list(wrapper.values()) == []
        assert list(wrapper.items()) == []

    def test_data_with_special_characters(self):
        """Test DictWrapper with keys containing special characters."""
        data = {
            "normal_key": "value1",
            "key-with-dashes": "value2",
            "key_with_underscores": "value3",
            "key.with.dots": "value4",
            "123numeric": "value5",
        }
        wrapper = DictWrapper(data)

        # Access via getitem (always works)
        assert wrapper["key-with-dashes"] == "value2"
        assert wrapper["key.with.dots"] == "value4"
        assert wrapper["123numeric"] == "value5"

        # Access via getattr (works for valid Python identifiers)
        assert wrapper.normal_key == "value1"
        assert wrapper.key_with_underscores == "value3"

    def test_data_modification_independence(self):
        """Test that modifying original data doesn't affect wrapper behavior."""
        data = {"key1": "original"}
        wrapper = DictWrapper(data)

        # Verify initial state
        assert wrapper.key1 == "original"

        # Modify original data
        data["key1"] = "modified"
        data["key2"] = "new"

        # Wrapper should reflect the changes since it holds a reference
        assert wrapper.key1 == "modified"
        assert wrapper.key2 == "new"

    def test_none_values(self):
        """Test DictWrapper with None values."""
        data = {"none_value": None, "empty_string": "", "zero": 0, "false": False}
        wrapper = DictWrapper(data)

        assert wrapper.none_value is None
        assert wrapper.empty_string == ""
        assert wrapper.zero == 0
        assert wrapper.false is False

        # All keys should exist
        assert "none_value" in wrapper
        assert "empty_string" in wrapper
        assert "zero" in wrapper
        assert "false" in wrapper
