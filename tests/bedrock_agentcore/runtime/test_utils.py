"""Tests for Bedrock AgentCore runtime utilities."""

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel

from bedrock_agentcore.runtime.utils import convert_complex_objects


class TestConvertComplexObjects:
    """Test convert_complex_objects functionality."""

    def test_primitive_types(self):
        """Test that primitive types are returned as-is."""
        # Test various primitive types
        assert convert_complex_objects("string") == "string"
        assert convert_complex_objects(42) == 42
        assert convert_complex_objects(3.14) == 3.14
        assert convert_complex_objects(True) is True
        assert convert_complex_objects(False) is False
        assert convert_complex_objects(None) is None

    def test_pydantic_models(self):
        """Test Pydantic model conversion using model_dump()."""

        class TestModel(BaseModel):
            name: str
            age: int
            active: bool

        model = TestModel(name="John", age=30, active=True)
        result = convert_complex_objects(model)

        assert isinstance(result, dict)
        assert result["name"] == "John"
        assert result["age"] == 30
        assert result["active"] is True

    def test_nested_pydantic_models(self):
        """Test nested Pydantic models are properly converted."""

        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        person = Person(name="Alice", address=Address(street="123 Main St", city="Anytown"))
        result = convert_complex_objects(person)

        assert isinstance(result, dict)
        assert result["name"] == "Alice"
        assert isinstance(result["address"], dict)
        assert result["address"]["street"] == "123 Main St"
        assert result["address"]["city"] == "Anytown"

    def test_dataclasses(self):
        """Test dataclass conversion using asdict()."""

        @dataclass
        class TestDataClass:
            name: str
            value: int
            items: List[str]

        data = TestDataClass(name="test", value=100, items=["a", "b", "c"])
        result = convert_complex_objects(data)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 100
        assert result["items"] == ["a", "b", "c"]

    def test_nested_dataclasses(self):
        """Test nested dataclasses are properly converted."""

        @dataclass
        class NestedData:
            id: int
            description: str

        @dataclass
        class ParentData:
            name: str
            nested: NestedData

        data = ParentData(name="parent", nested=NestedData(id=1, description="nested"))
        result = convert_complex_objects(data)

        assert isinstance(result, dict)
        assert result["name"] == "parent"
        assert isinstance(result["nested"], dict)
        assert result["nested"]["id"] == 1
        assert result["nested"]["description"] == "nested"

    def test_dictionaries(self):
        """Test dictionary conversion with recursive processing."""
        test_dict = {
            "string": "value",
            "number": 42,
            "nested": {"inner": "nested_value"},
            "list": [1, 2, 3],
        }
        result = convert_complex_objects(test_dict)

        assert isinstance(result, dict)
        assert result["string"] == "value"
        assert result["number"] == 42
        assert isinstance(result["nested"], dict)
        assert result["nested"]["inner"] == "nested_value"
        assert result["list"] == [1, 2, 3]

    def test_nested_dictionaries_with_complex_objects(self):
        """Test dictionaries containing Pydantic models and dataclasses."""

        class ConfigModel(BaseModel):
            setting: str
            enabled: bool

        @dataclass
        class ConfigData:
            version: str
            features: List[str]

        test_dict = {
            "config": ConfigModel(setting="test", enabled=True),
            "data": ConfigData(version="1.0", features=["a", "b"]),
            "simple": {"key": "value"},
        }
        result = convert_complex_objects(test_dict)

        assert isinstance(result, dict)
        assert isinstance(result["config"], dict)
        assert result["config"]["setting"] == "test"
        assert result["config"]["enabled"] is True
        assert isinstance(result["data"], dict)
        assert result["data"]["version"] == "1.0"
        assert result["data"]["features"] == ["a", "b"]
        assert result["simple"]["key"] == "value"

    def test_lists(self):
        """Test list conversion with recursive processing."""
        test_list = ["string", 42, True, {"nested": "value"}, [1, 2, 3]]
        result = convert_complex_objects(test_list)

        assert isinstance(result, list)
        assert result[0] == "string"
        assert result[1] == 42
        assert result[2] is True
        assert isinstance(result[3], dict)
        assert result[3]["nested"] == "value"
        assert isinstance(result[4], list)
        assert result[4] == [1, 2, 3]

    def test_tuples(self):
        """Test tuple conversion with recursive processing."""
        test_tuple = ("string", 42, {"nested": "value"})
        result = convert_complex_objects(test_tuple)

        assert isinstance(result, list)  # Tuples are converted to lists
        assert result[0] == "string"
        assert result[1] == 42
        assert isinstance(result[2], dict)
        assert result[2]["nested"] == "value"

    def test_sets(self):
        """Test set conversion with recursive processing."""
        test_set = {"a", "b", "c"}
        result = convert_complex_objects(test_set)

        assert isinstance(result, list)  # Sets are converted to lists
        # Order may vary, so check length and content
        assert len(result) == 3
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_nested_sets_with_complex_objects(self):
        """Test sets containing hashable objects (complex objects can't be in sets)."""

        # Use hashable objects instead of complex objects (which can't be in sets)
        test_set = {"item1", "item2", "item3"}
        result = convert_complex_objects(test_set)

        assert isinstance(result, list)  # Sets are converted to lists
        assert len(result) == 3
        # Check that all items are preserved
        assert "item1" in result
        assert "item2" in result
        assert "item3" in result

    def test_mixed_complex_structures(self):
        """Test complex nested structures with multiple object types."""

        class UserModel(BaseModel):
            username: str
            email: str

        @dataclass
        class UserProfile:
            bio: str
            avatar_url: Optional[str]

        class PostModel(BaseModel):
            title: str
            content: str
            author: UserModel

        # Create complex nested structure
        user = UserModel(username="john_doe", email="john@example.com")
        profile = UserProfile(bio="Software developer", avatar_url=None)
        post = PostModel(title="Hello World", content="This is a test post", author=user)

        complex_structure = {
            "users": [user, user],  # List of Pydantic models
            "profiles": [profile],  # List of dataclasses (changed from set since dataclasses aren't hashable)
            "posts": [post],  # List of nested Pydantic models
            "metadata": {"count": 2, "active": True, "tags": ["test", "example"]},
        }

        result = convert_complex_objects(complex_structure)

        # Verify structure
        assert isinstance(result, dict)
        assert "users" in result
        assert "profiles" in result
        assert "posts" in result
        assert "metadata" in result

        # Verify users list
        assert isinstance(result["users"], list)
        assert len(result["users"]) == 2
        for user_dict in result["users"]:
            assert isinstance(user_dict, dict)
            assert user_dict["username"] == "john_doe"
            assert user_dict["email"] == "john@example.com"

        # Verify profiles list
        assert isinstance(result["profiles"], list)
        assert len(result["profiles"]) == 1
        profile_dict = result["profiles"][0]
        assert isinstance(profile_dict, dict)
        assert profile_dict["bio"] == "Software developer"
        assert profile_dict["avatar_url"] is None

        # Verify posts list with nested author
        assert isinstance(result["posts"], list)
        assert len(result["posts"]) == 1
        post_dict = result["posts"][0]
        assert isinstance(post_dict, dict)
        assert post_dict["title"] == "Hello World"
        assert post_dict["content"] == "This is a test post"
        assert isinstance(post_dict["author"], dict)
        assert post_dict["author"]["username"] == "john_doe"

        # Verify metadata
        assert result["metadata"]["count"] == 2
        assert result["metadata"]["active"] is True
        assert result["metadata"]["tags"] == ["test", "example"]

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty containers
        assert convert_complex_objects({}) == {}
        assert convert_complex_objects([]) == []
        assert convert_complex_objects(()) == []
        assert convert_complex_objects(set()) == []

        # None values in containers
        test_dict = {"key": None, "list": [None, 1, None]}
        result = convert_complex_objects(test_dict)
        assert result["key"] is None
        assert result["list"] == [None, 1, None]

    def test_depth_limit_protection(self):
        """Test that excessive depth is handled gracefully."""
        # Create very deep nesting that exceeds the 50 depth limit
        deep_dict = {}
        current = deep_dict
        for _ in range(60):  # Exceed the 50 depth limit
            current["next"] = {}
            current = current["next"]
        current["value"] = "deep_value"

        result = convert_complex_objects(deep_dict)

        # Should have been truncated at some point
        assert isinstance(result, dict)
        # Navigate as deep as we can and verify depth limit was hit
        current = result
        depth_limited = False
        for _ in range(60):
            next_val = current.get("next", "")
            if "next" not in current or "<too_deep:" in str(next_val):
                depth_limited = True
                break
            current = current["next"]

        assert depth_limited, "Depth limit should have been triggered"

    def test_custom_objects_without_special_methods(self):
        """Test custom objects that don't have model_dump or are dataclasses."""

        class CustomObject:
            def __init__(self, value):
                self.value = value

        custom_obj = CustomObject("test_value")
        result = convert_complex_objects(custom_obj)

        # Custom objects without special methods should be returned as-is
        assert result == custom_obj
        assert result.value == "test_value"

    def test_pydantic_model_with_nested_dataclass(self):
        """Test Pydantic model containing a dataclass field."""

        @dataclass
        class Address:
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        person = Person(name="Bob", address=Address(street="456 Oak St", city="Somewhere"))
        result = convert_complex_objects(person)

        assert isinstance(result, dict)
        assert result["name"] == "Bob"
        assert isinstance(result["address"], dict)
        assert result["address"]["street"] == "456 Oak St"
        assert result["address"]["city"] == "Somewhere"
