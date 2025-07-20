"""Security validation tests."""

import pytest
from bedrock_agentcore._utils.security import SecurityValidator, TokenManager


class TestSecurityValidator:
    """Test security validation utilities."""
    
    def test_validate_endpoint_valid(self):
        """Test valid AWS endpoint validation."""
        valid_endpoints = [
            "https://bedrock-agentcore.us-west-2.amazonaws.com",
            "https://bedrock-agentcore-control.us-east-1.amazonaws.com",
        ]
        for endpoint in valid_endpoints:
            assert SecurityValidator.validate_endpoint(endpoint)
    
    def test_validate_endpoint_invalid(self):
        """Test invalid endpoint rejection."""
        invalid_endpoints = [
            "http://bedrock-agentcore.us-west-2.amazonaws.com",  # HTTP
            "https://malicious-site.com",  # Non-AWS
            "https://bedrock-agentcore.us-west-2.evil.com",  # Wrong domain
            "",  # Empty
            None,  # None
        ]
        for endpoint in invalid_endpoints:
            assert not SecurityValidator.validate_endpoint(endpoint)
    
    def test_validate_workload_name_valid(self):
        """Test valid workload name validation."""
        valid_names = [
            "workload-123",
            "my_workload",
            "test-workload_1",
            "a" * 64,  # Max length
        ]
        for name in valid_names:
            assert SecurityValidator.validate_workload_name(name)
    
    def test_validate_workload_name_invalid(self):
        """Test invalid workload name rejection."""
        invalid_names = [
            "workload with spaces",  # Spaces
            "workload@special",  # Special chars
            "a" * 65,  # Too long
            "",  # Empty
            None,  # None
        ]
        for name in invalid_names:
            assert not SecurityValidator.validate_workload_name(name)
    
    def test_sanitize_log_data(self):
        """Test log data sanitization."""
        test_cases = [
            ('token="secret123"', 'token="[REDACTED]"'),
            ('Bearer abc123def', 'Bearer [REDACTED]'),
            ('password: "mypass"', 'password="[REDACTED]"'),
            ('key=apikey123', 'key="[REDACTED]"'),
            ('normal log message', 'normal log message'),
        ]
        
        for input_data, expected in test_cases:
            result = SecurityValidator.sanitize_log_data(input_data)
            assert expected in result


class TestTokenManager:
    """Test token lifecycle management."""
    
    def test_token_registration(self):
        """Test token registration and tracking."""
        manager = TokenManager()
        assert manager.active_count == 0
        
        manager.register_token("token1")
        assert manager.active_count == 1
        
        manager.register_token("token2")
        assert manager.active_count == 2
    
    def test_token_cleanup(self):
        """Test individual token cleanup."""
        manager = TokenManager()
        manager.register_token("token1")
        manager.register_token("token2")
        
        manager.cleanup_token("token1")
        assert manager.active_count == 1
        
        manager.cleanup_token("nonexistent")  # Should not error
        assert manager.active_count == 1
    
    def test_cleanup_all(self):
        """Test cleanup of all tokens."""
        manager = TokenManager()
        manager.register_token("token1")
        manager.register_token("token2")
        
        manager.cleanup_all()
        assert manager.active_count == 0