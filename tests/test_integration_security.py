"""Integration tests for security enhancements."""

import pytest
import time
from unittest.mock import patch, MagicMock

from bedrock_agentcore._utils.endpoints import get_data_plane_endpoint, get_control_plane_endpoint
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from bedrock_agentcore.services.identity import IdentityClient


class TestEndpointSecurity:
    """Test endpoint validation integration."""
    
    def test_valid_region_endpoints(self):
        """Test endpoint generation with valid regions."""
        valid_regions = ["us-east-1", "us-west-2", "eu-west-1"]
        for region in valid_regions:
            dp_endpoint = get_data_plane_endpoint(region)
            cp_endpoint = get_control_plane_endpoint(region)
            
            assert dp_endpoint.startswith("https://")
            assert region in dp_endpoint
            assert "amazonaws.com" in dp_endpoint
            assert cp_endpoint.startswith("https://")
            assert region in cp_endpoint
            assert "amazonaws.com" in cp_endpoint
    
    def test_invalid_region_rejection(self):
        """Test invalid region format rejection."""
        invalid_regions = ["invalid", "us-east", "123-west-1", ""]
        for region in invalid_regions:
            with pytest.raises(ValueError, match="Invalid AWS region format"):
                get_data_plane_endpoint(region)
            with pytest.raises(ValueError, match="Invalid AWS region format"):
                get_control_plane_endpoint(region)


class TestContextSecurity:
    """Test context security enhancements."""
    
    def test_token_expiration(self):
        """Test token expiration handling."""
        # Set token with short expiry
        BedrockAgentCoreContext.set_workload_access_token("test-token", expiry_seconds=1)
        
        # Should be available immediately
        token = BedrockAgentCoreContext.get_workload_access_token()
        assert token == "test-token"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be None after expiration
        expired_token = BedrockAgentCoreContext.get_workload_access_token()
        assert expired_token is None
    
    def test_empty_token_rejection(self):
        """Test empty token rejection."""
        with pytest.raises(ValueError, match="Token cannot be empty"):
            BedrockAgentCoreContext.set_workload_access_token("")
        
        with pytest.raises(ValueError, match="Token cannot be empty"):
            BedrockAgentCoreContext.set_workload_access_token("   ")
    
    def test_token_cleanup(self):
        """Test token cleanup functionality."""
        BedrockAgentCoreContext.set_workload_access_token("test-token")
        assert BedrockAgentCoreContext.get_workload_access_token() == "test-token"
        
        BedrockAgentCoreContext.clear_workload_access_token()
        assert BedrockAgentCoreContext.get_workload_access_token() is None


class TestIdentityClientSecurity:
    """Test identity client security enhancements."""
    
    def setup_method(self):
        """Clean up environment before each test."""
        import os
        # Remove any endpoint overrides that might interfere
        os.environ.pop('BEDROCK_AGENTCORE_DP_ENDPOINT', None)
        os.environ.pop('BEDROCK_AGENTCORE_CP_ENDPOINT', None)
        
        # Reload endpoints module to pick up clean environment
        import importlib
        from bedrock_agentcore._utils import endpoints
        importlib.reload(endpoints)
    
    def test_empty_region_rejection(self):
        """Test empty region rejection in identity client."""
        with pytest.raises(ValueError, match="Region cannot be empty"):
            IdentityClient("")
    
    @patch('boto3.client')
    def test_workload_name_validation(self, mock_boto_client):
        """Test workload name validation."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        client = IdentityClient("us-east-1")
        
        # Valid workload names should work
        valid_names = ["workload-123", "my_workload", "test-workload_1"]
        for name in valid_names:
            mock_client.get_workload_access_token.return_value = {"accessToken": "token"}
            result = client.get_workload_access_token(name)
            assert "accessToken" in result
        
        # Invalid workload names should be rejected
        invalid_names = ["workload with spaces", "workload@special", "a" * 65]
        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid workload name format"):
                client.get_workload_access_token(name)
    
    @patch('boto3.client')
    def test_token_tracking(self, mock_boto_client):
        """Test token tracking in identity client."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.get_workload_access_token.return_value = {"accessToken": "token123"}
        
        client = IdentityClient("us-east-1")
        
        # Initially no tokens tracked
        assert client.token_manager.active_count == 0
        
        # Get token should track it
        client.get_workload_access_token("test-workload")
        assert client.token_manager.active_count == 1
        
        # Cleanup should clear all tokens
        client.cleanup_tokens()
        assert client.token_manager.active_count == 0


class TestThreadSafety:
    """Test thread safety of security components."""
    
    def test_token_manager_thread_safety(self):
        """Test TokenManager thread safety."""
        from bedrock_agentcore._utils.security import TokenManager
        import threading
        import time
        
        manager = TokenManager()
        results = []
        
        def register_tokens():
            for i in range(100):
                manager.register_token(f"token_{threading.current_thread().ident}_{i}")
                time.sleep(0.001)  # Small delay to increase contention
        
        # Create multiple threads
        threads = [threading.Thread(target=register_tokens) for _ in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have 500 tokens (5 threads * 100 tokens each)
        assert manager.active_count == 500
        
        # Cleanup should work safely
        manager.cleanup_all()
        assert manager.active_count == 0