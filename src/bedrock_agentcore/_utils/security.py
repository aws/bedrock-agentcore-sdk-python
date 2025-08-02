"""Security utilities for Bedrock AgentCore SDK."""

import re
import urllib.parse
from typing import Optional


class SecurityValidator:
    """Security validation utilities."""
    
    # AWS service endpoint pattern - must end with .amazonaws.com
    AWS_ENDPOINT_PATTERN = re.compile(
        r'^https://[\w-]+\.[\w-]+\.amazonaws\.com$'
    )
    
    # Safe characters for workload names
    WORKLOAD_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
    
    @classmethod
    def validate_endpoint(cls, endpoint: str) -> bool:
        """Validate AWS service endpoint format."""
        if not endpoint or not isinstance(endpoint, str):
            return False
        return bool(cls.AWS_ENDPOINT_PATTERN.match(endpoint))
    
    @classmethod
    def validate_workload_name(cls, name: str) -> bool:
        """Validate workload name format."""
        if not name or not isinstance(name, str):
            return False
        return bool(cls.WORKLOAD_NAME_PATTERN.match(name))
    
    @classmethod
    def sanitize_log_data(cls, data: str) -> str:
        """Remove sensitive data from log messages."""
        if not isinstance(data, str):
            return str(data)
        
        # Remove potential tokens/keys
        patterns = [
            (r'token["\s]*[:=]["\s]*[^"\s,}]+', 'token="[REDACTED]"'),
            (r'key["\s]*[:=]["\s]*[^"\s,}]+', 'key="[REDACTED]"'),
            (r'password["\s]*[:=]["\s]*[^"\s,}]+', 'password="[REDACTED]"'),
            (r'Bearer\s+[^\s]+', 'Bearer [REDACTED]'),
        ]
        
        result = data
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result


class TokenManager:
    """Secure token lifecycle management."""
    
    def __init__(self):
        import threading
        self._active_tokens = set()
        self._lock = threading.Lock()
    
    def register_token(self, token_id: str) -> None:
        """Register active token for cleanup tracking."""
        with self._lock:
            self._active_tokens.add(token_id)
    
    def cleanup_token(self, token_id: str) -> None:
        """Mark token for cleanup."""
        with self._lock:
            self._active_tokens.discard(token_id)
    
    def cleanup_all(self) -> None:
        """Cleanup all tracked tokens."""
        with self._lock:
            self._active_tokens.clear()
    
    @property
    def active_count(self) -> int:
        """Get count of active tokens."""
        with self._lock:
            return len(self._active_tokens)