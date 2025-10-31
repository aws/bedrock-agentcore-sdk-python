"""Integration tests for middleware."""

import subprocess
import time
import requests
import pytest


@pytest.fixture(scope="module")
def middleware_server(tmp_path_factory):
    """Start a real server with middleware."""
    tmp_dir = tmp_path_factory.mktemp("agent")
    agent_file = tmp_dir / "test_agent.py"
    
    # Write agent with middleware
    agent_file.write_text("""
import time
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from bedrock_agentcore import BedrockAgentCoreApp

class StandardNamespaces:
    AUTH = "auth"
    TIMING = "timing"
    RATE_LIMIT = "rate_limit"

class AuthMiddleware(BaseHTTPMiddleware):
    NAMESPACE = StandardNamespaces.AUTH
    
    VALID_TOKENS = {
        'test-admin-token': {'user_id': 'admin-1', 'role': 'admin'},
        'test-user-token': {'user_id': 'user-1', 'role': 'user'},
    }
    
    async def dispatch(self, request, call_next):
        if not hasattr(request.state, 'processing_data'):
            request.state.processing_data = {}
        
        auth_header = request.headers.get('authorization', '')
        
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            user_info = self.VALID_TOKENS.get(token)
            
            if user_info:
                request.state.processing_data[self.NAMESPACE] = {
                    'authenticated': True,
                    **user_info
                }
            else:
                request.state.processing_data[self.NAMESPACE] = {
                    'authenticated': False,
                    'error': 'Invalid token'
                }
        else:
            request.state.processing_data[self.NAMESPACE] = {
                'authenticated': False,
                'error': 'No authorization header'
            }
        
        return await call_next(request)

class TimingMiddleware(BaseHTTPMiddleware):
    NAMESPACE = StandardNamespaces.TIMING
    
    async def dispatch(self, request, call_next):
        if not hasattr(request.state, 'processing_data'):
            request.state.processing_data = {}
        
        start_time = time.time()
        request.state.processing_data[self.NAMESPACE] = {
            'start_time': start_time
        }
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        response.headers['X-Processing-Time'] = f"{duration:.3f}"
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    NAMESPACE = StandardNamespaces.RATE_LIMIT
    
    def __init__(self, app, max_requests=5):
        super().__init__(app)
        self.max_requests = max_requests
        self.request_counts = {}
    
    async def dispatch(self, request, call_next):
        if not hasattr(request.state, 'processing_data'):
            request.state.processing_data = {}
        
        auth_data = request.state.processing_data.get(StandardNamespaces.AUTH, {})
        user_id = auth_data.get('user_id', 'anonymous')
        
        current_time = time.time()
        
        self.request_counts = {
            uid: [t for t in times if current_time - t < 60]
            for uid, times in self.request_counts.items()
        }
        
        user_requests = self.request_counts.get(user_id, [])
        
        if len(user_requests) >= self.max_requests:
            request.state.processing_data[self.NAMESPACE] = {
                'allowed': False,
                'limit': self.max_requests
            }
        else:
            user_requests.append(current_time)
            self.request_counts[user_id] = user_requests
            
            request.state.processing_data[self.NAMESPACE] = {
                'allowed': True,
                'remaining': self.max_requests - len(user_requests)
            }
        
        return await call_next(request)

app = BedrockAgentCoreApp(
    middleware=[
        Middleware(TimingMiddleware),
        Middleware(AuthMiddleware),
        Middleware(RateLimitMiddleware, max_requests=5),
    ]
)

@app.entrypoint
def handler(payload, context):
    authenticated = context.processing.get(StandardNamespaces.AUTH, 'authenticated', False)
    
    if not authenticated:
        error = context.processing.get(StandardNamespaces.AUTH, 'error', 'Unknown')
        return {"error": "Authentication required", "details": error}
    
    rate_allowed = context.processing.get(StandardNamespaces.RATE_LIMIT, 'allowed', True)
    
    if not rate_allowed:
        limit = context.processing.get(StandardNamespaces.RATE_LIMIT, 'limit')
        return {"error": "Rate limit exceeded", "limit": limit}
    
    user_id = context.processing.get(StandardNamespaces.AUTH, 'user_id')
    role = context.processing.get(StandardNamespaces.AUTH, 'role')
    remaining = context.processing.get(StandardNamespaces.RATE_LIMIT, 'remaining')
    
    if payload.get('delay'):
        time.sleep(payload['delay'])
    
    return {
        "success": True,
        "user_id": user_id,
        "role": role,
        "rate_limit_remaining": remaining,
        "message": payload.get('message', 'Hello from middleware!')
    }

if __name__ == "__main__":
    app.run()
""")
    
    # Start server
    proc = subprocess.Popen(
        ["python", str(agent_file)],
        cwd=str(tmp_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    yield "http://127.0.0.1:8080"
    
    # Cleanup
    proc.terminate()
    proc.wait(timeout=5)


def test_server_ping(middleware_server):
    """Test server is running."""
    response = requests.get(f"{middleware_server}/ping")
    assert response.status_code == 200


def test_authenticated_request(middleware_server):
    """Test authenticated request succeeds."""
    response = requests.post(
        f"{middleware_server}/invocations",
        json={"message": "Hello middleware!"},
        headers={"Authorization": "Bearer test-admin-token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user_id"] == "admin-1"
    assert data["role"] == "admin"
    assert "rate_limit_remaining" in data


def test_unauthenticated_request(middleware_server):
    """Test unauthenticated request is rejected."""
    response = requests.post(
        f"{middleware_server}/invocations",
        json={"message": "This should fail"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "Authentication required" in data["error"]


def test_invalid_token(middleware_server):
    """Test invalid token is rejected."""
    response = requests.post(
        f"{middleware_server}/invocations",
        json={"message": "Invalid token"},
        headers={"Authorization": "Bearer invalid-token"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "Invalid token" in data["details"]


def test_timing_header(middleware_server):
    """Test timing middleware adds header."""
    response = requests.post(
        f"{middleware_server}/invocations",
        json={"message": "Timing test", "delay": 0.1},
        headers={"Authorization": "Bearer test-user-token"}
    )
    assert response.status_code == 200
    assert "X-Processing-Time" in response.headers
    processing_time = float(response.headers["X-Processing-Time"])
    assert processing_time >= 0.1


def test_rate_limiting(middleware_server):
    """Test rate limiting with multiple requests."""
    # Use test-user-token which hasn't hit rate limits yet
    headers = {"Authorization": "Bearer test-user-token"}
    
    # Track how many are remaining from first request
    response = requests.post(
        f"{middleware_server}/invocations",
        json={"message": "Request 1"},
        headers=headers
    )
    data = response.json()
    assert data["success"] is True
    
    # Get starting count
    first_remaining = data["rate_limit_remaining"]
    
    # Make more requests until we hit the limit
    for i in range(first_remaining):
        response = requests.post(
            f"{middleware_server}/invocations",
            json={"message": f"Request {i+2}"},
            headers=headers
        )
        data = response.json()
        assert data["success"] is True
    
    # Next request should be rate limited
    response = requests.post(
        f"{middleware_server}/invocations",
        json={"message": "Rate limited"},
        headers=headers
    )
    data = response.json()
    assert "error" in data
    assert "Rate limit exceeded" in data["error"]
