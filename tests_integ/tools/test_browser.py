"""Integration tests for browser client.

Note: These tests require valid AWS credentials and may incur costs.
To run: pytest tests_integ/tools/test_browser.py -v
"""

from bedrock_agentcore.tools.browser_client import browser_session

# Test 1: Basic browser session with system browser
print("Test 1: Basic system browser session")
with browser_session("us-west-2") as client:
    assert client.session_id is not None
    url, headers = client.generate_ws_headers()
    assert url.startswith("wss")

    url = client.generate_live_view_url()
    assert url.startswith("https")

    client.take_control()
    client.release_control()
print("✅ Test 1 passed")

# Test 2: Browser session with viewport
print("\nTest 2: Browser session with custom viewport")
with browser_session("us-west-2", viewport={"width": 1280, "height": 720}) as client:
    assert client.session_id is not None
    url, headers = client.generate_ws_headers()
    assert url.startswith("wss")
print("✅ Test 2 passed")
