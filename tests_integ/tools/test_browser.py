from bedrock_agentcore.tools.browser_client import browser_session

with browser_session("us-west-2") as client:
    assert client.session_id is not None
    url, headers = client.generate_ws_headers()
    assert url.startswith("wss")

    url = client.generate_live_view_url()
    assert url.startswith("https")

    client.take_control()
    client.release_control()
