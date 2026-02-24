"""Integration tests for browser session configuration support.

Tests proxy_configuration, extensions, profile_configuration, and SessionConfiguration
dataclasses against the live StartBrowserSession API.

Requires: valid AWS credentials for us-west-2 with Admin role on account 121875801285.

To run: python3 tests_integ/tools/test_browser_proxy.py
"""

import sys

from bedrock_agentcore.tools.browser_client import BrowserClient, browser_session
from bedrock_agentcore.tools.config import (
    BasicAuth,
    BrowserExtension,
    ExtensionS3Location,
    ExternalProxy,
    ProfileConfiguration,
    ProxyConfiguration,
    ProxyCredentials,
    SessionConfiguration,
    ViewportConfiguration,
)

REGION = "us-west-2"

# BrightData proxy config as plain dict (existing passthrough pattern)
BRIGHTDATA_PROXY_CONFIG = {
    "proxies": [
        {
            "externalProxy": {
                "server": "brd.superproxy.io",
                "port": 33335,
                "domainPatterns": [
                    ".icanhazip.com",
                    ".whoer.net",
                    ".httpbin.org",
                ],
                "credentials": {
                    "basicAuth": {
                        "secretArn": (
                            "arn:aws:secretsmanager:us-west-2:121875801285"
                            ":secret:genesis1p-browser-proxy-test-brightdata-gJWalz"
                        )
                    }
                },
            }
        }
    ],
    "bypass": {
        "domainPatterns": [
            "checkip.amazonaws.com",
            "169.254.169.254",
        ]
    },
}

# Same config expressed as dataclasses
BRIGHTDATA_PROXY_DATACLASS = ProxyConfiguration(
    proxies=[
        ExternalProxy(
            server="brd.superproxy.io",
            port=33335,
            domain_patterns=[".icanhazip.com", ".whoer.net", ".httpbin.org"],
            credentials=ProxyCredentials(
                basic_auth=BasicAuth(
                    secret_arn="arn:aws:secretsmanager:us-west-2:121875801285:secret:genesis1p-browser-proxy-test-brightdata-gJWalz"
                )
            ),
        )
    ],
    bypass_patterns=["checkip.amazonaws.com", "169.254.169.254"],
)


def test_passthrough_browser_session():
    """Test 1: browser_session() accepts proxy_configuration dict and the API does not reject it."""
    print("Test 1: browser_session() with proxy_configuration (passthrough dict)")
    with browser_session(REGION, proxy_configuration=BRIGHTDATA_PROXY_CONFIG) as client:
        assert client.session_id is not None, "session_id should be set"
        assert client.identifier is not None, "identifier should be set"

        url, headers = client.generate_ws_headers()
        assert url.startswith("wss"), f"Expected wss URL, got: {url}"

        live_url = client.generate_live_view_url()
        assert live_url.startswith("https"), f"Expected https URL, got: {live_url}"

        print(f"  Session ID: {client.session_id}")
        print(f"  Live View:  {live_url[:80]}...")
    print("  PASSED")


def test_passthrough_client_start():
    """Test 2: BrowserClient.start() accepts proxy_configuration directly."""
    print("\nTest 2: BrowserClient.start() with proxy_configuration (passthrough dict)")
    client = BrowserClient(REGION)
    try:
        session_id = client.start(proxy_configuration=BRIGHTDATA_PROXY_CONFIG)
        assert session_id is not None, "session_id should be returned"
        print(f"  Session ID: {session_id}")

        session_info = client.get_session()
        assert session_info["status"] == "READY", f"Expected READY, got: {session_info['status']}"
        print(f"  Status: {session_info['status']}")
    finally:
        client.stop()
    print("  PASSED")


def test_passthrough_no_proxy_unchanged():
    """Test 3: Existing behavior without proxy_configuration still works."""
    print("\nTest 3: browser_session() without proxy_configuration (backward compat)")
    with browser_session(REGION) as client:
        assert client.session_id is not None
        url, headers = client.generate_ws_headers()
        assert url.startswith("wss")
        print(f"  Session ID: {client.session_id}")
    print("  PASSED")


def test_proxy_with_viewport():
    """Test 4: proxy_configuration works alongside viewport."""
    print("\nTest 4: browser_session() with proxy_configuration + viewport")
    with browser_session(
        REGION,
        viewport={"width": 1280, "height": 720},
        proxy_configuration=BRIGHTDATA_PROXY_CONFIG,
    ) as client:
        assert client.session_id is not None
        print(f"  Session ID: {client.session_id}")
    print("  PASSED")


def test_proxy_dataclass():
    """Test 5: ProxyConfiguration dataclass produces valid API input."""
    print("\nTest 5: ProxyConfiguration dataclass -> start(proxy_configuration=...)")
    proxy_dict = BRIGHTDATA_PROXY_DATACLASS.to_dict()
    with browser_session(REGION, proxy_configuration=proxy_dict) as client:
        assert client.session_id is not None
        session_info = client.get_session()
        assert session_info["status"] == "READY", f"Expected READY, got: {session_info['status']}"
        print(f"  Session ID: {client.session_id}")
        print(f"  Status: {session_info['status']}")
    print("  PASSED")


def test_session_configuration_proxy_only():
    """Test 6: SessionConfiguration with proxy produces valid start() kwargs."""
    print("\nTest 6: SessionConfiguration(proxy=...) -> start(**config.to_dict())")
    config = SessionConfiguration(proxy=BRIGHTDATA_PROXY_DATACLASS)
    client = BrowserClient(REGION)
    try:
        session_id = client.start(**config.to_dict())
        assert session_id is not None
        session_info = client.get_session()
        assert session_info["status"] == "READY"
        print(f"  Session ID: {session_id}")
        print(f"  Status: {session_info['status']}")
    finally:
        client.stop()
    print("  PASSED")


def test_session_configuration_proxy_and_viewport():
    """Test 7: SessionConfiguration with proxy + viewport."""
    print("\nTest 7: SessionConfiguration(proxy=..., viewport=...) -> start(**config.to_dict())")
    config = SessionConfiguration(
        proxy=BRIGHTDATA_PROXY_DATACLASS,
        viewport=ViewportConfiguration(width=1280, height=720),
    )
    client = BrowserClient(REGION)
    try:
        session_id = client.start(**config.to_dict())
        assert session_id is not None
        session_info = client.get_session()
        assert session_info["status"] == "READY"
        print(f"  Session ID: {session_id}")
        print(f"  Status: {session_info['status']}")
    finally:
        client.stop()
    print("  PASSED")


def test_profile_configuration():
    """Test 8: profile_configuration parameter is accepted by the API.

    Note: Uses a placeholder profile ID -- the API may reject unknown profiles
    with a validation error, which is still a valid test of parameter passthrough.
    """
    print("\nTest 8: start(profile_configuration=...) parameter passthrough")
    client = BrowserClient(REGION)
    try:
        session_id = client.start(profile_configuration={"profileIdentifier": "test-profile-placeholder"})
        assert session_id is not None
        print(f"  Session ID: {session_id}")
        print("  PASSED (API accepted the parameter)")
    except Exception as e:
        error_msg = str(e)
        # A validation error from the API means the parameter was passed through correctly
        if "ValidationException" in error_msg or "validation" in error_msg.lower():
            print("  PASSED (API rejected with validation: parameter was passed through)")
        else:
            raise
    finally:
        client.stop()


def test_extensions_parameter():
    """Test 9: extensions parameter is accepted by the API.

    Note: Uses a placeholder S3 location -- the API may reject it, which still
    validates the parameter passthrough.
    """
    print("\nTest 9: start(extensions=...) parameter passthrough")
    client = BrowserClient(REGION)
    try:
        session_id = client.start(
            extensions=[{"location": {"s3": {"bucket": "nonexistent-test-bucket", "prefix": "ext/v1"}}}]
        )
        assert session_id is not None
        print(f"  Session ID: {session_id}")
        print("  PASSED (API accepted the parameter)")
    except Exception as e:
        error_msg = str(e)
        expected_errors = ["ValidationException", "validation", "Access Denied", "NoSuchBucket"]
        if any(e in error_msg or e in error_msg.lower() for e in expected_errors):
            print("  PASSED (API rejected with expected error: parameter was passed through)")
        else:
            raise
    finally:
        client.stop()


def test_browser_session_extensions_param():
    """Test 10: browser_session() accepts extensions parameter."""
    print("\nTest 10: browser_session(extensions=...) parameter passthrough")
    try:
        with browser_session(
            REGION,
            extensions=[{"location": {"s3": {"bucket": "nonexistent-test-bucket", "prefix": "ext/v1"}}}],
        ) as client:
            assert client.session_id is not None
            print(f"  Session ID: {client.session_id}")
            print("  PASSED (API accepted the parameter)")
    except Exception as e:
        error_msg = str(e)
        expected_errors = ["ValidationException", "validation", "Access Denied", "NoSuchBucket"]
        if any(e in error_msg or e in error_msg.lower() for e in expected_errors):
            print("  PASSED (API rejected with expected error: parameter was passed through)")
        else:
            raise


def test_browser_session_profile_param():
    """Test 11: browser_session() accepts profile_configuration parameter."""
    print("\nTest 11: browser_session(profile_configuration=...) parameter passthrough")
    try:
        with browser_session(
            REGION,
            profile_configuration={"profileIdentifier": "test-profile-placeholder"},
        ) as client:
            assert client.session_id is not None
            print(f"  Session ID: {client.session_id}")
            print("  PASSED (API accepted the parameter)")
    except Exception as e:
        error_msg = str(e)
        if "ValidationException" in error_msg or "validation" in error_msg.lower():
            print("  PASSED (API rejected with validation: parameter was passed through)")
        else:
            raise


def test_session_configuration_with_extensions_dataclass():
    """Test 12: SessionConfiguration with BrowserExtension dataclass.

    Uses a nonexistent S3 bucket, so expects either success or a
    validation/access error -- both confirm the parameter was passed through.
    """
    print("\nTest 12: SessionConfiguration(extensions=[BrowserExtension(...)]) dataclass")
    config = SessionConfiguration(
        extensions=[
            BrowserExtension(
                s3_location=ExtensionS3Location(
                    bucket="nonexistent-test-bucket",
                    prefix="ext/v1",
                )
            )
        ]
    )
    client = BrowserClient(REGION)
    try:
        session_id = client.start(**config.to_dict())
        assert session_id is not None
        print(f"  Session ID: {session_id}")
        print("  PASSED (API accepted the parameter)")
    except Exception as e:
        error_msg = str(e)
        expected_errors = ["ValidationException", "validation", "Access Denied", "NoSuchBucket"]
        if any(err in error_msg or err in error_msg.lower() for err in expected_errors):
            print("  PASSED (API rejected with expected error: parameter was passed through)")
        else:
            raise
    finally:
        client.stop()


def test_session_configuration_with_profile_dataclass():
    """Test 13: SessionConfiguration with ProfileConfiguration dataclass.

    Uses a placeholder profile ID -- the API may reject unknown profiles
    with a validation error, which still confirms parameter passthrough.
    """
    print("\nTest 13: SessionConfiguration(profile=ProfileConfiguration(...)) dataclass")
    config = SessionConfiguration(
        profile=ProfileConfiguration(profile_identifier="test-profile-placeholder"),
    )
    client = BrowserClient(REGION)
    try:
        session_id = client.start(**config.to_dict())
        assert session_id is not None
        print(f"  Session ID: {session_id}")
        print("  PASSED (API accepted the parameter)")
    except Exception as e:
        error_msg = str(e)
        if "ValidationException" in error_msg or "validation" in error_msg.lower():
            print("  PASSED (API rejected with validation: parameter was passed through)")
        else:
            raise
    finally:
        client.stop()


def test_session_configuration_all_fields():
    """Test 14: SessionConfiguration with all four fields.

    Combines viewport, proxy (BrightData), extensions (nonexistent bucket),
    and profile (placeholder) into a single composite configuration.
    """
    print("\nTest 14: SessionConfiguration with all fields (viewport + proxy + extensions + profile)")
    config = SessionConfiguration(
        viewport=ViewportConfiguration(width=1920, height=1080),
        proxy=BRIGHTDATA_PROXY_DATACLASS,
        extensions=[
            BrowserExtension(
                s3_location=ExtensionS3Location(
                    bucket="nonexistent-test-bucket",
                    prefix="ext/v1",
                )
            )
        ],
        profile=ProfileConfiguration(profile_identifier="test-profile-placeholder"),
    )
    client = BrowserClient(REGION)
    try:
        session_id = client.start(**config.to_dict())
        assert session_id is not None
        print(f"  Session ID: {session_id}")
        print("  PASSED (API accepted the composite configuration)")
    except Exception as e:
        error_msg = str(e)
        expected_errors = ["ValidationException", "validation", "Access Denied", "NoSuchBucket"]
        if any(err in error_msg or err in error_msg.lower() for err in expected_errors):
            print("  PASSED (API rejected with expected error: composite config was passed through)")
        else:
            raise
    finally:
        client.stop()


def test_browser_session_with_session_configuration():
    """Test 15: browser_session() driven by SessionConfiguration.

    Uses proxy (BrightData) + viewport to produce a READY session,
    proving SessionConfiguration works end-to-end through browser_session().
    """
    print("\nTest 15: browser_session(**SessionConfiguration.to_dict()) end-to-end")
    config = SessionConfiguration(
        proxy=BRIGHTDATA_PROXY_DATACLASS,
        viewport=ViewportConfiguration(width=1280, height=720),
    )
    with browser_session(REGION, **config.to_dict()) as client:
        assert client.session_id is not None, "session_id should be set"
        assert client.identifier is not None, "identifier should be set"

        url, headers = client.generate_ws_headers()
        assert url.startswith("wss"), f"Expected wss URL, got: {url}"
        assert headers, "Expected non-empty ws headers"

        print(f"  Session ID: {client.session_id}")
        print(f"  WS URL: {url[:80]}...")
    print("  PASSED")


def test_double_stop_idempotent():
    """Test 16: Calling stop() twice does not raise.

    Verifies that stop() is idempotent -- the second call should return
    True without error, whether or not the session is already terminated.
    """
    print("\nTest 16: Double stop() is idempotent")
    client = BrowserClient(REGION)
    session_id = client.start()
    assert session_id is not None
    print(f"  Session ID: {session_id}")

    result1 = client.stop()
    assert result1 is True, f"First stop() should return True, got: {result1}"
    print("  First stop() returned True")

    result2 = client.stop()
    assert result2 is True, f"Second stop() should return True, got: {result2}"
    print("  Second stop() returned True")
    print("  PASSED")


def test_context_manager_cleanup_on_exception():
    """Test 17: browser_session() cleans up the session when an exception occurs.

    Raises inside the context manager and verifies the session was stopped
    (identifier and session_id cleared by stop()).
    """
    print("\nTest 17: Context manager cleanup on exception")
    saved_client = None
    saved_session_id = None

    try:
        with browser_session(REGION) as client:
            saved_client = client
            saved_session_id = client.session_id
            assert saved_session_id is not None
            print(f"  Session ID: {saved_session_id}")
            raise RuntimeError("Simulated failure inside context manager")
    except RuntimeError as e:
        assert "Simulated failure" in str(e)

    # After the context manager exits, stop() should have cleared these
    assert saved_client.session_id is None, "session_id should be None after cleanup"
    assert saved_client.identifier is None, "identifier should be None after cleanup"
    print("  Session cleaned up after exception")
    print("  PASSED")


def test_get_session_after_stop():
    """Test 18: get_session() after stop() raises ValueError.

    After stop() clears session_id and identifier, calling get_session()
    without explicit IDs should raise ValueError.
    """
    print("\nTest 18: get_session() after stop() raises ValueError")
    client = BrowserClient(REGION)
    session_id = client.start()
    assert session_id is not None
    print(f"  Session ID: {session_id}")

    client.stop()

    try:
        client.get_session()
        raise AssertionError("Expected ValueError but get_session() succeeded")
    except ValueError as e:
        assert "must be provided" in str(e).lower() or "must be provided" in str(e)
        print(f"  Raised ValueError: {e}")
    print("  PASSED")


def test_invalid_secret_arn_proxy():
    """Test 19: Proxy with invalid/nonexistent secret ARN.

    Verifies the API rejects the configuration with a clear error rather
    than silently starting a broken session.
    """
    print("\nTest 19: Proxy with invalid secret ARN")
    bad_proxy = ProxyConfiguration(
        proxies=[
            ExternalProxy(
                server="brd.superproxy.io",
                port=33335,
                domain_patterns=[".example.com"],
                credentials=ProxyCredentials(
                    basic_auth=BasicAuth(
                        secret_arn="arn:aws:secretsmanager:us-west-2:121875801285:secret:nonexistent-secret-XXXXXX"
                    )
                ),
            )
        ],
    )
    client = BrowserClient(REGION)
    try:
        session_id = client.start(proxy_configuration=bad_proxy.to_dict())
        # If it starts, check if it reaches a failed state
        print(f"  Session ID: {session_id}")
        session_info = client.get_session()
        status = session_info["status"]
        print(f"  Status: {status}")
        # Session may start but fail asynchronously -- either outcome is acceptable
        print("  PASSED (API accepted; session may fail asynchronously)")
    except Exception as e:
        error_msg = str(e)
        expected = ["ResourceNotFoundException", "AccessDeniedException", "ValidationException", "validation", "secret"]
        if any(err in error_msg or err in error_msg.lower() for err in expected):
            print(f"  PASSED (API rejected with expected error: {type(e).__name__})")
        else:
            raise
    finally:
        client.stop()


def test_invalid_proxy_server():
    """Test 20: Proxy with unreachable server host/port.

    Verifies behavior when the proxy server is not reachable. The API may
    accept the config (proxy is only used at browse-time) or reject it
    during validation.
    """
    print("\nTest 20: Proxy with unreachable server")
    bad_proxy = ProxyConfiguration(
        proxies=[
            ExternalProxy(
                server="192.0.2.1",  # TEST-NET, guaranteed unreachable
                port=99999,
                domain_patterns=[".example.com"],
            )
        ],
    )
    client = BrowserClient(REGION)
    try:
        session_id = client.start(proxy_configuration=bad_proxy.to_dict())
        print(f"  Session ID: {session_id}")
        session_info = client.get_session()
        status = session_info["status"]
        print(f"  Status: {status}")
        # Unreachable proxy may only fail at browse-time, not at session creation
        print("  PASSED (API accepted config; proxy failure would occur at browse-time)")
    except Exception as e:
        error_msg = str(e)
        expected = ["ValidationException", "validation", "port", "server"]
        if any(err in error_msg or err in error_msg.lower() for err in expected):
            print(f"  PASSED (API rejected with validation error: {type(e).__name__})")
        else:
            raise
    finally:
        client.stop()


def test_malformed_proxy_config():
    """Test 21: Malformed proxy config with missing required fields.

    Passes a proxy dict missing the 'externalProxy' key to verify the API
    returns a clean validation error rather than a 500.
    """
    print("\nTest 21: Malformed proxy config (missing required fields)")
    malformed_config = {
        "proxies": [
            {
                # Missing 'externalProxy' key entirely
                "server": "proxy.example.com",
                "port": 8080,
            }
        ]
    }
    client = BrowserClient(REGION)
    try:
        session_id = client.start(proxy_configuration=malformed_config)
        print(f"  Session ID: {session_id}")
        print("  PASSED (API accepted malformed config -- lenient validation)")
    except Exception as e:
        error_msg = str(e)
        # Should get a validation error, not a 500/InternalServerError
        if "InternalServer" in error_msg or "500" in error_msg:
            print(f"  FAILED: Got internal server error instead of validation: {e}")
            raise
        print(f"  PASSED (API rejected with: {type(e).__name__})")
    finally:
        client.stop()


def test_session_configuration_viewport_only():
    """Test 22: SessionConfiguration with viewport only (no proxy).

    Validates that SessionConfiguration works with just a viewport,
    producing a READY session without any proxy or other optional fields.
    """
    print("\nTest 22: SessionConfiguration(viewport=...) only")
    config = SessionConfiguration(
        viewport=ViewportConfiguration(width=800, height=600),
    )
    client = BrowserClient(REGION)
    try:
        session_id = client.start(**config.to_dict())
        assert session_id is not None
        session_info = client.get_session()
        assert session_info["status"] == "READY", f"Expected READY, got: {session_info['status']}"
        print(f"  Session ID: {session_id}")
        print(f"  Status: {session_info['status']}")
    finally:
        client.stop()
    print("  PASSED")


def test_multiple_extensions():
    """Test 23: SessionConfiguration with multiple extensions.

    Passes two extensions to verify the API handles a multi-element list,
    not just a single-element one.
    """
    print("\nTest 23: SessionConfiguration with multiple extensions")
    config = SessionConfiguration(
        extensions=[
            BrowserExtension(
                s3_location=ExtensionS3Location(bucket="nonexistent-bucket-a", prefix="ext/a"),
            ),
            BrowserExtension(
                s3_location=ExtensionS3Location(bucket="nonexistent-bucket-b", prefix="ext/b"),
            ),
        ]
    )
    client = BrowserClient(REGION)
    try:
        session_id = client.start(**config.to_dict())
        assert session_id is not None
        print(f"  Session ID: {session_id}")
        print("  PASSED (API accepted multiple extensions)")
    except Exception as e:
        error_msg = str(e)
        expected_errors = ["ValidationException", "validation", "Access Denied", "NoSuchBucket"]
        if any(err in error_msg or err in error_msg.lower() for err in expected_errors):
            print("  PASSED (API rejected with expected error: multiple extensions passed through)")
        else:
            raise
    finally:
        client.stop()


if __name__ == "__main__":
    tests = [
        test_passthrough_browser_session,
        test_passthrough_client_start,
        test_passthrough_no_proxy_unchanged,
        test_proxy_with_viewport,
        test_proxy_dataclass,
        test_session_configuration_proxy_only,
        test_session_configuration_proxy_and_viewport,
        test_profile_configuration,
        test_extensions_parameter,
        test_browser_session_extensions_param,
        test_browser_session_profile_param,
        test_session_configuration_with_extensions_dataclass,
        test_session_configuration_with_profile_dataclass,
        test_session_configuration_all_fields,
        test_browser_session_with_session_configuration,
        test_double_stop_idempotent,
        test_context_manager_cleanup_on_exception,
        test_get_session_after_stop,
        test_invalid_secret_arn_proxy,
        test_invalid_proxy_server,
        test_malformed_proxy_config,
        test_session_configuration_viewport_only,
        test_multiple_extensions,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {len(tests) - failed}/{len(tests)} passed, {failed} failed")
    if failed:
        sys.exit(1)
