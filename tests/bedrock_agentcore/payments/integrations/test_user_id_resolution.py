"""Regression tests for defense-in-depth user_id resolution.

The Strands plugin and LangGraph tools expose ``user_id`` as an LLM-controllable
tool argument. A prompt-injected agent could supply a different ``user_id`` to attempt
an identity override. The managed service already scopes every read to the server-resolved
identity, so this never crossed a tenant boundary -- but the SDK should not let the model
attempt it. ``resolve_authoritative_user_id`` makes the configured identity authoritative
and ignores (and logs) a differing tool-supplied value.
"""

import logging

from bedrock_agentcore.payments.integrations.config import resolve_authoritative_user_id

_LOGGER_NAME = "bedrock_agentcore.payments.integrations.config"


def test_differing_requested_value_is_ignored():
    """A tool-supplied user_id that differs from config is ignored; config wins."""
    assert resolve_authoritative_user_id("config-user", "attacker-user") == "config-user"


def test_matching_requested_value_returns_config():
    assert resolve_authoritative_user_id("config-user", "config-user") == "config-user"


def test_no_requested_value_returns_config():
    assert resolve_authoritative_user_id("config-user", None) == "config-user"


def test_whitespace_requested_value_returns_config():
    assert resolve_authoritative_user_id("config-user", "   ") == "config-user"


def test_requested_value_is_stripped_before_compare():
    """A whitespace-padded value equal to config is treated as matching, not a mismatch."""
    assert resolve_authoritative_user_id("config-user", "  config-user  ") == "config-user"


def test_no_config_passes_through_requested_value():
    """Bearer/JWT auth: no configured identity, so the server resolves from the token.

    Nothing to override, so the (stripped) requested value is passed through.
    """
    assert resolve_authoritative_user_id(None, "  bearer-user  ") == "bearer-user"


def test_no_config_and_no_requested_returns_none():
    assert resolve_authoritative_user_id(None, None) is None


def test_mismatch_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        resolve_authoritative_user_id("config-user", "attacker-user")
    assert any("Ignoring tool-supplied user_id" in r.message for r in caplog.records)
    # the attacker value is surfaced for audit, the configured identity is not leaked as "ignored"
    assert any("attacker-user" in r.getMessage() for r in caplog.records)


def test_no_warning_when_values_match(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        resolve_authoritative_user_id("config-user", "config-user")
    assert not any("Ignoring tool-supplied user_id" in r.message for r in caplog.records)


def test_no_warning_when_no_config(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        resolve_authoritative_user_id(None, "bearer-user")
    assert not any("Ignoring tool-supplied user_id" in r.message for r in caplog.records)
