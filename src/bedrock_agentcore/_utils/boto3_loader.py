"""Botocore loader helpers for bundled AgentCore service models.

Provides targeted injection so each client only loads the model it needs:
- _inject_eval_cp_models: agentcore-evaluation-controlplane (2024-01-01)
- _inject_eval_dp_models: agentcore-evaluation-dataplane (2022-07-26)

TODO: Remove once the service models ship in an official botocore release.
"""

import os
from typing import Optional

import boto3

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
_EVAL_CP_DATA_DIR = os.path.join(_BASE, "eval_cp")
_EVAL_DP_DATA_DIR = os.path.join(_BASE, "eval_dp")


def _inject(session: boto3.Session, data_dir: str) -> None:
    loader = session._session.get_component("data_loader")
    if data_dir not in loader.search_paths:
        loader.search_paths.insert(0, data_dir)


def _inject_eval_cp_models(session: Optional[boto3.Session] = None) -> boto3.Session:
    """Return a boto3 Session with the bundled agentcore-evaluation-controlplane model registered."""
    if session is None:
        session = boto3.Session()
    _inject(session, _EVAL_CP_DATA_DIR)
    return session


def _inject_eval_dp_models(session: Optional[boto3.Session] = None) -> boto3.Session:
    """Return a boto3 Session with the bundled agentcore-evaluation-dataplane model registered."""
    if session is None:
        session = boto3.Session()
    _inject(session, _EVAL_DP_DATA_DIR)
    return session
