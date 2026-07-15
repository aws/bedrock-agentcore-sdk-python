"""E2E tests: Online eval mode (Tests 22-23) — P2 stretch.

Tests the full online evaluation pipeline:
  Agent invocation → ADOT sidecar → CloudWatch → evaluation trigger → score

These tests take 20-30 minutes each due to span propagation delay.
Run manually for final validation, not as part of regular test suite.

Usage:
    .venv/bin/python -m pytest e2e_tests/test_online_eval.py -v --timeout=1800
"""

import pytest

from e2e_tests.conftest import EVALUATOR_IDS, assert_success, run_evaluation


@pytest.mark.p2
@pytest.mark.skip(reason="Online eval takes 20-30 min — run manually for final validation")
class TestOnlineEval:
    """Tests 22-23: Online evaluation pipeline (P2 stretch)."""

    def test_deepeval_online(self, dp_client):
        """Test 22: DeepEval AnswerRelevancy via online eval pipeline."""
        # This test requires:
        # 1. Agent deployed with ADOT sidecar
        # 2. Online eval config enabled (sampling rate 100%)
        # 3. Agent invoked → spans exported to CloudWatch
        # 4. Wait for evaluation pipeline to trigger (~20-30 min)
        # 5. Check evaluation result
        pytest.skip("Implement after online eval config is set up")

    def test_autoevals_online(self, dp_client):
        """Test 23: Autoevals Factuality via online eval pipeline."""
        pytest.skip("Implement after online eval config is set up")
