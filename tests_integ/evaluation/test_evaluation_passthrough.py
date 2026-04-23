"""Integration tests for EvaluationClient passthrough and *_and_wait methods."""

import os
import time

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.evaluation.client import EvaluationClient


@pytest.mark.integration
class TestEvaluationClientPassthrough:
    """Read-only passthrough tests. No resources needed."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = EvaluationClient(region_name=cls.region)

    @pytest.mark.order(1)
    def test_list_evaluators_passthrough(self):
        response = self.client.list_evaluators()
        assert "evaluators" in response

    @pytest.mark.order(2)
    def test_list_evaluators_snake_case(self):
        response = self.client.list_evaluators(max_results=5)
        assert "evaluators" in response

    @pytest.mark.order(3)
    def test_get_builtin_evaluator(self):
        response = self.client.get_evaluator(evaluatorId="Builtin.Helpfulness")
        assert response["evaluatorId"] == "Builtin.Helpfulness"
        assert response["level"] in ("SESSION", "TRACE", "TOOL_CALL")

    @pytest.mark.order(4)
    def test_list_online_evaluation_configs_passthrough(self):
        response = self.client.list_online_evaluation_configs()
        assert "onlineEvaluationConfigs" in response

    @pytest.mark.order(5)
    def test_non_allowlisted_method_raises(self):
        with pytest.raises(AttributeError):
            self.client.not_a_real_method()


@pytest.mark.integration
class TestEvaluatorCrud:
    """CRUD tests for custom evaluators."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = EvaluationClient(region_name=cls.region)
        cls.test_prefix = f"sdk_integ_{int(time.time())}"
        cls.evaluator_ids = []

    @classmethod
    def teardown_class(cls):
        for eid in cls.evaluator_ids:
            try:
                cls.client.delete_evaluator(evaluatorId=eid)
            except Exception as e:
                print(f"Failed to delete evaluator {eid}: {e}")

    @pytest.mark.order(6)
    def test_create_evaluator_and_wait(self):
        evaluator = self.client.create_evaluator_and_wait(
            evaluatorName=f"{self.test_prefix}_eval",
            level="SESSION",
            evaluatorConfig={
                "llmAsAJudge": {
                    "instructions": "Rate the helpfulness of the response. {context}",
                    "ratingScale": {
                        "numerical": [
                            {"definition": "Not helpful", "value": 1, "label": "Poor"},
                            {"definition": "Very helpful", "value": 5, "label": "Excellent"},
                        ]
                    },
                    "modelConfig": {
                        "bedrockEvaluatorModelConfig": {
                            "modelId": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        }
                    },
                }
            },
        )
        self.__class__.evaluator_ids.append(evaluator["evaluatorId"])
        assert evaluator["status"] == "ACTIVE"

    @pytest.mark.order(7)
    def test_get_evaluator_passthrough(self):
        if not self.evaluator_ids:
            pytest.skip("prerequisite test did not create evaluator")
        evaluator = self.client.get_evaluator(
            evaluatorId=self.evaluator_ids[0],
        )
        assert evaluator["status"] == "ACTIVE"

    @pytest.mark.order(8)
    def test_update_evaluator_and_wait(self):
        if not self.evaluator_ids:
            pytest.skip("prerequisite test did not create evaluator")
        updated = self.client.update_evaluator_and_wait(
            evaluatorId=self.evaluator_ids[0],
            description="updated by integ test",
        )
        assert updated["status"] == "ACTIVE"

    @pytest.mark.order(9)
    def test_delete_evaluator_and_wait(self):
        if not self.evaluator_ids:
            pytest.skip("prerequisite test did not create evaluator")
        eid = self.evaluator_ids.pop(0)
        self.client.delete_evaluator_and_wait(evaluatorId=eid)
        with pytest.raises(ClientError):
            self.client.get_evaluator(evaluatorId=eid)


@pytest.mark.integration
class TestOnlineEvaluationConfigCrud:
    """CRUD tests for online evaluation configs.

    Requires EVAL_ROLE_ARN and EVAL_LOG_GROUP environment variables.
    """

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.role_arn = os.environ.get("EVAL_ROLE_ARN")
        cls.log_group = os.environ.get("EVAL_LOG_GROUP")
        if not cls.role_arn or not cls.log_group:
            pytest.skip("EVAL_ROLE_ARN and EVAL_LOG_GROUP must be set")
        cls.client = EvaluationClient(region_name=cls.region)
        cls.test_prefix = f"sdk_integ_{int(time.time())}"
        cls.config_ids = []

    @classmethod
    def teardown_class(cls):
        for cid in cls.config_ids:
            try:
                cls.client.delete_online_evaluation_config(
                    onlineEvaluationConfigId=cid,
                )
            except Exception as e:
                print(f"Failed to delete config {cid}: {e}")

    @pytest.mark.order(10)
    def test_create_online_eval_config_and_wait(self):
        config = self.client.create_online_evaluation_config_and_wait(
            onlineEvaluationConfigName=f"{self.test_prefix}_config",
            rule={"samplingConfig": {"samplingPercentage": 100.0}},
            dataSourceConfig={
                "cloudWatchLogs": {
                    "logGroupNames": [self.log_group],
                    "serviceNames": ["sdk-integ-test"],
                }
            },
            evaluators=[{"evaluatorId": "Builtin.Helpfulness"}],
            evaluationExecutionRoleArn=self.role_arn,
            enableOnCreate=False,
        )
        self.__class__.config_ids.append(config["onlineEvaluationConfigId"])
        assert config["status"] == "ACTIVE"

    @pytest.mark.order(11)
    def test_get_online_eval_config_passthrough(self):
        if not self.config_ids:
            pytest.skip("prerequisite test did not create config")
        config = self.client.get_online_evaluation_config(
            onlineEvaluationConfigId=self.config_ids[0],
        )
        assert config["status"] == "ACTIVE"

    @pytest.mark.order(12)
    def test_update_online_eval_config_and_wait(self):
        if not self.config_ids:
            pytest.skip("prerequisite test did not create config")
        updated = self.client.update_online_evaluation_config_and_wait(
            onlineEvaluationConfigId=self.config_ids[0],
            description="updated by integ test",
        )
        assert updated["status"] == "ACTIVE"

    @pytest.mark.order(13)
    def test_delete_online_eval_config_and_wait(self):
        if not self.config_ids:
            pytest.skip("prerequisite test did not create config")
        cid = self.config_ids.pop(0)
        self.client.delete_online_evaluation_config_and_wait(
            onlineEvaluationConfigId=cid,
        )
        with pytest.raises(ClientError):
            self.client.get_online_evaluation_config(
                onlineEvaluationConfigId=cid,
            )
