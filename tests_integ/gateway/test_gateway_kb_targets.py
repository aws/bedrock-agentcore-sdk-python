"""Integration tests for GatewayClient KB target helper methods.

Requires environment variables:
    BEDROCK_TEST_REGION: AWS region (default: us-east-1)
    GATEWAY_ROLE_ARN: IAM role ARN with AgentCore gateway trust policy
    KB_ROLE_ARN: IAM role ARN with bedrock:InvokeModel, s3:*, and s3vectors:* permissions
"""

import os
import time
import uuid

import boto3
import pytest

from bedrock_agentcore.gateway.client import GatewayClient
from bedrock_agentcore.knowledge_base.client import KnowledgeBaseClient


@pytest.mark.integration
class TestGatewayKnowledgeBaseTarget:
    """Integration tests for create_knowledge_base_target."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.gateway_role_arn = os.environ.get("GATEWAY_ROLE_ARN")
        cls.kb_role_arn = os.environ.get("KB_ROLE_ARN")
        if not cls.gateway_role_arn or not cls.kb_role_arn:
            pytest.fail("GATEWAY_ROLE_ARN and KB_ROLE_ARN must be set")

        cls.gateway_client = GatewayClient(region_name=cls.region)
        cls.kb_client = KnowledgeBaseClient(region_name=cls.region)
        cls.test_suffix = uuid.uuid4().hex[:8]
        cls.test_prefix = f"sdk-integ-kb-tgt-{int(time.time())}"
        cls.gateway_id = None
        cls.kb_id = None
        cls.target_ids = []

        # Create S3 Vectors resources
        cls.s3vectors_client = boto3.client("s3vectors", region_name=cls.region)
        cls.vector_bucket_name = f"kb-gw-integ-vb-{cls.test_suffix}"
        cls.vector_index_name = f"kb-gw-integ-idx-{cls.test_suffix}"
        cls.s3vectors_client.create_vector_bucket(vectorBucketName=cls.vector_bucket_name)
        index_resp = cls.s3vectors_client.create_index(
            vectorBucketName=cls.vector_bucket_name,
            indexName=cls.vector_index_name,
            dataType="float32",
            dimension=1024,
            distanceMetric="cosine",
        )
        cls.index_arn = index_resp["indexArn"]

        # Create Knowledge Base
        cls.kb = cls.kb_client.create_knowledge_base_and_wait(
            name=f"{cls.test_prefix}-kb",
            roleArn=cls.kb_role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{cls.region}::foundation-model/amazon.titan-embed-text-v2:0",
                },
            },
            storageConfiguration={
                "type": "S3_VECTORS",
                "s3VectorsConfiguration": {
                    "indexArn": cls.index_arn,
                },
            },
        )
        cls.kb_id = cls.kb["knowledgeBaseId"]

        # Create Gateway
        gw = cls.gateway_client.create_gateway_and_wait(
            name=f"{cls.test_prefix}-gw",
            roleArn=cls.gateway_role_arn,
            authorizerType="NONE",
            protocolType="MCP",
        )
        cls.gateway_id = gw["gatewayId"]

    @classmethod
    def teardown_class(cls):
        # Delete targets
        for target_id in cls.target_ids:
            try:
                cls.gateway_client.delete_gateway_target_and_wait(
                    gatewayIdentifier=cls.gateway_id,
                    targetId=target_id,
                )
            except Exception as e:
                print(f"Failed to delete target {target_id}: {e}")

        # Delete gateway
        if cls.gateway_id:
            try:
                cls.gateway_client.delete_gateway_and_wait(gatewayIdentifier=cls.gateway_id)
            except Exception as e:
                print(f"Failed to delete gateway {cls.gateway_id}: {e}")

        # Delete KB
        if cls.kb_id:
            try:
                cls.kb_client.delete_knowledge_base_and_wait(knowledgeBaseId=cls.kb_id)
            except Exception as e:
                print(f"Failed to delete KB {cls.kb_id}: {e}")

        # Delete S3 Vectors
        try:
            cls.s3vectors_client.delete_index(
                vectorBucketName=cls.vector_bucket_name,
                indexName=cls.vector_index_name,
            )
            cls.s3vectors_client.delete_vector_bucket(vectorBucketName=cls.vector_bucket_name)
        except Exception as e:
            print(f"Failed to clean up vector bucket: {e}")

    @pytest.mark.order(1)
    def test_create_knowledge_base_target_minimal(self):
        target = self.gateway_client.create_knowledge_base_target(
            gateway_identifier=self.gateway_id,
            knowledge_base_id=self.kb_id,
        )
        self.__class__.target_ids.append(target["targetId"])
        assert target["status"] == "READY"
        assert target["name"] == f"kb-{self.kb_id}"

    @pytest.mark.order(2)
    def test_create_knowledge_base_target_with_options(self):
        target = self.gateway_client.create_knowledge_base_target(
            gateway_identifier=self.gateway_id,
            knowledge_base_id=self.kb_id,
            name=f"{self.test_prefix}-custom",
            description="Search the test KB",
            retrieval_configuration={"vectorSearchConfiguration": {"numberOfResults": 3}},
            parameter_overrides=[{"path": "$.retrievalQuery.text", "visible": True, "description": "The search query"}],
        )
        self.__class__.target_ids.append(target["targetId"])
        assert target["status"] == "READY"
        assert target["name"] == f"{self.test_prefix}-custom"

    @pytest.mark.order(3)
    def test_create_knowledge_base_target_with_credential_config(self):
        target = self.gateway_client.create_knowledge_base_target(
            gateway_identifier=self.gateway_id,
            knowledge_base_id=self.kb_id,
            name=f"{self.test_prefix}-cred",
            credentialProviderConfigurations=[
                {"credentialProviderType": "GATEWAY_IAM_ROLE"},
            ],
        )
        self.__class__.target_ids.append(target["targetId"])
        assert target["status"] == "READY"
