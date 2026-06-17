"""Integration tests for KnowledgeBaseClient.

Requires environment variables:
    BEDROCK_TEST_REGION: AWS region (default: us-east-1)
    KB_ROLE_ARN: IAM role ARN with bedrock:InvokeModel, s3:*, and s3vectors:* permissions
"""

import os
import time
import uuid

import boto3
import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.knowledge_base.client import KnowledgeBaseClient


@pytest.mark.integration
class TestKnowledgeBaseClient:
    """Integration tests for KnowledgeBaseClient CRUD and wait methods."""

    @classmethod
    def setup_class(cls):
        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-east-1")
        cls.role_arn = os.environ.get("KB_ROLE_ARN")
        if not cls.role_arn:
            pytest.fail("KB_ROLE_ARN must be set")

        cls.client = KnowledgeBaseClient(region_name=cls.region)
        cls.test_suffix = uuid.uuid4().hex[:8]
        cls.test_prefix = f"sdk-integ-{int(time.time())}"
        cls.kb_id = None
        cls.ds_id = None

        cls.s3_client = boto3.client("s3", region_name=cls.region)
        cls.s3vectors_client = boto3.client("s3vectors", region_name=cls.region)

        # Create S3 bucket for data source
        cls.bucket_name = f"kb-integ-test-{cls.test_suffix}"
        create_params = {"Bucket": cls.bucket_name}
        if cls.region != "us-east-1":
            create_params["CreateBucketConfiguration"] = {"LocationConstraint": cls.region}
        cls.s3_client.create_bucket(**create_params)
        cls.s3_client.put_object(
            Bucket=cls.bucket_name,
            Key="test-doc.txt",
            Body="This is a test document for knowledge base integration testing.",
        )

        # Create S3 Vectors bucket and index
        cls.vector_bucket_name = f"kb-integ-vb-{cls.test_suffix}"
        cls.vector_index_name = f"kb-integ-idx-{cls.test_suffix}"
        cls.s3vectors_client.create_vector_bucket(vectorBucketName=cls.vector_bucket_name)
        index_resp = cls.s3vectors_client.create_index(
            vectorBucketName=cls.vector_bucket_name,
            indexName=cls.vector_index_name,
            dataType="float32",
            dimension=1024,
            distanceMetric="cosine",
        )
        cls.index_arn = index_resp["indexArn"]

    @classmethod
    def teardown_class(cls):
        if cls.kb_id:
            try:
                cls.client.delete_knowledge_base_and_wait(knowledge_base_id=cls.kb_id)
            except Exception as e:
                print(f"Failed to delete knowledge base {cls.kb_id}: {e}")

        # Clean up S3 bucket
        try:
            objects = cls.s3_client.list_objects_v2(Bucket=cls.bucket_name)
            for obj in objects.get("Contents", []):
                cls.s3_client.delete_object(Bucket=cls.bucket_name, Key=obj["Key"])
            cls.s3_client.delete_bucket(Bucket=cls.bucket_name)
        except Exception as e:
            print(f"Failed to clean up S3 bucket {cls.bucket_name}: {e}")

        # Clean up S3 Vectors
        try:
            cls.s3vectors_client.delete_index(
                vectorBucketName=cls.vector_bucket_name,
                indexName=cls.vector_index_name,
            )
            cls.s3vectors_client.delete_vector_bucket(vectorBucketName=cls.vector_bucket_name)
        except Exception as e:
            print(f"Failed to clean up vector bucket {cls.vector_bucket_name}: {e}")

    @pytest.mark.order(1)
    def test_create_knowledge_base_and_wait(self):
        kb = self.client.create_knowledge_base_and_wait(
            name=f"{self.test_prefix}-kb",
            roleArn=self.role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": (
                        f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0"
                    ),
                },
            },
            storageConfiguration={
                "type": "S3_VECTORS",
                "s3VectorsConfiguration": {
                    "indexArn": self.index_arn,
                },
            },
        )
        self.__class__.kb_id = kb["knowledgeBaseId"]
        assert kb["status"] == "ACTIVE"
        assert kb["name"] == f"{self.test_prefix}-kb"

    @pytest.mark.order(2)
    def test_get_knowledge_base_passthrough(self):
        if not self.kb_id:
            pytest.skip("prerequisite test did not create knowledge base")
        response = self.client.get_knowledge_base(knowledgeBaseId=self.kb_id)
        assert response["knowledgeBase"]["status"] == "ACTIVE"

    @pytest.mark.order(3)
    def test_get_knowledge_base_snake_case(self):
        if not self.kb_id:
            pytest.skip("prerequisite test did not create knowledge base")
        response = self.client.get_knowledge_base(knowledge_base_id=self.kb_id)
        assert response["knowledgeBase"]["status"] == "ACTIVE"

    @pytest.mark.order(4)
    def test_list_knowledge_bases_passthrough(self):
        response = self.client.list_knowledge_bases()
        assert "knowledgeBaseSummaries" in response

    @pytest.mark.order(5)
    def test_update_knowledge_base_and_wait(self):
        if not self.kb_id:
            pytest.skip("prerequisite test did not create knowledge base")
        current = self.client.get_knowledge_base(knowledgeBaseId=self.kb_id)["knowledgeBase"]
        kb = self.client.update_knowledge_base_and_wait(
            knowledgeBaseId=self.kb_id,
            name=current["name"],
            roleArn=self.role_arn,
            knowledgeBaseConfiguration=current["knowledgeBaseConfiguration"],
            storageConfiguration=current["storageConfiguration"],
            description="updated by integ test",
        )
        assert kb["status"] == "ACTIVE"
        assert kb.get("description") == "updated by integ test"

    @pytest.mark.order(6)
    def test_create_data_source_passthrough(self):
        if not self.kb_id:
            pytest.skip("prerequisite test did not create knowledge base")
        response = self.client.create_data_source(
            knowledgeBaseId=self.kb_id,
            name=f"{self.test_prefix}-ds",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{self.bucket_name}",
                },
            },
        )
        self.__class__.ds_id = response["dataSource"]["dataSourceId"]
        assert response["dataSource"]["status"] == "AVAILABLE"

    @pytest.mark.order(7)
    def test_start_ingestion_job_and_wait(self):
        if not self.kb_id or not self.ds_id:
            pytest.skip("prerequisite tests did not create resources")
        job = self.client.start_ingestion_job_and_wait(
            knowledgeBaseId=self.kb_id,
            dataSourceId=self.ds_id,
        )
        assert job["status"] == "COMPLETE"

    @pytest.mark.order(8)
    def test_retrieve_passthrough(self):
        if not self.kb_id:
            pytest.skip("prerequisite test did not create knowledge base")
        response = self.client.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": "test query"},
        )
        assert "retrievalResults" in response

    @pytest.mark.order(9)
    def test_delete_knowledge_base_and_wait(self):
        if not self.kb_id:
            pytest.skip("prerequisite test did not create knowledge base")
        kb_id = self.kb_id
        self.__class__.kb_id = None
        self.client.delete_knowledge_base_and_wait(knowledgeBaseId=kb_id)
        with pytest.raises(ClientError):
            self.client.get_knowledge_base(knowledgeBaseId=kb_id)
