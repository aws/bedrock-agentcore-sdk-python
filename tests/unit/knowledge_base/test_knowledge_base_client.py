"""Tests for KnowledgeBaseClient."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from bedrock_agentcore.knowledge_base.client import KnowledgeBaseClient


class TestKnowledgeBaseClientInit:
    """Tests for KnowledgeBaseClient initialization."""

    def test_init_with_region(self):
        mock_session = MagicMock()
        mock_session.region_name = "eu-west-1"
        client = KnowledgeBaseClient(region_name="us-west-2", boto3_session=mock_session)
        assert client.region_name == "us-west-2"

    def test_init_default_region_from_session(self):
        mock_session = MagicMock()
        mock_session.region_name = "eu-west-1"
        client = KnowledgeBaseClient(boto3_session=mock_session)
        assert client.region_name == "eu-west-1"

    def test_init_default_region_fallback(self):
        mock_session = MagicMock()
        mock_session.region_name = None
        client = KnowledgeBaseClient(boto3_session=mock_session)
        assert client.region_name == "us-west-2"

    def test_init_creates_both_clients(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        KnowledgeBaseClient(boto3_session=mock_session)

        assert mock_session.client.call_count == 2
        call_args_list = [call[0][0] for call in mock_session.client.call_args_list]
        assert "bedrock-agent" in call_args_list
        assert "bedrock-agent-runtime" in call_args_list

    def test_init_with_integration_source(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = KnowledgeBaseClient(boto3_session=mock_session, integration_source="strands")
        assert client.integration_source == "strands"


class TestKnowledgeBaseClientPassthrough:
    """Tests for __getattr__ passthrough."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = KnowledgeBaseClient(boto3_session=mock_session)
        client.cp_client = Mock()
        client.dp_client = Mock()
        return client

    def test_cp_method_forwarded(self):
        client = self._make_client()
        client.cp_client.create_knowledge_base.return_value = {"knowledgeBase": {"knowledgeBaseId": "kb-123"}}

        result = client.create_knowledge_base(name="test-kb")

        client.cp_client.create_knowledge_base.assert_called_once_with(name="test-kb")
        assert result["knowledgeBase"]["knowledgeBaseId"] == "kb-123"

    def test_dp_method_forwarded(self):
        client = self._make_client()
        client.dp_client.retrieve.return_value = {"retrievalResults": [{"content": "hello"}]}

        result = client.retrieve(knowledgeBaseId="kb-123", retrievalQuery={"text": "test"})

        client.dp_client.retrieve.assert_called_once_with(knowledgeBaseId="kb-123", retrievalQuery={"text": "test"})
        assert result["retrievalResults"][0]["content"] == "hello"

    def test_snake_case_kwargs_converted(self):
        client = self._make_client()
        client.cp_client.get_knowledge_base.return_value = {"knowledgeBase": {"knowledgeBaseId": "kb-123"}}

        client.get_knowledge_base(knowledge_base_id="kb-123")

        client.cp_client.get_knowledge_base.assert_called_once_with(knowledgeBaseId="kb-123")

    def test_non_allowlisted_method_raises_attribute_error(self):
        client = self._make_client()

        with pytest.raises(AttributeError, match="has no attribute 'not_a_real_method'"):
            client.not_a_real_method()

    def test_allowed_cp_methods_set(self):
        expected = {
            "create_knowledge_base",
            "get_knowledge_base",
            "update_knowledge_base",
            "delete_knowledge_base",
            "list_knowledge_bases",
            "create_data_source",
            "get_data_source",
            "update_data_source",
            "delete_data_source",
            "list_data_sources",
            "start_ingestion_job",
            "get_ingestion_job",
            "stop_ingestion_job",
            "list_ingestion_jobs",
            "ingest_knowledge_base_documents",
            "get_knowledge_base_documents",
            "delete_knowledge_base_documents",
            "list_knowledge_base_documents",
            "tag_resource",
            "untag_resource",
            "list_tags_for_resource",
        }
        assert expected == KnowledgeBaseClient._ALLOWED_CP_METHODS

    def test_allowed_dp_methods_set(self):
        expected = {
            "retrieve",
            "retrieve_and_generate",
            "retrieve_and_generate_stream",
            "generate_query",
            "rerank",
            "agentic_retrieve_stream",
        }
        assert expected == KnowledgeBaseClient._ALLOWED_DP_METHODS


class TestCreateKnowledgeBaseAndWait:
    """Tests for create_knowledge_base_and_wait."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = KnowledgeBaseClient(boto3_session=mock_session)
        client.cp_client = Mock()
        client.dp_client = Mock()
        return client

    def test_immediate_active(self):
        client = self._make_client()
        client.cp_client.create_knowledge_base.return_value = {"knowledgeBase": {"knowledgeBaseId": "kb-123"}}
        client.cp_client.get_knowledge_base.return_value = {
            "knowledgeBase": {"status": "ACTIVE", "knowledgeBaseId": "kb-123"}
        }

        result = client.create_knowledge_base_and_wait(name="test")

        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 1, 1])
    def test_polls_through_creating(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.create_knowledge_base.return_value = {"knowledgeBase": {"knowledgeBaseId": "kb-123"}}
        client.cp_client.get_knowledge_base.side_effect = [
            {"knowledgeBase": {"status": "CREATING"}},
            {"knowledgeBase": {"status": "ACTIVE", "knowledgeBaseId": "kb-123"}},
        ]

        result = client.create_knowledge_base_and_wait(name="test")

        assert result["status"] == "ACTIVE"
        assert client.cp_client.get_knowledge_base.call_count == 2

    def test_raises_on_failed(self):
        client = self._make_client()
        client.cp_client.create_knowledge_base.return_value = {"knowledgeBase": {"knowledgeBaseId": "kb-123"}}
        client.cp_client.get_knowledge_base.return_value = {
            "knowledgeBase": {"status": "FAILED", "statusReasons": ["something broke"]}
        }

        with pytest.raises(RuntimeError, match="FAILED"):
            client.create_knowledge_base_and_wait(name="test")

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 0, 301])
    def test_timeout(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.create_knowledge_base.return_value = {"knowledgeBase": {"knowledgeBaseId": "kb-123"}}
        client.cp_client.get_knowledge_base.return_value = {"knowledgeBase": {"status": "CREATING"}}

        with pytest.raises(TimeoutError):
            client.create_knowledge_base_and_wait(name="test")


class TestUpdateKnowledgeBaseAndWait:
    """Tests for update_knowledge_base_and_wait."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = KnowledgeBaseClient(boto3_session=mock_session)
        client.cp_client = Mock()
        client.dp_client = Mock()
        return client

    def test_immediate_active(self):
        client = self._make_client()
        client.cp_client.update_knowledge_base.return_value = {"knowledgeBase": {"knowledgeBaseId": "kb-123"}}
        client.cp_client.get_knowledge_base.return_value = {
            "knowledgeBase": {"status": "ACTIVE", "knowledgeBaseId": "kb-123"}
        }

        result = client.update_knowledge_base_and_wait(knowledge_base_id="kb-123")

        assert result["status"] == "ACTIVE"

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 1, 1])
    def test_polls_through_updating(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.update_knowledge_base.return_value = {"knowledgeBase": {"knowledgeBaseId": "kb-123"}}
        client.cp_client.get_knowledge_base.side_effect = [
            {"knowledgeBase": {"status": "UPDATING"}},
            {"knowledgeBase": {"status": "ACTIVE", "knowledgeBaseId": "kb-123"}},
        ]

        result = client.update_knowledge_base_and_wait(knowledge_base_id="kb-123")

        assert result["status"] == "ACTIVE"
        assert client.cp_client.get_knowledge_base.call_count == 2

    def test_raises_on_failed(self):
        client = self._make_client()
        client.cp_client.update_knowledge_base.return_value = {"knowledgeBase": {"knowledgeBaseId": "kb-123"}}
        client.cp_client.get_knowledge_base.return_value = {
            "knowledgeBase": {"status": "FAILED", "statusReasons": ["bad config"]}
        }

        with pytest.raises(RuntimeError, match="FAILED"):
            client.update_knowledge_base_and_wait(knowledge_base_id="kb-123")


class TestDeleteKnowledgeBaseAndWait:
    """Tests for delete_knowledge_base_and_wait."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = KnowledgeBaseClient(boto3_session=mock_session)
        client.cp_client = Mock()
        client.dp_client = Mock()
        return client

    def test_immediate_not_found(self):
        client = self._make_client()
        client.cp_client.delete_knowledge_base.return_value = {"knowledgeBaseId": "kb-123"}
        client.cp_client.get_knowledge_base.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}},
            "GetKnowledgeBase",
        )

        client.delete_knowledge_base_and_wait(knowledge_base_id="kb-123")

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 1, 1])
    def test_polls_through_deleting(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.delete_knowledge_base.return_value = {"knowledgeBaseId": "kb-123"}
        client.cp_client.get_knowledge_base.side_effect = [
            {"knowledgeBase": {"status": "DELETING"}},
            ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}},
                "GetKnowledgeBase",
            ),
        ]

        client.delete_knowledge_base_and_wait(knowledge_base_id="kb-123")

    def test_raises_on_delete_unsuccessful(self):
        client = self._make_client()
        client.cp_client.delete_knowledge_base.return_value = {"knowledgeBaseId": "kb-123"}
        client.cp_client.get_knowledge_base.return_value = {
            "knowledgeBase": {"status": "DELETE_UNSUCCESSFUL", "failureReasons": ["dependencies exist"]}
        }

        with pytest.raises(RuntimeError, match="DELETE_UNSUCCESSFUL"):
            client.delete_knowledge_base_and_wait(knowledge_base_id="kb-123")


class TestStartIngestionJobAndWait:
    """Tests for start_ingestion_job_and_wait."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = KnowledgeBaseClient(boto3_session=mock_session)
        client.cp_client = Mock()
        client.dp_client = Mock()
        return client

    def test_immediate_complete(self):
        client = self._make_client()
        client.cp_client.start_ingestion_job.return_value = {
            "ingestionJob": {
                "knowledgeBaseId": "kb-123",
                "dataSourceId": "ds-456",
                "ingestionJobId": "job-789",
            }
        }
        client.cp_client.get_ingestion_job.return_value = {
            "ingestionJob": {"status": "COMPLETE", "ingestionJobId": "job-789"}
        }

        result = client.start_ingestion_job_and_wait(knowledge_base_id="kb-123", data_source_id="ds-456")

        assert result["status"] == "COMPLETE"

    @patch("bedrock_agentcore._utils.polling.time.sleep")
    @patch("bedrock_agentcore._utils.polling.time.time", side_effect=[0, 0, 1, 1, 2, 2])
    def test_polls_through_in_progress(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.start_ingestion_job.return_value = {
            "ingestionJob": {
                "knowledgeBaseId": "kb-123",
                "dataSourceId": "ds-456",
                "ingestionJobId": "job-789",
            }
        }
        client.cp_client.get_ingestion_job.side_effect = [
            {"ingestionJob": {"status": "STARTING"}},
            {"ingestionJob": {"status": "IN_PROGRESS"}},
            {"ingestionJob": {"status": "COMPLETE", "ingestionJobId": "job-789"}},
        ]

        result = client.start_ingestion_job_and_wait(knowledge_base_id="kb-123", data_source_id="ds-456")

        assert result["status"] == "COMPLETE"
        assert client.cp_client.get_ingestion_job.call_count == 3

    def test_raises_on_failed(self):
        client = self._make_client()
        client.cp_client.start_ingestion_job.return_value = {
            "ingestionJob": {
                "knowledgeBaseId": "kb-123",
                "dataSourceId": "ds-456",
                "ingestionJobId": "job-789",
            }
        }
        client.cp_client.get_ingestion_job.return_value = {
            "ingestionJob": {"status": "FAILED", "failureReasons": ["data source error"]}
        }

        with pytest.raises(RuntimeError, match="FAILED"):
            client.start_ingestion_job_and_wait(knowledge_base_id="kb-123", data_source_id="ds-456")

    def test_raises_on_stopped(self):
        client = self._make_client()
        client.cp_client.start_ingestion_job.return_value = {
            "ingestionJob": {
                "knowledgeBaseId": "kb-123",
                "dataSourceId": "ds-456",
                "ingestionJobId": "job-789",
            }
        }
        client.cp_client.get_ingestion_job.return_value = {
            "ingestionJob": {"status": "STOPPED", "failureReasons": ["user stopped"]}
        }

        with pytest.raises(RuntimeError, match="STOPPED"):
            client.start_ingestion_job_and_wait(knowledge_base_id="kb-123", data_source_id="ds-456")


class TestIngestDocumentsAndWait:
    """Tests for ingest_knowledge_base_documents_and_wait."""

    def _make_client(self):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        client = KnowledgeBaseClient(boto3_session=mock_session)
        client.cp_client = Mock()
        client.dp_client = Mock()
        return client

    def _base_kwargs(self):
        return {
            "knowledge_base_id": "kb-123",
            "data_source_id": "ds-456",
            "documents": [
                {
                    "identifier": {"dataSourceType": "CUSTOM", "custom": {"id": "doc-1"}},
                    "content": {
                        "custom": {
                            "customDocumentIdentifier": {"id": "doc-1"},
                            "sourceType": "IN_LINE",
                            "inlineContent": {"type": "TEXT", "textContent": {"data": "hello"}},
                        }
                    },
                },
                {
                    "identifier": {"dataSourceType": "CUSTOM", "custom": {"id": "doc-2"}},
                    "content": {
                        "custom": {
                            "customDocumentIdentifier": {"id": "doc-2"},
                            "sourceType": "IN_LINE",
                            "inlineContent": {"type": "TEXT", "textContent": {"data": "world"}},
                        }
                    },
                },
            ],
        }

    def test_all_indexed_immediately(self):
        client = self._make_client()
        client.cp_client.ingest_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "STARTING"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "STARTING"},
            ]
        }
        client.cp_client.get_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "INDEXED"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "INDEXED"},
            ]
        }

        result = client.ingest_knowledge_base_documents_and_wait(**self._base_kwargs())

        assert len(result["documentDetails"]) == 2
        assert all(d["status"] == "INDEXED" for d in result["documentDetails"])

    @patch("bedrock_agentcore.knowledge_base.client.time.sleep")
    @patch("bedrock_agentcore.knowledge_base.client.time.time", side_effect=[0, 1, 2])
    def test_polls_through_in_progress(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.ingest_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "STARTING"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "STARTING"},
            ]
        }
        client.cp_client.get_knowledge_base_documents.side_effect = [
            {
                "documentDetails": [
                    {"identifier": {"custom": {"id": "doc-1"}}, "status": "IN_PROGRESS"},
                    {"identifier": {"custom": {"id": "doc-2"}}, "status": "IN_PROGRESS"},
                ]
            },
            {
                "documentDetails": [
                    {"identifier": {"custom": {"id": "doc-1"}}, "status": "INDEXED"},
                    {"identifier": {"custom": {"id": "doc-2"}}, "status": "INDEXED"},
                ]
            },
        ]

        result = client.ingest_knowledge_base_documents_and_wait(**self._base_kwargs())

        assert all(d["status"] == "INDEXED" for d in result["documentDetails"])
        assert client.cp_client.get_knowledge_base_documents.call_count == 2

    def test_mixed_success_statuses(self):
        client = self._make_client()
        client.cp_client.ingest_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "STARTING"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "STARTING"},
            ]
        }
        client.cp_client.get_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "INDEXED"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "PARTIALLY_INDEXED"},
            ]
        }

        result = client.ingest_knowledge_base_documents_and_wait(**self._base_kwargs())

        assert result["documentDetails"][0]["status"] == "INDEXED"
        assert result["documentDetails"][1]["status"] == "PARTIALLY_INDEXED"

    def test_ignored_docs_merged_into_response(self):
        """IGNORED docs from initial response are merged back into the final poll response."""
        client = self._make_client()
        client.cp_client.ingest_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "STARTING"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "IGNORED", "statusReason": "unchanged"},
            ]
        }
        client.cp_client.get_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "INDEXED"},
            ]
        }

        result = client.ingest_knowledge_base_documents_and_wait(**self._base_kwargs())

        assert len(result["documentDetails"]) == 2
        statuses = {d["status"] for d in result["documentDetails"]}
        assert "INDEXED" in statuses
        assert "IGNORED" in statuses

    def test_all_ignored_returns_immediately(self):
        """If all docs are IGNORED (none accepted), return the initial response without polling."""
        client = self._make_client()
        client.cp_client.ingest_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "IGNORED", "statusReason": "unchanged"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "IGNORED", "statusReason": "unchanged"},
            ]
        }

        result = client.ingest_knowledge_base_documents_and_wait(**self._base_kwargs())

        assert len(result["documentDetails"]) == 2
        assert all(d["status"] == "IGNORED" for d in result["documentDetails"])
        # Should not have called get_knowledge_base_documents since no docs were accepted
        client.cp_client.get_knowledge_base_documents.assert_not_called()

    def test_failed_docs_returned_without_raising(self):
        """Failed docs are returned as-is without raising RuntimeError."""
        client = self._make_client()
        client.cp_client.ingest_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "STARTING"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "STARTING"},
            ]
        }
        client.cp_client.get_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "INDEXED"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "FAILED", "statusReason": "parse error"},
            ]
        }

        result = client.ingest_knowledge_base_documents_and_wait(**self._base_kwargs())

        assert len(result["documentDetails"]) == 2
        statuses = [d["status"] for d in result["documentDetails"]]
        assert "INDEXED" in statuses
        assert "FAILED" in statuses

    def test_metadata_update_failed_returned_without_raising(self):
        """METADATA_UPDATE_FAILED docs are returned without raising RuntimeError."""
        client = self._make_client()
        client.cp_client.ingest_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "STARTING"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "STARTING"},
            ]
        }
        client.cp_client.get_knowledge_base_documents.return_value = {
            "documentDetails": [
                {
                    "identifier": {"custom": {"id": "doc-1"}},
                    "status": "METADATA_UPDATE_FAILED",
                    "statusReason": "schema mismatch",
                },
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "INDEXED"},
            ]
        }

        result = client.ingest_knowledge_base_documents_and_wait(**self._base_kwargs())

        assert len(result["documentDetails"]) == 2
        statuses = [d["status"] for d in result["documentDetails"]]
        assert "METADATA_UPDATE_FAILED" in statuses
        assert "INDEXED" in statuses

    @patch("bedrock_agentcore.knowledge_base.client.time.sleep")
    @patch("bedrock_agentcore.knowledge_base.client.time.time", side_effect=[0, 0, 301])
    def test_timeout(self, _mock_time, _mock_sleep):
        client = self._make_client()
        client.cp_client.ingest_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "STARTING"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "STARTING"},
            ]
        }
        client.cp_client.get_knowledge_base_documents.return_value = {
            "documentDetails": [
                {"identifier": {"custom": {"id": "doc-1"}}, "status": "IN_PROGRESS"},
                {"identifier": {"custom": {"id": "doc-2"}}, "status": "IN_PROGRESS"},
            ]
        }

        with pytest.raises(TimeoutError):
            client.ingest_knowledge_base_documents_and_wait(**self._base_kwargs())
