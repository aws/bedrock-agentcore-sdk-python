"""AgentCore Knowledge Base SDK - Client for Knowledge Base control and data plane operations."""

import logging
import time
from typing import Any, Dict, Optional, Set

import boto3
from botocore.config import Config

from .._utils.config import WaitConfig
from .._utils.polling import wait_until, wait_until_deleted
from .._utils.snake_case import accept_snake_case_kwargs, convert_kwargs
from .._utils.user_agent import build_user_agent_suffix

logger = logging.getLogger(__name__)

_KB_FAILED_STATUSES: Set[str] = {"FAILED"}
_KB_DELETE_FAILED_STATUSES: Set[str] = {"DELETE_UNSUCCESSFUL", "FAILED"}
_INGESTION_FAILED_STATUSES: Set[str] = {"FAILED", "STOPPED"}
_DOC_SUCCESS_STATUSES: Set[str] = {"INDEXED", "PARTIALLY_INDEXED", "METADATA_PARTIALLY_INDEXED"}
_DOC_FAILED_STATUSES: Set[str] = {"FAILED", "METADATA_UPDATE_FAILED", "NOT_FOUND", "IGNORED"}
_DOC_TERMINAL_STATUSES: Set[str] = _DOC_SUCCESS_STATUSES | _DOC_FAILED_STATUSES


class KnowledgeBaseClient:
    """Client for Amazon Bedrock Knowledge Base operations.

    Provides unified access to Knowledge Base control plane (bedrock-agent) and
    data plane (bedrock-agent-runtime) APIs. Allowlisted boto3 methods can be
    called directly on this client. Parameters accept both camelCase and
    snake_case (auto-converted).

    Example::

        from bedrock_agentcore.knowledge_base import KnowledgeBaseClient

        client = KnowledgeBaseClient(region_name="us-east-1")

        # Pass-through to boto3 control plane
        kb = client.create_knowledge_base(
            name="my-kb",
            roleArn="arn:aws:iam::123456789012:role/KBRole",
            knowledgeBaseConfiguration={...},
            storageConfiguration={...},
        )

        # Pass-through to boto3 data plane
        results = client.retrieve(
            knowledgeBaseId="KB123",
            retrievalQuery={"text": "What is..."},
        )
    """

    _ALLOWED_CP_METHODS = {
        # KnowledgeBase CRUD
        "create_knowledge_base",
        "get_knowledge_base",
        "update_knowledge_base",
        "delete_knowledge_base",
        "list_knowledge_bases",
        # DataSource CRUD
        "create_data_source",
        "get_data_source",
        "update_data_source",
        "delete_data_source",
        "list_data_sources",
        # Ingestion Jobs
        "start_ingestion_job",
        "get_ingestion_job",
        "stop_ingestion_job",
        "list_ingestion_jobs",
        # Document Management
        "ingest_knowledge_base_documents",
        "get_knowledge_base_documents",
        "delete_knowledge_base_documents",
        "list_knowledge_base_documents",
        # Tagging
        "tag_resource",
        "untag_resource",
        "list_tags_for_resource",
    }

    _ALLOWED_DP_METHODS = {
        "retrieve",
        "retrieve_and_generate",
        "retrieve_and_generate_stream",
        "generate_query",
        "rerank",
        "agentic_retrieve_stream",
    }

    def __init__(
        self,
        region_name: Optional[str] = None,
        integration_source: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
    ):
        """Initialize the Knowledge Base client.

        Args:
            region_name: AWS region name. If not provided, uses the session's region or "us-west-2".
            integration_source: Optional integration source for user-agent telemetry.
            boto3_session: Optional boto3 Session to use. If not provided, a default session
                          is created. Useful for named profiles or custom credentials.
        """
        session = boto3_session if boto3_session else boto3.Session()
        self.region_name = region_name or session.region_name or "us-west-2"
        self.integration_source = integration_source

        user_agent_extra = build_user_agent_suffix(integration_source)
        client_config = Config(user_agent_extra=user_agent_extra)

        self.cp_client = session.client("bedrock-agent", region_name=self.region_name, config=client_config)
        self.dp_client = session.client("bedrock-agent-runtime", region_name=self.region_name, config=client_config)

        logger.info("Initialized KnowledgeBaseClient for region: %s", self.region_name)

    # Pass-through
    # -------------------------------------------------------------------------
    def __getattr__(self, name: str):
        """Dynamically forward allowlisted method calls to the appropriate boto3 client."""
        if name in self._ALLOWED_CP_METHODS and hasattr(self.cp_client, name):
            method = getattr(self.cp_client, name)
            logger.debug("Forwarding method '%s' to cp_client", name)
            return accept_snake_case_kwargs(method)

        if name in self._ALLOWED_DP_METHODS and hasattr(self.dp_client, name):
            method = getattr(self.dp_client, name)
            logger.debug("Forwarding method '%s' to dp_client", name)
            return accept_snake_case_kwargs(method)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Method not found on cp_client or dp_client. "
            f"Available methods can be found in the boto3 documentation for "
            f"'bedrock-agent' and 'bedrock-agent-runtime' services."
        )

    # *_and_wait methods
    # -------------------------------------------------------------------------
    def create_knowledge_base_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Create a knowledge base and wait for it to reach ACTIVE status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the create_knowledge_base API.

        Returns:
            Knowledge base details when ACTIVE.

        Raises:
            RuntimeError: If the knowledge base reaches FAILED status.
            TimeoutError: If the knowledge base doesn't become ACTIVE within max_wait.
        """
        response = self.cp_client.create_knowledge_base(**convert_kwargs(kwargs))
        kb_id = response["knowledgeBase"]["knowledgeBaseId"]
        return wait_until(
            lambda: self.cp_client.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"],
            "ACTIVE",
            _KB_FAILED_STATUSES,
            wait_config,
        )

    def update_knowledge_base_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Update a knowledge base and wait for it to reach ACTIVE status.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the update_knowledge_base API.

        Returns:
            Knowledge base details when ACTIVE.

        Raises:
            RuntimeError: If the knowledge base reaches FAILED status.
            TimeoutError: If the knowledge base doesn't become ACTIVE within max_wait.
        """
        response = self.cp_client.update_knowledge_base(**convert_kwargs(kwargs))
        kb_id = response["knowledgeBase"]["knowledgeBaseId"]
        return wait_until(
            lambda: self.cp_client.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"],
            "ACTIVE",
            _KB_FAILED_STATUSES,
            wait_config,
        )

    def delete_knowledge_base_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> None:
        """Delete a knowledge base and wait for deletion to complete.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the delete_knowledge_base API.

        Raises:
            RuntimeError: If the knowledge base reaches DELETE_UNSUCCESSFUL status.
            TimeoutError: If the knowledge base isn't deleted within max_wait.
        """
        response = self.cp_client.delete_knowledge_base(**convert_kwargs(kwargs))
        kb_id = response["knowledgeBaseId"]
        wait_until_deleted(
            lambda: self.cp_client.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"],
            failed=_KB_DELETE_FAILED_STATUSES,
            wait_config=wait_config,
            error_field="failureReasons",
        )

    def start_ingestion_job_and_wait(self, wait_config: Optional[WaitConfig] = None, **kwargs) -> Dict[str, Any]:
        """Start an ingestion job and wait for it to complete.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the start_ingestion_job API.

        Returns:
            Ingestion job details when COMPLETE.

        Raises:
            RuntimeError: If the job reaches FAILED or STOPPED status.
            TimeoutError: If the job doesn't complete within max_wait.
        """
        response = self.cp_client.start_ingestion_job(**convert_kwargs(kwargs))
        job = response["ingestionJob"]
        kb_id = job["knowledgeBaseId"]
        ds_id = job["dataSourceId"]
        job_id = job["ingestionJobId"]
        return wait_until(
            lambda: self.cp_client.get_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=ds_id,
                ingestionJobId=job_id,
            )["ingestionJob"],
            "COMPLETE",
            _INGESTION_FAILED_STATUSES,
            wait_config,
            error_field="failureReasons",
        )

    def ingest_knowledge_base_documents_and_wait(
        self, wait_config: Optional[WaitConfig] = None, **kwargs
    ) -> Dict[str, Any]:
        """Ingest documents and wait for all to reach a terminal state.

        Args:
            wait_config: Optional WaitConfig for polling behavior (default: max_wait=300, poll_interval=10).
            **kwargs: Arguments forwarded to the ingest_knowledge_base_documents API.
                Must include knowledgeBaseId, dataSourceId, and documents.

        Returns:
            Final get_knowledge_base_documents response with terminal status for all docs.

        Raises:
            TimeoutError: If not all documents reach terminal state within max_wait.
        """
        converted = convert_kwargs(kwargs)
        response = self.cp_client.ingest_knowledge_base_documents(**converted)

        initial_details = response.get("documentDetails", [])
        ignored_docs = [d for d in initial_details if d.get("status") == "IGNORED"]
        accepted_docs = [d for d in initial_details if d.get("status") == "STARTING"]

        if not accepted_docs:
            return response

        kb_id = converted["knowledgeBaseId"]
        ds_id = converted["dataSourceId"]
        doc_identifiers = [doc["identifier"] for doc in accepted_docs]

        wait = wait_config or WaitConfig()
        start_time = time.time()

        while True:
            poll_response = self.cp_client.get_knowledge_base_documents(
                knowledgeBaseId=kb_id,
                dataSourceId=ds_id,
                documentIdentifiers=doc_identifiers,
            )

            doc_details = poll_response.get("documentDetails", [])
            all_terminal = all(d.get("status") in _DOC_TERMINAL_STATUSES for d in doc_details)

            if all_terminal:
                poll_response["documentDetails"] = doc_details + ignored_docs
                return poll_response

            if time.time() - start_time >= wait.max_wait:
                raise TimeoutError("Not all documents reached terminal state within %d seconds" % wait.max_wait)

            time.sleep(wait.poll_interval)
