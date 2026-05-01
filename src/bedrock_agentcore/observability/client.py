"""Client for managing CloudWatch observability delivery and X-Ray Transaction Search for AgentCore resources."""

import json
import logging
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SUPPORTED_RESOURCE_TYPES = {"memory", "gateway", "runtime"}
AUTO_LOG_RESOURCE_TYPES = {"runtime"}


class ObservabilityClient:
    """Manages CloudWatch delivery configuration and X-Ray Transaction Search for AgentCore resources."""

    def __init__(
        self,
        region_name: Optional[str] = None,
        session: Optional[boto3.Session] = None,
    ):
        """Initialize the ObservabilityClient."""
        self._session = session or boto3.Session()
        self.region = region_name or self._session.region_name
        if not self.region:
            raise ValueError(
                "AWS region must be specified either via region_name parameter "
                "or configured in boto3 session/environment"
            )
        self._logs_client = self._session.client("logs", region_name=self.region)
        self._xray_client = self._session.client("xray", region_name=self.region)
        sts_client = self._session.client("sts", region_name=self.region)
        self._account_id = sts_client.get_caller_identity()["Account"]

    def enable_observability_for_resource(
        self,
        resource_arn: str,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        enable_logs: bool = True,
        enable_traces: bool = True,
        custom_log_group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enable CloudWatch logs and/or traces delivery for an AgentCore resource."""
        if resource_type is None or resource_id is None:
            try:
                resource_part = resource_arn.split(":")[-1]
                parsed_type, parsed_id = resource_part.split("/", 1)
                resource_type = resource_type or parsed_type
                resource_id = resource_id or parsed_id
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"Could not parse resource_type/resource_id from ARN: {resource_arn}. "
                    f"Please provide them explicitly. Error: {e}"
                ) from e

        if resource_type not in SUPPORTED_RESOURCE_TYPES:
            raise ValueError(
                f"Unsupported resource_type: '{resource_type}'. Must be one of: {SUPPORTED_RESOURCE_TYPES}"
            )

        results: Dict[str, Any] = {
            "resource_id": resource_id,
            "resource_type": resource_type,
            "resource_arn": resource_arn,
            "logs_enabled": False,
            "traces_enabled": False,
            "log_group": None,
            "deliveries": {},
        }

        if custom_log_group:
            log_group_name = custom_log_group
        elif resource_type == "runtime":
            log_group_name = f"/aws/bedrock-agentcore/runtimes/{resource_id}"
        else:
            log_group_name = f"/aws/vendedlogs/bedrock-agentcore/{resource_type}/APPLICATION_LOGS/{resource_id}"

        log_group_arn = f"arn:aws:logs:{self.region}:{self._account_id}:log-group:{log_group_name}"
        results["log_group"] = log_group_name

        try:
            if resource_type not in AUTO_LOG_RESOURCE_TYPES:
                self._create_log_group_if_not_exists(log_group_name)

            if enable_logs and resource_type not in AUTO_LOG_RESOURCE_TYPES:
                results["deliveries"]["logs"] = self._setup_logs_delivery(resource_arn, resource_id, log_group_arn)
                results["logs_enabled"] = True
            elif resource_type in AUTO_LOG_RESOURCE_TYPES:
                results["logs_enabled"] = True
                results["deliveries"]["logs"] = {"status": "auto-created by AWS"}

            if enable_traces:
                results["deliveries"]["traces"] = self._setup_traces_delivery(resource_arn, resource_id)
                results["traces_enabled"] = True

            results["status"] = "success"
        except Exception as e:
            logger.error("Failed to enable observability for %s/%s: %s", resource_type, resource_id, e)
            results["status"] = "error"
            results["error"] = str(e)

        return results

    def disable_observability_for_resource(
        self,
        resource_id: str,
        delete_log_group: bool = False,
    ) -> Dict[str, Any]:
        """Disable CloudWatch observability delivery for a resource."""
        results: Dict[str, Any] = {"resource_id": resource_id, "deleted": [], "errors": []}

        for suffix in ["logs", "traces"]:
            source_name = f"{resource_id}-{suffix}-source"
            dest_name = f"{resource_id}-{suffix}-destination"

            # Delete deliveries referencing this source first
            try:
                deliveries = self._logs_client.describe_deliveries()
                for delivery in deliveries.get("deliveries", []):
                    if delivery.get("deliverySourceName") == source_name:
                        try:
                            self._logs_client.delete_delivery(id=delivery["id"])
                            results["deleted"].append(f"delivery:{delivery['id']}")
                        except ClientError as e:
                            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                                results["errors"].append(f"Failed to delete delivery {delivery['id']}: {e}")
            except ClientError:
                pass

            try:
                self._logs_client.delete_delivery_source(name=source_name)
                results["deleted"].append(f"source:{source_name}")
            except ClientError as e:
                if e.response["Error"]["Code"] != "ResourceNotFoundException":
                    results["errors"].append(f"Failed to delete {source_name}: {e}")

            try:
                self._logs_client.delete_delivery_destination(name=dest_name)
                results["deleted"].append(f"destination:{dest_name}")
            except ClientError as e:
                if e.response["Error"]["Code"] != "ResourceNotFoundException":
                    results["errors"].append(f"Failed to delete {dest_name}: {e}")

        if delete_log_group:
            for resource_type in SUPPORTED_RESOURCE_TYPES:
                if resource_type == "runtime":
                    lg = f"/aws/bedrock-agentcore/runtimes/{resource_id}"
                else:
                    lg = f"/aws/vendedlogs/bedrock-agentcore/{resource_type}/APPLICATION_LOGS/{resource_id}"
                try:
                    self._logs_client.delete_log_group(logGroupName=lg)
                    results["deleted"].append(f"log_group:{lg}")
                except ClientError as e:
                    if e.response["Error"]["Code"] != "ResourceNotFoundException":
                        results["errors"].append(f"Failed to delete log group {lg}: {e}")

        results["status"] = "success" if not results["errors"] else "partial"
        return results

    def enable_transaction_search(self) -> bool:
        """Enable X-Ray Transaction Search (resource policy, trace destination, indexing rule)."""
        try:
            if self._need_resource_policy():
                self._create_resource_policy()

            if self._need_trace_destination():
                try:
                    self._xray_client.update_trace_segment_destination(Destination="CloudWatchLogs")
                except ClientError as e:
                    if e.response["Error"]["Code"] != "InvalidRequestException":
                        raise

            if self._need_indexing_rule():
                try:
                    self._xray_client.update_indexing_rule(
                        Name="Default", Rule={"Probabilistic": {"DesiredSamplingPercentage": 1}}
                    )
                except ClientError as e:
                    if e.response["Error"]["Code"] != "InvalidRequestException":
                        raise

            return True
        except Exception as e:
            logger.warning("Transaction Search configuration failed: %s", e)
            return False

    def get_observability_status(
        self,
        resource_id: str,
    ) -> Dict[str, Any]:
        """Check the observability configuration status for a resource."""
        status: Dict[str, Any] = {
            "resource_id": resource_id,
            "logs": {"configured": False},
            "traces": {"configured": False},
        }

        for suffix in ["logs", "traces"]:
            source_name = f"{resource_id}-{suffix}-source"
            try:
                self._logs_client.get_delivery_source(name=source_name)
                status[suffix]["configured"] = True
                status[suffix]["source_name"] = source_name
            except ClientError:
                pass

        return status

    # Private helpers
    # -------------------------------------------------------------------------

    def _create_log_group_if_not_exists(self, log_group_name: str) -> None:
        try:
            self._logs_client.create_log_group(logGroupName=log_group_name)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                raise

    def _setup_logs_delivery(self, resource_arn: str, resource_id: str, log_group_arn: str) -> Dict[str, str]:
        source_name = f"{resource_id}-logs-source"
        dest_name = f"{resource_id}-logs-destination"

        try:
            logs_source = self._logs_client.put_delivery_source(
                name=source_name, logType="APPLICATION_LOGS", resourceArn=resource_arn
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                logs_source = {"deliverySource": {"name": source_name}}
            else:
                raise

        try:
            logs_dest = self._logs_client.put_delivery_destination(
                name=dest_name,
                deliveryDestinationType="CWL",
                deliveryDestinationConfiguration={"destinationResourceArn": log_group_arn},
            )
            dest_arn = logs_dest["deliveryDestination"]["arn"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                dest_arn = f"arn:aws:logs:{self.region}:{self._account_id}:delivery-destination:{dest_name}"
            else:
                raise

        try:
            delivery = self._logs_client.create_delivery(
                deliverySourceName=logs_source["deliverySource"]["name"], deliveryDestinationArn=dest_arn
            )
            delivery_id = delivery.get("id", "created")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConflictException":
                delivery_id = "existing"
            else:
                raise

        return {"delivery_id": delivery_id, "source_name": source_name, "destination_name": dest_name}

    def _setup_traces_delivery(self, resource_arn: str, resource_id: str) -> Dict[str, str]:
        source_name = f"{resource_id}-traces-source"
        dest_name = f"{resource_id}-traces-destination"

        try:
            traces_source = self._logs_client.put_delivery_source(
                name=source_name, logType="TRACES", resourceArn=resource_arn
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                traces_source = {"deliverySource": {"name": source_name}}
            else:
                raise

        try:
            traces_dest = self._logs_client.put_delivery_destination(name=dest_name, deliveryDestinationType="XRAY")
            dest_arn = traces_dest["deliveryDestination"]["arn"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                dest_arn = f"arn:aws:logs:{self.region}:{self._account_id}:delivery-destination:{dest_name}"
            else:
                raise

        try:
            delivery = self._logs_client.create_delivery(
                deliverySourceName=traces_source["deliverySource"]["name"], deliveryDestinationArn=dest_arn
            )
            delivery_id = delivery.get("id", "created")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConflictException":
                delivery_id = "existing"
            else:
                raise

        return {"delivery_id": delivery_id, "source_name": source_name, "destination_name": dest_name}

    def _need_resource_policy(self, policy_name: str = "TransactionSearchXRayAccess") -> bool:
        try:
            response = self._logs_client.describe_resource_policies()
            return not any(p.get("policyName") == policy_name for p in response.get("resourcePolicies", []))
        except Exception:
            return True

    def _create_resource_policy(self) -> None:
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "TransactionSearchXRayAccess",
                    "Effect": "Allow",
                    "Principal": {"Service": "xray.amazonaws.com"},
                    "Action": "logs:PutLogEvents",
                    "Resource": [
                        f"arn:aws:logs:{self.region}:{self._account_id}:log-group:aws/spans:*",
                        f"arn:aws:logs:{self.region}:{self._account_id}:log-group:/aws/application-signals/data:*",
                    ],
                    "Condition": {
                        "ArnLike": {"aws:SourceArn": f"arn:aws:xray:{self.region}:{self._account_id}:*"},
                        "StringEquals": {"aws:SourceAccount": self._account_id},
                    },
                }
            ],
        }
        try:
            self._logs_client.put_resource_policy(
                policyName="TransactionSearchXRayAccess", policyDocument=json.dumps(policy_document)
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "InvalidParameterException":
                raise

    def _need_trace_destination(self) -> bool:
        try:
            response = self._xray_client.get_trace_segment_destination()
            return response.get("Destination") != "CloudWatchLogs"
        except Exception:
            return True  # If check fails, assume we need it

    def _need_indexing_rule(self) -> bool:
        try:
            response = self._xray_client.get_indexing_rules()
            return not any(r.get("Name") == "Default" for r in response.get("IndexingRules", []))
        except Exception:
            return True
