# Payment Integration Tests

This directory contains integration tests for the Bedrock AgentCore Payment Control Plane and Data Plane Clients.

## Prerequisites

- Python 3.10+
- AWS credentials configured
- Access to Bedrock AgentCore Payment service
- pytest and dependencies installed

## Environment Setup

### Required Environment Variables

```bash
# Bedrock AgentCore Control Plane endpoint
export BEDROCK_AGENTCORE_CONTROL_ENDPOINT="https://bedrock-agentcore-control.us-west-2.amazonaws.com"
```

### Optional Environment Variables

```bash
# AWS region (default: us-west-2)
export BEDROCK_TEST_REGION="us-west-2"

# IAM role ARN for payment manager operations
export TEST_PAYMENT_ROLE_ARN="arn:aws:iam::123456789012:role/PaymentRole"

# Credential provider ARN for payment connector operations
export TEST_CREDENTIAL_PROVIDER_ARN="arn:aws:iam::123456789012:role/CredentialProvider"

# Data Plane specific environment variables
export TEST_PAYMENT_MANAGER_ARN="arn:aws:bedrock:us-west-2:123456789012:payment-manager/pm-123"
export TEST_PAYMENT_CONNECTOR_ID="pc-123"
export TEST_USER_ID="test-user"
```

### AWS Credentials

Configure AWS credentials using one of these methods:

1. **AWS CLI Configuration** (recommended):
   ```bash
   aws configure
   ```

2. **Environment Variables**:
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_DEFAULT_REGION="us-west-2"
   ```

3. **IAM Role** (if running on EC2/ECS):
   - Ensure the instance has an IAM role with appropriate permissions

## Running Tests

### Run All Integration Tests

```bash
pytest tests_integ/payment/ -v
```

### Run Control Plane Tests Only

```bash
pytest tests_integ/payment/test_payment_controlplane.py -v
```

### Run Data Plane Tests Only

```bash
pytest tests_integ/payment/test_payment_client.py -v
```

### Run Specific Test Class

```bash
# Payment manager tests
pytest tests_integ/payment/test_payment_controlplane.py::TestPaymentControlPlaneClient -v

# Payment data plane tests
pytest tests_integ/payment/test_payment_client.py::TestPaymentClientDataPlane -v
```

### Run Specific Test

```bash
pytest tests_integ/payment/test_payment_controlplane.py::TestPaymentControlPlaneClient::test_create_and_get_payment_manager -v
pytest tests_integ/payment/test_payment_client.py::TestPaymentClientDataPlane::test_create_and_get_payment_instrument -v
```

### Run with Verbose Output

```bash
pytest tests_integ/payment/ -vv -s
```

### Run with Coverage

```bash
pytest tests_integ/payment/ --cov=bedrock_agentcore.payment --cov-report=html
```

## Test Structure

### TestPaymentControlPlaneClient

Tests for payment control plane CRUD operations:
- `test_create_and_get_payment_manager`: Create and retrieve a payment manager
- `test_list_payment_managers`: List all payment managers
- `test_update_payment_manager`: Update payment manager properties
- `test_create_and_get_payment_connector`: Create and retrieve a payment connector
- `test_list_payment_connectors`: List connectors for a manager
- `test_update_payment_connector`: Update connector properties
- `test_complete_payment_setup_workflow`: Complete setup workflow with manager and connectors

### TestPaymentClientDataPlane

Tests for payment data plane operations:
- `test_create_and_get_payment_instrument`: Create and retrieve a payment instrument
- `test_list_payment_instruments`: List payment instruments for a user
- `test_create_and_get_payment_session`: Create and retrieve a payment session
- `test_list_payment_sessions`: List payment sessions for a user
- `test_delete_payment_session`: Delete a payment session
- `test_delete_payment_instrument`: Delete a payment instrument
- `test_process_payment`: Process a payment transaction
- `test_complete_payment_workflow`: End-to-end workflow (instrument → session → payment)
- `test_idempotency_with_client_token`: Verify idempotency using client tokens

## Service Side Verification

Monitor service logs to verify test execution:

### Expected Log Patterns - Control Plane

```
Creating payment manager: test-manager-integration
Payment manager created: arn:aws:payments:us-west-2:123456789012:manager/pm-123
Creating payment connector: test-connector for manager pm-123
Payment connector created: pc-123
Updating payment manager: pm-123
Deleting payment connector: pc-123 for manager pm-123
Deleting payment manager: pm-123
```

### Expected Log Patterns - Data Plane

```
Creating payment instrument for user: test-user
Successfully created instrument for user: test-user
Retrieving payment instrument for user: test-user
Creating payment session for user: test-user
Processing payment of type CRYPTO_X402 for user: test-user
Successfully processed payment for user: test-user
```

### Log Levels

- **INFO**: Normal operations (create, update, delete, list)
- **WARNING**: Retry attempts, timeouts
- **ERROR**: Failed operations, validation errors

## Troubleshooting

### Tests Are Skipped

**Issue**: All tests are skipped with message "BEDROCK_AGENTCORE_CONTROL_ENDPOINT not set"

**Solution**: Set the required environment variable:
```bash
export BEDROCK_AGENTCORE_CONTROL_ENDPOINT="https://bedrock-agentcore-control.us-west-2.amazonaws.com"
```

### Connection Timeout

**Issue**: Tests fail with connection timeout

**Solution**:
1. Verify the endpoint URL is correct
2. Check network connectivity to the endpoint
3. Verify AWS credentials are valid
4. Check security group/firewall rules

### Authentication Errors

**Issue**: Tests fail with "AccessDeniedException" or "UnauthorizedException"

**Solution**:
1. Verify AWS credentials are configured correctly
2. Check IAM permissions for the credentials
3. Verify the role ARNs are correct and accessible

### Resource Cleanup Issues

**Issue**: Tests fail because resources weren't cleaned up from previous runs

**Solution**:
1. Manually delete orphaned resources using AWS CLI
2. Check service logs for cleanup errors
3. Increase timeout values if cleanup is slow

## Performance Considerations

- Tests use `wait_for_active=True` by default, which polls for status changes
- Default timeout is 60 seconds per operation
- Adjust `max_wait` and `poll_interval` parameters if needed
- Run tests sequentially to avoid resource conflicts

## Best Practices

1. **Isolation**: Each test creates and cleans up its own resources
2. **Error Handling**: Tests include proper cleanup in finally blocks
3. **Logging**: Enable debug logging for troubleshooting:
   ```bash
   pytest tests_integ/payment/test_payment_controlplane.py -v --log-cli-level=DEBUG
   ```
4. **Monitoring**: Watch service logs during test execution
5. **Cleanup**: Ensure all resources are deleted after tests complete

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        env:
          BEDROCK_AGENTCORE_CONTROL_ENDPOINT: ${{ secrets.BEDROCK_ENDPOINT }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: pytest tests_integ/payment/ -v
```

## Additional Resources

- [Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [Payment Control Plane API Reference](https://docs.aws.amazon.com/bedrock/latest/userguide/payment-api/)
- [AWS SDK for Python (Boto3)](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
