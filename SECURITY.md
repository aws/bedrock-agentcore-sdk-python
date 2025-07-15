# Security Policy

## Reporting Security Vulnerabilities

We take security seriously at AWS. If you discover a security vulnerability in the Bedrock AgentCore Python SDK, we appreciate your help in disclosing it to us in a responsible manner.

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report a Security Vulnerability

If you believe you have found a security vulnerability in this SDK, please report it to us through one of the following methods:

#### For All Users
- **Email**: aws-security@amazon.com
- **Web Form**: [AWS Vulnerability Reporting](https://aws.amazon.com/security/vulnerability-reporting/)

Please provide the following information to help us understand the nature and scope of the issue:

- **Type of issue** (e.g., credential exposure, injection vulnerability, authentication bypass, etc.)
- **Full paths of source file(s)** related to the issue
- **Location of affected code** (tag/branch/commit or direct URL)
- **Special configuration** required to reproduce
- **Step-by-step instructions** to reproduce
- **Proof-of-concept or exploit code** (if possible)
- **Impact assessment** - how an attacker might exploit this

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Initial Assessment**: Our security team will evaluate your report and respond within 5 business days
- **Status Updates**: We will keep you informed about our progress
- **Resolution**: We will notify you when the vulnerability is fixed
- **Recognition**: We will acknowledge your contribution (unless you prefer to remain anonymous)

## Security Response Process

1. **Report received** - Security team acknowledges receipt
2. **Triage** - Severity assessment and impact analysis
3. **Fix development** - Creating and testing patches
4. **Release** - Coordinated disclosure and patch release
5. **Post-mortem** - Analysis and process improvements

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| 1.x.x   | :white_check_mark: | Current stable release |
| 0.x.x   | :x:                | Pre-release versions |

## Security Best Practices for SDK Users

### 1. Credential Management

**❌ NEVER DO THIS:**
```python
# Never hardcode credentials
client = MemoryClient(
    aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
    aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
)
```

**✅ DO THIS INSTEAD:**
```python
# Use environment variables
client = MemoryClient()  # Uses AWS credential chain

# Or use IAM roles (recommended for production)
client = MemoryClient()  # Automatically uses instance role
```

### 2. Secure Communication

- Always use HTTPS endpoints (enforced by default)
- Never disable SSL certificate verification
- Keep TLS libraries updated

### 3. Token Handling

```python
# ✅ Good: Token handled securely
@requires_access_token(provider_name="github", scopes=["repo:read"])
async def my_function(payload, access_token):
    # Token is injected securely, never logged
    pass

# ❌ Bad: Never log tokens
logger.info(f"Token: {access_token}")  # NEVER DO THIS
```

### 4. Input Validation

- Always validate user inputs before passing to SDK
- Use the built-in Pydantic models for type safety
- Sanitize data that will be stored or processed

### 5. Least Privilege

- Grant minimal IAM permissions required
- Use resource-based policies where possible
- Regularly audit and reduce permissions

### 6. Monitoring & Logging

- Enable CloudTrail for API audit logs
- Use CloudWatch for operational monitoring
- Never log sensitive data (tokens, credentials, PII)

## Security Features

The Bedrock AgentCore SDK includes these security features:

### Built-in Protections
- **Automatic credential handling** via AWS credential provider chain
- **TLS 1.2+ enforcement** for all AWS API calls
- **Request signing** using AWS Signature Version 4
- **Input validation** using Pydantic models
- **Memory safety** - no credential storage, secure cleanup

### Authentication Support
- AWS IAM (SigV4) authentication
- OAuth2 with PKCE support
- API key management
- Workload identity tokens

### Secure Defaults
- SSL verification always enabled
- Secure session management
- Request size limits
- Timeout configurations

## Common Security Vulnerabilities to Avoid

### 1. Credential Exposure
- Never commit credentials to version control
- Don't pass credentials as command-line arguments
- Avoid credentials in configuration files

### 2. Injection Attacks
- Always use parameterized inputs
- Validate and sanitize user data
- Use SDK-provided methods for data handling

### 3. Insufficient Access Controls
- Implement proper authentication
- Use IAM policies effectively
- Enable MFA where possible

### 4. Insecure Data Transmission
- Always use HTTPS
- Verify SSL certificates
- Use latest TLS versions

## Security Tools Integration

### For Development
```bash
# Install security scanning tools
pip install bandit safety

# Run security scan
bandit -r src/

# Check for known vulnerabilities
safety check
```

### For CI/CD
- Enable GitHub Dependabot
- Use CodeQL analysis
- Implement pre-commit hooks
- Regular dependency updates

## Compliance

This SDK is designed to help you build applications that can comply with:
- AWS Well-Architected Security Pillar
- OWASP Secure Coding Practices
- Common compliance frameworks (when properly configured)

## Additional Resources

- [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [OWASP Python Security](https://owasp.org/www-project-python-security/)
- [Python Security Guidelines](https://python.org/dev/security/)

## Contact

For non-security related issues, please use [GitHub Issues](https://github.com/aws/bedrock-agentcore-python-sdk/issues).

For security-related questions that don't require immediate attention, please see our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

---

*Last updated: July 2025*
*This security policy may be updated at any time. Please check back regularly for updates.*
