"""Integration tests for code interpreter client.

Note: These tests require valid AWS credentials and may incur costs.
To run: pytest tests_integ/tools/test_code.py -v
"""

import os
import time

import boto3

from bedrock_agentcore._utils.endpoints import get_control_plane_endpoint
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter, code_session

# Test 1: Basic code execution with system interpreter
print("Test 1: Basic code execution using execute_code()")
with code_session("us-west-2") as client:
    result = client.execute_code("""
import math
print(f"Pi = {math.pi}")
print(f"Square root of 2 = {math.sqrt(2)}")
print("Code execution completed successfully!")
""")
    for event in result["stream"]:
        print(event["result"]["content"])
print("✅ Test 1 passed\n")


# Test 2: List files in sandbox
print("Test 2: List files in sandbox")
with code_session("us-west-2") as client:
    result = client.invoke("listFiles")
    print("Files in sandbox:")
    for event in result["stream"]:
        print(event["result"]["content"])
print("✅ Test 2 passed\n")


# Test 3: Upload file and verify
print("Test 3: Upload file using upload_file()")
with code_session("us-west-2") as client:
    # Upload a CSV file
    csv_content = "name,age,city\nAlice,30,Seattle\nBob,25,Portland\nCharlie,35,Denver"
    client.upload_file(
        path="data.csv", content=csv_content, description="Sample user data with name, age, and city columns"
    )

    # Verify by listing files
    result = client.invoke("listFiles")
    print("Files after upload:")
    for event in result["stream"]:
        print(event["result"]["content"])
print("✅ Test 3 passed\n")


# Test 4: Upload multiple files
print("Test 4: Upload multiple files using upload_files()")
with code_session("us-west-2") as client:
    files = [
        {"path": "config.json", "content": '{"setting1": true, "setting2": 42}'},
        {"path": "script.py", "content": "print('Hello from script!')"},
        {"path": "notes.txt", "content": "These are some notes."},
    ]
    client.upload_files(files)

    # Verify
    result = client.invoke("listFiles")
    print("Files after multi-upload:")
    for event in result["stream"]:
        print(event["result"]["content"])
print("✅ Test 4 passed\n")


# Test 5: Install packages
print("Test 5: Install packages using install_packages()")
with code_session("us-west-2") as client:
    # Install packages
    result = client.install_packages(["requests", "beautifulsoup4"])
    print("Package installation result:")
    for event in result["stream"]:
        print(event["result"]["content"])

    # Verify installation by importing
    verify_result = client.execute_code("""
import requests
import bs4
print(f"requests version: {requests.__version__}")
print(f"beautifulsoup4 version: {bs4.__version__}")
""")
    print("Verification:")
    for event in verify_result["stream"]:
        print(event["result"]["content"])
print("✅ Test 5 passed\n")


# Test 6: Execute shell command
print("Test 6: Execute shell command using execute_command()")
with code_session("us-west-2") as client:
    # Check Python version
    result = client.execute_command("python --version")
    print("Shell command result:")
    for event in result["stream"]:
        print(event["result"]["content"])

    # List directory
    result = client.execute_command("ls -la")
    print("Directory listing:")
    for event in result["stream"]:
        print(event["result"]["content"])
print("✅ Test 6 passed\n")


# Test 7: Upload, process, and download file
print("Test 7: Full workflow - upload, process, download")
with code_session("us-west-2") as client:
    # Upload data
    csv_data = "x,y\n1,2\n3,4\n5,6\n7,8\n9,10"
    client.upload_file(path="input.csv", content=csv_data)

    # Process with pandas
    client.install_packages(["pandas"])

    process_result = client.execute_code("""
import pandas as pd

# Read input
df = pd.read_csv('input.csv')

# Process - add computed column
df['sum'] = df['x'] + df['y']
df['product'] = df['x'] * df['y']

# Save output
df.to_csv('output.csv', index=False)
print(df)
print("\\nOutput saved to output.csv")
""")
    print("Processing result:")
    for event in process_result["stream"]:
        print(event["result"]["content"])

    # Download result
    output_content = client.download_file("output.csv")
    print(f"\nDownloaded output.csv:\n{output_content}")
print("✅ Test 7 passed\n")


# Test 8: Download multiple files
print("Test 8: Download multiple files using download_files()")
with code_session("us-west-2") as client:
    # Create some files
    client.execute_code("""
with open('file1.txt', 'w') as f:
    f.write('Content of file 1')
with open('file2.txt', 'w') as f:
    f.write('Content of file 2')
print('Files created')
""")

    # Download both
    files = client.download_files(["file1.txt", "file2.txt"])
    print("Downloaded files:")
    for path, content in files.items():
        print(f"  {path}: {content}")
print("✅ Test 8 passed\n")


# Test 9: Execute code with clear_context
print("Test 9: Execute code with clear_context")
with code_session("us-west-2") as client:
    # Set a variable
    client.execute_code("my_variable = 42")

    # Verify it exists
    result = client.execute_code("print(f'my_variable = {my_variable}')")
    print("Before clear_context:")
    for event in result["stream"]:
        print(event["result"]["content"])

    # Clear context and try again
    result = client.execute_code(
        """
try:
    print(f'my_variable = {my_variable}')
except NameError:
    print('my_variable is not defined (context was cleared)')
""",
        clear_context=True,
    )
    print("After clear_context:")
    for event in result["stream"]:
        print(event["result"]["content"])
print("✅ Test 9 passed\n")


# Test 10: Data visualization workflow
print("Test 10: Data visualization with matplotlib")
with code_session("us-west-2") as client:
    client.install_packages(["matplotlib", "numpy"])

    result = client.execute_code("""
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.savefig('sine_wave.png', dpi=100, bbox_inches='tight')
plt.close()

print("Plot saved to sine_wave.png")
""")

    print("Visualization result:")
    for event in result["stream"]:
        print(event["result"]["content"])

    # Verify file was created
    list_result = client.execute_command("ls -la *.png")
    for event in list_result["stream"]:
        print(event["result"]["content"])
print("✅ Test 10 passed\n")

# Test 11: Create code interpreter with custom root CA certificate
print("Test 11: Code interpreter with custom root CA certificate")

REGION = "us-west-2"
secret_arn = os.environ.get("TEST_ROOT_CA_SECRET_ARN")
execution_role_arn = os.environ.get("TEST_EXECUTION_ROLE_ARN")

if secret_arn and execution_role_arn:
    cp_client = boto3.client(
        "bedrock-agentcore-control",
        region_name=REGION,
        endpoint_url=get_control_plane_endpoint(REGION),
    )

    # Create interpreter with root CA
    import time

    response = cp_client.create_code_interpreter(
        name=f"integ_test_rootca_{int(time.time())}",
        executionRoleArn=execution_role_arn,
        networkConfiguration={"networkMode": "PUBLIC"},
        certificates=[{"location": {"secretsManager": {"secretArn": secret_arn}}}],
        description="Integration test: code interpreter with custom root CA",
    )
    interpreter_id = response["codeInterpreterId"]
    print(f"  Created interpreter: {interpreter_id}")

    # Wait for ready
    ci_client = CodeInterpreter(REGION)
    for _ in range(40):
        info = ci_client.get_code_interpreter(interpreter_id)
        if info["status"] == "READY":
            break
        elif info["status"] == "CREATE_FAILED":
            raise RuntimeError(f"Interpreter creation failed: {info.get('failureReason')}")
        time.sleep(3)

    print(f"  Interpreter status: {info['status']}")

    # Start session and test SSL connection to untrusted-root.badssl.com
    ci_client.start(identifier=interpreter_id)
    print(f"  Session started: {ci_client.session_id}")

    result = ci_client.invoke(
        "executeCode",
        {
            "code": (
                "import urllib.request\n"
                "response = urllib.request.urlopen("
                '"https://untrusted-root.badssl.com")\n'
                'print(f"Status: {response.status}")'
            ),
            "language": "python",
        },
    )

    success = False
    for event in result.get("stream", []):
        if "result" in event:
            structured = event["result"].get("structuredContent", {})
            stdout = structured.get("stdout", "")
            if "200" in stdout:
                success = True
                print(f"  SSL connection succeeded: {stdout.strip()}")

    ci_client.stop()

    # Cleanup - delete interpreter
    sessions = ci_client.list_sessions(interpreter_id=interpreter_id, status="READY")
    for s in sessions.get("items", []):
        try:
            ci_client.data_plane_client.stop_code_interpreter_session(
                codeInterpreterIdentifier=interpreter_id, sessionId=s["sessionId"]
            )
        except Exception:
            pass
    time.sleep(5)
    cp_client.delete_code_interpreter(codeInterpreterId=interpreter_id)
    print(f"  Cleaned up interpreter: {interpreter_id}")

    assert success, "SSL connection to untrusted-root.badssl.com should succeed with custom root CA"
    print("✅ Test 11 passed\n")
else:
    print("  ⏭️ Skipped (set TEST_ROOT_CA_SECRET_ARN and TEST_EXECUTION_ROLE_ARN env vars to enable)")
    print("  Example:")
    print("    export TEST_ROOT_CA_SECRET_ARN=arn:aws:secretsmanager:us-west-2:123456789012:secret:badssl-root-ca")
    print("    export TEST_EXECUTION_ROLE_ARN=arn:aws:iam::123456789012:role/AgentCoreRole")
    print("✅ Test 11 skipped\n")

print("=" * 50)
print("All integration tests passed! ✅")
print("=" * 50)
