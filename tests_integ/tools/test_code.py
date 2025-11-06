"""Integration tests for code interpreter client.

Note: These tests require valid AWS credentials and may incur costs.
To run: pytest tests_integ/tools/test_code.py -v
"""

from bedrock_agentcore.tools.code_interpreter_client import code_session

# Test 1: Basic code execution with system interpreter
print("Test 1: Basic code execution")
with code_session("us-west-2") as client:
    # Execute Python code
    code_to_execute = """
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
plt.show()

print("Code execution completed successfully!")
"""

    # Execute the code
    result = client.invoke("executeCode", {"language": "python", "code": code_to_execute})

    # Process the streaming results
    for event in result["stream"]:
        print(event["result"]["content"])
print("✅ Test 1 passed")

# Test 2: List files in sandbox
print("\nTest 2: List files in sandbox")
with code_session("us-west-2") as client:
    result = client.invoke("listFiles")
    print("Files in sandbox:")
    for event in result["stream"]:
        print(event["result"]["content"])
print("✅ Test 2 passed")
