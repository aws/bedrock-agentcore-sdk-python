# Testing Guide for BedrockAgentCore Async Functionality

This guide explains how to test the async status and task management features.

## 🧪 Test Scripts

### 1. `async_status_example.py` - Demo Server
The main example server demonstrating all async functionality.
**Note:** The server is initialized with `debug=True` to enable debug actions.

### 2. `test_async_status_example.py` - Test Client
Comprehensive test script that validates all functionality.

## 🚀 Quick Start

### Step 1: Start the Example Server
```bash
# Terminal 1 - Navigate to async integration tests
cd tests_integ/async

# Start the server
python async_status_example.py
```

### Step 2: Run Tests
```bash
# Terminal 2 - From the async directory, run tests (choose one)

# Quick validation test (30 seconds)
python test_async_status_example.py --quick

# Full comprehensive test (2+ minutes)
python test_async_status_example.py
```

## 📋 Test Coverage

The test script validates:

### ✅ Core Endpoints
- **GET /ping** - Basic ping endpoint with timestamp
- **POST /invocations** - Main invocation endpoint

### ✅ Debug Actions (requires debug=True)
- `ping_status` - Get current status with timestamp
- `job_status` - Get running task information
- `force_healthy` - Force status to "Healthy"
- `force_busy` - Force status to "HealthyBusy"

### ✅ Business Logic
- Default info action
- Start single background task
- Start multiple background tasks
- Get task info via business logic
- Force status via business logic

### ✅ Status Transitions
- Initial "Healthy" status
- Transition to "HealthyBusy" with active tasks
- Manual status forcing and clearing
- Timestamp updates on status changes

## 🔍 Test Output Example

```
🔬 BedrockAgentCore Async Status Example Tester
==================================================

🚀 Starting comprehensive async status example test...
============================================================

📍 Test 1: Initial ping status
🔍 Testing GET /ping endpoint...
   Status: 200
   Response: {'status': 'Healthy', 'time_of_last_update': 1752264567}
   ✅ Ping endpoint working correctly

📍 Test 2: Debug Actions
🔍 Testing debug action: ping_status
   Status: 200
   Response: {'status': 'Healthy', 'time_of_last_update': 1752264567}
   ✅ Debug action 'ping_status' working correctly

🔍 Testing debug action: job_status
   Status: 200
   Response: {'active_count': 0, 'running_jobs': []}
   ✅ Debug action 'job_status' working correctly

📍 Test 3: Business Logic - Default Info
🔍 Testing business action: info
   Status: 200
   Response: {'message': 'BedrockAgentCore Async Status Demo', 'available_actions': [...]}
   ✅ Business action 'info' working correctly

...

🎉 Comprehensive test completed!
📊 Final async status: HealthyBusy
📝 Note: Background tasks may still be running (they run for 5000+ seconds in the example)
🔧 Use debug actions to force status or check job details as needed (requires debug=True)
```

## 🛠️ Manual Testing

You can also test manually using curl:

### Test Ping Endpoint
```bash
curl http://localhost:8080/ping
# Response: {"status":"Healthy","time_of_last_update":1752264567}
```

### Test Debug Actions (requires debug=True)
```bash
# Check ping status
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"_agent_core_app_action": "ping_status"}'

# Check job status
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"_agent_core_app_action": "job_status"}'

# Force status to busy
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"_agent_core_app_action": "force_busy"}'
```

### Test Business Actions
```bash
# Start background task
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"action": "start_background_task"}'

# Get task info
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"action": "get_task_info"}'
```

## 🐛 Troubleshooting

### Server Not Starting
- Check if port 8080 is available
- Look for import errors in the console
- Ensure Python 3.8+ is being used
- Verify you're running from the `tests_integ/async/` directory

### Tests Failing
- Make sure server is running first
- Check firewall/network connectivity
- Verify no other services on port 8080
- Ensure both server and test script are in the same directory

### Import Errors
- Ensure you're running from the `tests_integ/async/` directory
- Check that all source files are present
- Verify Python path includes the src directory (handled by relative imports)

## 📚 Understanding Test Results

### Status Values
- **"Healthy"** - No active tasks, ready for work
- **"HealthyBusy"** - Tasks running or status forced

### Task Information
- **active_count** - Number of currently running async tasks
- **running_jobs** - Details of each task (name, duration)
- **time_of_last_update** - Unix timestamp of last status change

### Expected Behavior
1. Server starts with "Healthy" status
2. Starting tasks changes status to "HealthyBusy"
3. Forcing status overrides automatic detection
4. Tasks can be monitored via debug actions (when debug=True)
5. Multiple concurrent tasks are tracked correctly

## 🏗️ Integration Test Structure

This async functionality is organized as integration tests because:

- **End-to-End Testing**: Tests full server/client interaction
- **Real Network Communication**: Uses actual HTTP requests
- **Complete Workflow Validation**: Tests entire async task lifecycle
- **Operational Scenarios**: Validates real-world usage patterns

### Directory Structure
```
tests_integ/async/
├── __init__.py                     # Package initialization
├── async_status_example.py        # Demo server
├── test_async_status_example.py   # Test client
├── README.md                       # API documentation
└── TESTING_GUIDE.md              # This file
```

This testing framework validates that all async status functionality works as designed in a real deployment scenario!
