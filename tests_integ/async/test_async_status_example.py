#!/usr/bin/env python3
"""
Test script for async_status_example.py - demonstrates async task management and ping status functionality.

This script tests all the endpoints and features of the async status example.
"""

import time
from typing import Any, Dict

import requests


class AsyncStatusExampleTester:
    """Test harness for the async status example."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "X-Custom-Header": "TestValue"})

    def test_ping_endpoint(self):
        """Test the GET /ping endpoint."""
        print("🔍 Testing GET /ping endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/ping")
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"   Response: {data}")

                # Validate response structure
                assert "status" in data, "Missing 'status' field"
                assert "time_of_last_update" in data, "Missing 'time_of_last_update' field"
                assert data["status"] in ["Healthy", "HealthyBusy"], f"Invalid status: {data['status']}"
                assert isinstance(data["time_of_last_update"], int), "Timestamp should be integer"

                print("   ✅ Ping endpoint working correctly")
                return data
            else:
                print(f"   ❌ Ping endpoint failed with status {response.status_code}")
                return None
        except Exception as e:
            print(f"   ❌ Error testing ping endpoint: {e}")
            return None

    def test_rpc_action(self, action: str, expected_fields: list = None) -> Dict[Any, Any]:
        """Test a debug action via POST /invocations."""
        print(f"🔍 Testing debug action: {action}")
        try:
            payload = {"_agent_core_app_action": action}
            response = self.session.post(f"{self.base_url}/invocations", json=payload)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"   Response: {data}")

                if expected_fields:
                    for field in expected_fields:
                        assert field in data, f"Missing expected field: {field}"

                print(f"   ✅ debug action '{action}' working correctly")
                return data
            else:
                print(f"   ❌ debug action '{action}' failed with status {response.status_code}")
                return {}
        except Exception as e:
            print(f"   ❌ Error testing debug action '{action}': {e}")
            return {}

    def test_business_action(self, action: str, payload: dict = None) -> Dict[Any, Any]:
        """Test a regular business logic action."""
        print(f"🔍 Testing business action: {action}")
        try:
            request_payload = {"action": action}
            if payload:
                request_payload.update(payload)

            response = self.session.post(f"{self.base_url}/invocations", json=request_payload)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"   Response: {data}")
                print(f"   ✅ Business action '{action}' working correctly")
                return data
            else:
                print(f"   ❌ Business action '{action}' failed with status {response.status_code}")
                return {}
        except Exception as e:
            print(f"   ❌ Error testing business action '{action}': {e}")
            return {}

    def run_comprehensive_test(self):
        """Run a comprehensive test of all functionality."""
        print("🚀 Starting comprehensive async status example test...")
        print("=" * 60)

        # Test 1: Initial ping status (should be Healthy)
        print("\n📍 Test 1: Initial ping status")
        initial_ping = self.test_ping_endpoint()
        if initial_ping and initial_ping["status"] != "Healthy":
            print(f"   ⚠️  Expected 'Healthy' status initially, got: {initial_ping['status']}")

        # Test 2: Debug Actions
        print("\n📍 Test 2: Debug Actions")
        self.test_rpc_action("ping_status", ["status", "time_of_last_update"])
        self.test_rpc_action("job_status", ["active_count", "running_jobs"])

        # Test 3: Business Logic - Get Info
        print("\n📍 Test 3: Business Logic - Default Info")
        self.test_business_action("info")

        # Test 4: Force Status to Busy
        print("\n📍 Test 4: Force Status to HealthyBusy")
        self.test_rpc_action("force_busy")

        # Verify status changed
        print("\n📍 Test 4a: Verify status is now HealthyBusy")
        busy_ping = self.test_ping_endpoint()
        if busy_ping and busy_ping["status"] != "HealthyBusy":
            print(f"   ⚠️  Expected 'HealthyBusy' after forcing, got: {busy_ping['status']}")

        # Test 5: Force Status back to Healthy
        print("\n📍 Test 5: Force Status back to Healthy")
        self.test_rpc_action("force_healthy")

        # Verify status changed back
        print("\n📍 Test 5a: Verify status is now Healthy")
        healthy_ping = self.test_ping_endpoint()
        if healthy_ping and healthy_ping["status"] != "Healthy":
            print(f"   ⚠️  Expected 'Healthy' after forcing, got: {healthy_ping['status']}")

        # Test 6: Start Background Tasks
        print("\n📍 Test 6: Start Single Background Task")
        self.test_business_action("start_background_task")

        # Wait a moment for task to start
        print("   ⏳ Waiting 2 seconds for task to start...")
        time.sleep(2)

        # Check if status became busy
        print("\n📍 Test 6a: Check if status became HealthyBusy")
        task_ping = self.test_ping_endpoint()
        if task_ping and task_ping["status"] == "HealthyBusy":
            print("   ✅ Status correctly changed to HealthyBusy with active task")
        else:
            print(f"   ⚠️  Expected 'HealthyBusy' with active task, got: {task_ping['status'] if task_ping else 'None'}")

        # Test 7: Check Job Status
        print("\n📍 Test 7: Check Job Status with Active Tasks")
        job_status = self.test_rpc_action("job_status", ["active_count", "running_jobs"])
        if job_status and job_status.get("active_count", 0) > 0:
            print(f"   ✅ Found {job_status['active_count']} active task(s)")
            for i, job in enumerate(job_status.get("running_jobs", [])):
                print(f"   Task {i + 1}: {job.get('name', 'unknown')} - Duration: {job.get('duration', 0):.1f}s")

        # Test 8: Start Multiple Tasks
        print("\n📍 Test 8: Start Multiple Background Tasks")
        self.test_business_action("start_multiple_tasks")

        # Wait a moment for tasks to start
        print("   ⏳ Waiting 2 seconds for tasks to start...")
        time.sleep(2)

        # Check job status again
        print("\n📍 Test 8a: Check Job Status with Multiple Tasks")
        multi_job_status = self.test_rpc_action("job_status", ["active_count", "running_jobs"])
        if multi_job_status and multi_job_status.get("active_count", 0) > 1:
            print(f"   ✅ Found {multi_job_status['active_count']} active tasks")

        # Test 9: Use business action to get task info
        print("\n📍 Test 9: Use Business Action to Get Task Info")
        self.test_business_action("get_task_info")

        # Test 10: Force status with business action
        print("\n📍 Test 10: Force Status via Business Action")
        self.test_business_action("force_status", {"ping_status": "HealthyBusy"})

        # Final status check
        print("\n📍 Final Test: Check Final Status")
        final_ping = self.test_ping_endpoint()

        print("\n" + "=" * 60)
        print("🎉 Comprehensive test completed!")
        print(f"📊 Final async status: {final_ping['status'] if final_ping else 'Unknown'}")
        print("📝 Note: Background tasks may still be running (they run for 5000+ seconds in the example)")
        print("🔧 Use debug actions to force status or check job details as needed")


def run_server_test():
    """Run the test assuming server is already running."""
    print("🧪 Testing async_status_example.py functionality")
    print("📋 Make sure the server is running: python async_status_example.py")
    print("")

    tester = AsyncStatusExampleTester()

    # Test server connection first
    try:
        requests.get("http://localhost:8080/ping", timeout=5)
        print("✅ Server is responding")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Please start the server first: python async_status_example.py")
        return

    # Run comprehensive test
    tester.run_comprehensive_test()


def run_quick_tests():
    """Run quick tests to validate basic functionality."""
    print("🏃‍♂️ Running quick validation tests...")

    tester = AsyncStatusExampleTester()

    try:
        # Quick connectivity test
        response = requests.get("http://localhost:8080/ping", timeout=3)
        if response.status_code != 200:
            print("❌ Server not responding correctly")
            return

        print("✅ Server connectivity OK")

        # Test basic debug actions
        ping_result = tester.test_rpc_action("ping_status")
        job_result = tester.test_rpc_action("job_status")

        # Test basic business action
        info_result = tester.test_business_action("info")

        if ping_result and job_result and info_result:
            print("🎉 Quick tests passed! Server is working correctly.")
        else:
            print("⚠️  Some quick tests failed - see details above")

    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Please start: python async_status_example.py")


if __name__ == "__main__":
    print("🔬 BedrockAgentCore Async Status Example Tester")
    print("=" * 50)

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_tests()
    else:
        print("Usage:")
        print("  python test_async_status_example.py           # Full comprehensive test")
        print("  python test_async_status_example.py --quick   # Quick validation test")
        print("")
        print("⚠️  Make sure to start the server first:")
        print("  python async_status_example.py")
        print("")

        input("Press Enter to start comprehensive test (or Ctrl+C to cancel)...")
        run_server_test()
