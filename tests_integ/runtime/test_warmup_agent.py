import logging
import textwrap
import threading
import time

import requests

from tests_integ.runtime.base_test import BaseSDKRuntimeTest, start_agent_server
from tests_integ.runtime.http_client import HttpClient

logger = logging.getLogger("sdk-runtime-warmup-agent-test")

# Own port so this test can run in parallel with the other runtime tests (pytest -n auto).
WARMUP_PORT = 8082
WARMUP_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Warmup-Type"
MEMORY_WARMUP_SECONDS = 2


class TestSDKWarmupAgent(BaseSDKRuntimeTest):
    def setup(self):
        self.agent_module = "warmup_agent"
        with open(self.agent_module + ".py", "w") as file:
            content = textwrap.dedent(f"""
                import time

                from bedrock_agentcore import BedrockAgentCoreApp, HookConfig, WarmupType

                app = BedrockAgentCoreApp(debug=True, warmup_hook=HookConfig(path="/warmup"))
                WARMED = False

                @app.warmup(WarmupType.MEMORY)
                def load_shared_state():
                    global WARMED
                    time.sleep({MEMORY_WARMUP_SECONDS})  # blocking work; /ping must keep answering
                    WARMED = True

                # No disk handler: the disk call must still succeed.

                @app.entrypoint
                def invoke(payload):
                    return {{"warmed": WARMED}}

                app.run(port={WARMUP_PORT})
            """).strip()
            file.write(content)

    def run_test(self):
        base_url = f"http://127.0.0.1:{WARMUP_PORT}"
        with start_agent_server(self.agent_module, port=WARMUP_PORT):
            client = HttpClient(base_url)
            assert "Healthy" in client.ping()

            # Before warmup, the entrypoint sees no prepared state.
            assert '"warmed": false' in client.invoke_endpoint("{}")

            # Poll /ping from another thread while the memory warmup blocks.
            pings = []
            stop = threading.Event()

            def poll_ping():
                while not stop.is_set():
                    start = time.monotonic()
                    response = requests.get(f"{base_url}/ping", timeout=1)
                    pings.append((start, response.status_code, response.json()["status"], time.monotonic() - start))
                    time.sleep(0.1)

            poller = threading.Thread(target=poll_ping, daemon=True)
            poller.start()
            try:
                memory_start = time.monotonic()
                memory = requests.post(f"{base_url}/warmup", headers={WARMUP_HEADER: "memory"}, timeout=10)
                memory_end = time.monotonic()
                memory_duration = memory_end - memory_start
            finally:
                stop.set()
                poller.join(timeout=5)

            logger.info("memory warmup: %s %s (%.2fs)", memory.status_code, memory.text, memory_duration)
            assert memory.status_code == 200
            assert memory.json() == {"status": "ok", "type": "memory", "result": "completed"}
            assert memory_duration >= MEMORY_WARMUP_SECONDS  # responded only after the work was done

            # Only pings that started while the memory warmup was running prove /ping overlapped it.
            during = [ping for ping in pings if memory_start <= ping[0] <= memory_end]
            logger.info("pings during memory warmup: %s", during)
            assert len(during) >= 5
            assert all(code == 200 and status == "Healthy" for _, code, status, _ in during)
            assert max(duration for _, _, _, duration in during) < 0.5

            disk = requests.post(f"{base_url}/warmup", headers={WARMUP_HEADER: "disk"}, timeout=10)
            logger.info("disk warmup: %s %s", disk.status_code, disk.text)
            assert disk.status_code == 200
            assert disk.json() == {"status": "ok", "type": "disk", "result": "skipped"}

            # A type this SDK version doesn't know is skipped, so a future type can't fail the create.
            unknown = requests.post(f"{base_url}/warmup", headers={WARMUP_HEADER: "cpu"}, timeout=10)
            assert unknown.status_code == 200
            assert unknown.json() == {"status": "ok", "type": "cpu", "result": "skipped"}

            missing = requests.post(f"{base_url}/warmup", timeout=10)
            assert missing.status_code == 400

            response = client.invoke_endpoint("{}")
            logger.info(response)
            assert '"warmed": true' in response


def test(tmp_path):
    TestSDKWarmupAgent().run(tmp_path)
