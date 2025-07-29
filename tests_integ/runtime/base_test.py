import logging
import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from subprocess import Popen
from typing import IO, Generator

logger = logging.getLogger("sdk-runtime-base-test")

AGENT_SERVER_ENDPOINT = "http://127.0.0.1:8080"


class BaseSDKRuntimeTest(ABC):
    def run(self, tmp_path) -> None:
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            self.setup()

            logger.info("Running test...")
            self.run_test()

        finally:
            os.chdir(original_dir)

    def setup(self) -> None:
        return

    @abstractmethod
    def run_test(self) -> None:
        raise NotImplementedError


@contextmanager
def start_agent_server(agent_module, timeout=5) -> Generator[Popen, None, None]:
    logger.info("Starting agent server...")
    start_time = time.time()

    try:
        agent_server = Popen(
            ["python", "-m", agent_module], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        while time.time() - start_time < timeout:
            if agent_server.stdout is None:
                raise RuntimeError("Agent server has no configured output")

            if agent_server.poll() is not None:
                out = agent_server.stdout.read()
                raise RuntimeError(f"Error when running agent server: {out}")

            line = agent_server.stdout.readline()
            while line:
                line = line.strip()
                if line:
                    logger.info(line)
                    if "Uvicorn running on http://127.0.0.1:8080" in line:
                        _start_logging_thread(agent_server.stdout)
                        yield agent_server
                        return
                line = agent_server.stdout.readline()

            time.sleep(0.5)
        raise TimeoutError(f"Agent server did not start within {timeout} seconds")
    finally:
        _stop_agent_server(agent_server)


def _stop_agent_server(agent_server: Popen) -> None:
    logger.info("Stopping agent server...")
    if agent_server.poll() is None:  # Process is still running
        logger.info("Terminating agent server process...")
        agent_server.terminate()

        # Wait for graceful shutdown
        try:
            agent_server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Agent server didn't terminate, force killing...")
            agent_server.kill()
            agent_server.wait()
        finally:
            if agent_server.stdout:
                agent_server.stdout.close()
    logger.info("Agent server terminated")


def _start_logging_thread(stdout: IO[str]):
    def log_server_output():
        logger.info("Server logging thread started")
        # thread is stopped when stdout is closed
        for line in iter(stdout.readline, ""):
            if line.strip():
                logger.info(line.strip())
        logger.info("Server logging thread stopped")

    logging_thread = threading.Thread(target=log_server_output, daemon=True, name="AgentServerLogger")
    logging_thread.start()
    return logging_thread
