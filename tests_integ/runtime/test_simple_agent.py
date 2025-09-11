import logging
import textwrap

from tests_integ.runtime.base_test import AGENT_SERVER_ENDPOINT, BaseSDKRuntimeTest, start_agent_server
from tests_integ.runtime.http_client import HttpClient

logger = logging.getLogger("sdk-runtime-simple-agent-test")


class TestSDKSimpleAgent(BaseSDKRuntimeTest):
    def setup(self):
        self.agent_module = "agent"
        with open(self.agent_module + ".py", "w") as file:
            content = textwrap.dedent("""
                from bedrock_agentcore import BedrockAgentCoreApp
                from strands import Agent

                app = BedrockAgentCoreApp(debug=True)
                agent = Agent()

                @app.entrypoint
                async def agent_invocation(payload):
                    return agent(payload.get("message"))

                app.run()
            """).strip()
            file.write(content)

    def run_test(self):
        with start_agent_server(self.agent_module):
            client = HttpClient(AGENT_SERVER_ENDPOINT)

            ping_response = client.ping()
            logger.info(ping_response)
            assert "Healthy" in ping_response

            response = client.invoke_endpoint("tell me a joke")
            logger.info(response)
            assert "Because they make up everything!" in response


def test(tmp_path):
    TestSDKSimpleAgent().run(tmp_path)
