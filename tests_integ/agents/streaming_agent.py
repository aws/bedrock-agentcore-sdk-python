from strands import Agent

from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()
agent = Agent()


@app.entrypoint
async def agent_invocation(payload):
    """Handler for agent invocation"""
    user_message = payload.get(
        "prompt", "No prompt found in input, please guide customer to create a json payload with prompt key"
    )
    stream = agent.stream_async(user_message)
    async for event in stream:
        app.logger.info("Streaming event: %s", event)
        yield (event)


if __name__ == "__main__":
    app.run()
