import asyncio

from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload):
    app.logger.info("Received payload: %s", payload)
    app.logger.info("Starting long invoke...")
    await asyncio.sleep(60)  # 1 minute sleep
    app.logger.info("Finished long invoke")
    return {"message": "hello after 1 minute"}


app.run()
