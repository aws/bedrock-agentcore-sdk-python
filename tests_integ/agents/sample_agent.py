import asyncio

from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload):
    print(payload)
    print("Starting long invoke...")
    await asyncio.sleep(60)  # 1 minute sleep
    print("Finished long invoke")
    return {"message": "hello after 1 minute"}


app.run()
