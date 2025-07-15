# BedrockAgentCore Async Task Management

## Three Ways to Manage Async Tasks

### 1. Async Task Annotation
Automatically track async functions:

```python
@app.async_task
async def background_work():
    await asyncio.sleep(10)  # Status becomes "HealthyBusy"
    return "done"

@app.entrypoint
async def handler(event):
    asyncio.create_task(background_work())
    return {"status": "started"}
```

### 2. Custom Ping Handler
Override automatic status with custom logic:

```python
@app.ping
def custom_status():
    if system_busy():
        return PingStatus.HEALTHY_BUSY
    return PingStatus.HEALTHY
```

### 3. Manual Task Management
Manually control task tracking:

```python
@app.entrypoint
async def handler(event):
    # Start tracking
    task_id = app.add_async_task("data_processing", {"batch": 100})

    # Do work
    process_data()

    # Stop tracking
    app.complete_async_task(task_id)
    return {"status": "completed"}
```

## Ping Status Contract

- **HEALTHY**: Ready for new work
- **HEALTHY_BUSY**: Currently processing, avoid new work

**Priority Order:**
1. **Forced Status** (debug actions)
2. **Custom Handler** (`@app.ping`)
3. **Automatic** (based on active `@app.async_task` functions)

## Debug Methods

Enable with `app = BedrockAgentCoreApp(debug=True)`

**Check Status:**
```json
{"_agent_core_app_action": "ping_status"}
```

**List Running Tasks:**
```json
{"_agent_core_app_action": "job_status"}
```

**Force Status:**
```json
{"_agent_core_app_action": "force_healthy"}
{"_agent_core_app_action": "force_busy"}
{"_agent_core_app_action": "clear_forced_status"}
```

## API Reference

```python
# Manual task management
task_id = app.add_async_task("task_name", metadata={"key": "value"})
success = app.complete_async_task(task_id)  # Returns True/False

# Status control
app.force_ping_status(PingStatus.HEALTHY)
app.clear_forced_ping_status()

# Information
status = app.get_current_ping_status()
info = app.get_async_task_info()
