# E2E Test Fixtures — Source Provenance

| File | Source | Description |
|------|--------|-------------|
| strands_qa_spans.json | Captured from Strands BMI agent deployed via AgentCore runtime | invoke_agent + execute_tool spans (calculate_bmi, calculate_daily_calories) |
| strands_rag_spans.json | AgentCoreEvaluationOpenTelemetryMapper `test/resources/agentcore_observability_logs/strands_telemetry_tracer/healthcare_single_agent_2_turns_with_tools_info/downloaded_logs.json` | Strands healthcare agent, 2 turns with tool info (CloudWatch ADOT format) |
| strands_tool_call_spans.json | AgentCoreEvaluationOpenTelemetryMapper `test/resources/in_memory_spans/strands_telemetry_tracer/travel_agent/sea-nyc-trip-2-turns-20260414130401.json` | Strands travel agent, in-memory format with inline events |
| openinference_langchain_spans.json | AgentCoreEvaluationOpenTelemetryMapper `test/resources/agentcore_observability_logs/openinference_instrumentation_langchain/weather_prebuilt_create_agent_api_2_turns_no_custom_agent_name_invoke_using_langgraphnative/downloaded_logs.json` | LangGraph weather agent, OpenInference instrumentation |
| opentelemetry_langchain_spans.json | AgentCoreEvaluationOpenTelemetryMapper `test/resources/agentcore_observability_logs/opentelemetry_instrumentation_langchain/travel_prebuilt_agent/sea-nyc-trip-2-turns-adot_v17-opentelemetry_0_60_0-20260428005329.json` | LangGraph travel agent, OpenTelemetry instrumentation (adot v17, 2 turns with tool calls) |
| custom_flat_spans.json | AgentCoreEvaluationOpenTelemetryMapper `test/resources/generic_openinference/in_memory/google_adk_travel/full_trace.json` | Google ADK travel agent (unsupported by built-in mapper — tests custom_mapper path) |
| custom_nested_spans.json | AgentCoreEvaluationOpenTelemetryMapper `test/resources/generic_openinference/in_memory/openai_agents_travel/full_trace.json` | OpenAI Agents travel agent (unsupported by built-in mapper — tests custom_mapper path) |



