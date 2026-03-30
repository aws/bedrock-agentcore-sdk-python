"""Invoker type definitions for the AgentCore Experiment Framework.

Defines the interface between the evaluation runner and the user's agent.
"""

from typing import Any, Callable, Optional

from pydantic import BaseModel


class AgentInvokerInput(BaseModel):
    """Input passed to the agent invoker for each turn.

    Attributes:
        payload: Input data for the agent (from dataset turn or actor simulator).
        session_id: Framework-managed session ID. Generated once per scenario and
            stable across all turns. The invoker should pass this to the agent to
            maintain conversation continuity.
    """

    payload: Any
    session_id: Optional[str] = None


class AgentInvokerOutput(BaseModel):
    """Output returned by the agent invoker after processing a single turn.

    Attributes:
        agent_output: The agent's response.
    """

    agent_output: Any


AgentInvokerFn = Callable[[AgentInvokerInput], AgentInvokerOutput]
