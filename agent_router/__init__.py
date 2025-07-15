"""
Multi-Agent Router Package

A pure LLM-based routing system for multi-agent chat applications.
"""

from .models import Agent, RoutingResult, RoutingConfidence, AgentParameter, ParameterType
from .router import LLMAgentRouter
from .agents import create_default_agents
from .exceptions import AgentNotFoundError, RoutingError, InvalidAgentError, LLMResponseError

__version__ = "1.0.0"
__all__ = [
    "Agent",
    "RoutingResult",
    "RoutingConfidence",
    "AgentParameter",
    "ParameterType",
    "LLMAgentRouter",
    "create_default_agents",
    "AgentNotFoundError",
    "RoutingError",
    "InvalidAgentError",
    "LLMResponseError"
]

