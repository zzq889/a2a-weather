"""
Custom exceptions for the agent router package.
"""

class AgentRouterError(Exception):
    """Base exception for agent router errors."""
    pass

class AgentNotFoundError(AgentRouterError):
    """Raised when a specified agent is not found."""
    def __init__(self, agent_name: str, available_agents: list):
        self.agent_name = agent_name
        self.available_agents = available_agents
        super().__init__(f"Agent '{agent_name}' not found. Available agents: {available_agents}")

class RoutingError(AgentRouterError):
    """Raised when routing fails."""
    def __init__(self, message: str, original_error: Exception = None):
        self.original_error = original_error
        super().__init__(message)

class InvalidAgentError(AgentRouterError):
    """Raised when agent configuration is invalid."""
    pass

class LLMResponseError(RoutingError):
    """Raised when LLM response cannot be parsed."""
    pass
