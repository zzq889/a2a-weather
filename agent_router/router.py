"""
Main router implementation for the multi-agent system.
"""

import json
import time
from typing import Dict, List, Optional, Callable, Set
from pydantic import ValidationError, field_validator

from .models import Agent, RoutingResult, RoutingConfidence, LLMRoutingResponse
from .exceptions import (
    AgentNotFoundError,
    RoutingError,
    InvalidAgentError,
    LLMResponseError
)

class LLMAgentRouter:
    """Main router class for directing user prompts to appropriate agents."""

    def __init__(self, agents: List[Agent]):
        """
        Initialize the router with a list of agents.

        Args:
            agents: List of Agent objects to route between

        Raises:
            InvalidAgentError: If agent configuration is invalid
        """
        self.agents = {}
        self._initialize_agents(agents)

    def _initialize_agents(self, agents: List[Agent]) -> None:
        """Initialize and validate agents."""
        for agent in agents:
            if agent.name in self.agents:
                raise InvalidAgentError(f"Duplicate agent name: {agent.name}")
            self.agents[agent.name] = agent

        if not self.agents:
            raise InvalidAgentError("At least one agent must be provided")

        self._valid_agent_names = set(self.agents.keys())
        self._create_validated_routing_result()

    def _create_validated_routing_result(self) -> None:
        """Create a RoutingResult class with agent name validation."""
        valid_agents = self._valid_agent_names

        class ValidatedRoutingResult(RoutingResult):
            @field_validator('selected_agent')
            @classmethod
            def validate_selected_agent(cls, v: str) -> str:
                if v not in valid_agents:
                    raise ValueError(f"Agent '{v}' not found. Valid agents: {list(valid_agents)}")
                return v

            @field_validator('fallback_agents')
            @classmethod
            def validate_fallback_agents(cls, v: Optional[List[str]]) -> Optional[List[str]]:
                if v is None:
                    return v
                # Filter out invalid agents instead of raising error
                valid_fallbacks = [agent for agent in v if agent in valid_agents]
                return valid_fallbacks if valid_fallbacks else None

        self._ValidatedRoutingResult = ValidatedRoutingResult

    def create_routing_prompt(self, user_prompt: str, include_examples: bool = True) -> str:
        """
        Create the routing prompt for the LLM.

        Args:
            user_prompt: The user's input prompt
            include_examples: Whether to include example prompts for each agent

        Returns:
            Formatted prompt for the LLM
        """
        # Sort agents by priority (higher priority first)
        sorted_agents = sorted(self.agents.values(), key=lambda x: x.priority, reverse=True)

        agent_descriptions = []
        for agent in sorted_agents:
            capabilities_str = "\n    - ".join(agent.capabilities)

            agent_desc = f"""
**{agent.name}** (Priority: {agent.priority})
Description: {agent.description}
Capabilities:
    - {capabilities_str}"""

            if include_examples:
                examples_str = "\n    - ".join(agent.examples)
                agent_desc += f"""
Examples of suitable prompts:
    - {examples_str}"""

            agent_descriptions.append(agent_desc)

        agents_text = "\n".join(agent_descriptions)
        valid_agents = list(self.agents.keys())

        routing_prompt = f"""
You are an intelligent routing system for a multi-agent chat application. Your job is to analyze user prompts and determine which specialized agent should handle the request.

Available Agents:
{agents_text}

Valid agent names: {valid_agents}

Instructions:
1. Analyze the user prompt carefully for intent, domain, and complexity
2. Consider which agent's capabilities best match the request
3. Factor in agent priorities (higher priority agents are preferred when capabilities overlap)
4. Provide clear reasoning for your decision
5. Suggest 1-2 fallback agents if applicable

User Prompt: "{user_prompt}"

Respond with a JSON object in this exact format:
{{
    "selected_agent": "agent_name",
    "confidence": "high|medium|low",
    "reasoning": "Detailed explanation of why this agent was selected",
    "fallback_agents": ["agent1", "agent2"]
}}

Requirements:
- selected_agent MUST be one of: {valid_agents}
- confidence MUST be one of: ["high", "medium", "low"]
- reasoning should be clear and specific (minimum 10 characters)
- fallback_agents should be 1-2 valid agent names (optional)
"""
        return routing_prompt

    def parse_routing_response(self, llm_response: str) -> RoutingResult:
        """
        Parse the LLM response into a RoutingResult object.

        Args:
            llm_response: Raw response from the LLM

        Returns:
            RoutingResult object

        Raises:
            LLMResponseError: If response cannot be parsed
        """
        try:
            # Extract JSON from response
            start = llm_response.find('{')
            end = llm_response.rfind('}') + 1
            if start == -1 or end == 0:
                raise LLMResponseError("No JSON object found in response")

            json_str = llm_response[start:end]

            # Parse with Pydantic for validation
            llm_result = LLMRoutingResponse.model_validate_json(json_str)

            # Create validated result
            result = self._ValidatedRoutingResult(
                selected_agent=llm_result.selected_agent,
                confidence=llm_result.confidence,
                reasoning=llm_result.reasoning,
                fallback_agents=llm_result.fallback_agents
            )

            # Convert back to standard RoutingResult
            return RoutingResult(
                selected_agent=result.selected_agent,
                confidence=result.confidence,
                reasoning=result.reasoning,
                fallback_agents=result.fallback_agents
            )

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            raise LLMResponseError(f"Failed to parse routing response: {str(e)}", e)

    def _get_fallback_agent(self) -> str:
        """Get a sensible fallback agent."""
        # Prefer general_assistant, then highest priority agent
        if "general_assistant" in self.agents:
            return "general_assistant"

        # Return highest priority agent
        sorted_agents = sorted(self.agents.values(), key=lambda x: x.priority, reverse=True)
        return sorted_agents[0].name

    def route_prompt(
        self,
        user_prompt: str,
        llm_function: Callable[[str], str],
        include_examples: bool = True
    ) -> RoutingResult:
        """
        Route a user prompt to the appropriate agent.

        Args:
            user_prompt: The user's input prompt
            llm_function: Callable that takes a prompt and returns LLM response
            include_examples: Whether to include examples in the routing prompt

        Returns:
            RoutingResult object
        """
        start_time = time.time()

        try:
            if not user_prompt or not user_prompt.strip():
                return RoutingResult(
                    selected_agent=self._get_fallback_agent(),
                    confidence=RoutingConfidence.LOW,
                    reasoning="Empty prompt provided, using fallback agent",
                    fallback_agents=list(self._valid_agent_names),
                    processing_time=time.time() - start_time
                )

            routing_prompt = self.create_routing_prompt(user_prompt, include_examples)
            llm_response = llm_function(routing_prompt)
            result = self.parse_routing_response(llm_response)

            # Add processing time
            result.processing_time = time.time() - start_time
            return result

        except LLMResponseError:
            # Re-raise LLM response errors
            raise
        except Exception as e:
            # Handle other errors with fallback
            return RoutingResult(
                selected_agent=self._get_fallback_agent(),
                confidence=RoutingConfidence.LOW,
                reasoning=f"Routing failed: {str(e)}. Using fallback agent.",
                fallback_agents=list(self._valid_agent_names),
                processing_time=time.time() - start_time
            )

    def get_agent_info(self, agent_name: str) -> Agent:
        """
        Get information about a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent object

        Raises:
            AgentNotFoundError: If agent is not found
        """
        if agent_name not in self.agents:
            raise AgentNotFoundError(agent_name, list(self.agents.keys()))
        return self.agents[agent_name]

    def list_agents(self) -> List[str]:
        """Get list of all available agent names."""
        return list(self.agents.keys())

    def add_agent(self, agent: Agent) -> None:
        """
        Add a new agent to the router.

        Args:
            agent: Agent object to add

        Raises:
            InvalidAgentError: If agent already exists
        """
        if agent.name in self.agents:
            raise InvalidAgentError(f"Agent '{agent.name}' already exists")

        self.agents[agent.name] = agent
        self._valid_agent_names.add(agent.name)
        self._create_validated_routing_result()

    def remove_agent(self, agent_name: str) -> bool:
        """
        Remove an agent from the router.

        Args:
            agent_name: Name of the agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            self._valid_agent_names.remove(agent_name)
            self._create_validated_routing_result()
            return True
        return False

    def update_agent(self, agent: Agent) -> None:
        """
        Update an existing agent.

        Args:
            agent: Updated agent object

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        if agent.name not in self.agents:
            raise AgentNotFoundError(agent.name, list(self.agents.keys()))

        self.agents[agent.name] = agent
        self._create_validated_routing_result()

