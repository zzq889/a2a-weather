"""
Pydantic models for the agent router package.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class RoutingConfidence(str, Enum):
    """Confidence levels for routing decisions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Agent(BaseModel):
    """Model representing a specialized agent in the system."""
    name: str = Field(..., description="Unique identifier for the agent")
    description: str = Field(..., description="What this agent specializes in")
    capabilities: List[str] = Field(..., description="List of things this agent can do")
    examples: List[str] = Field(..., description="Example prompts this agent handles well")
    priority: int = Field(default=1, description="Priority level (higher = more preferred)")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent name cannot be empty")
        # Normalize name: lowercase, replace spaces with underscores
        return v.strip().lower().replace(' ', '_').replace('-', '_')

    @field_validator('capabilities', 'examples')
    @classmethod
    def validate_non_empty_lists(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("List cannot be empty")
        return [item.strip() for item in v if item.strip()]

    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Priority must be at least 1")
        return v

class RoutingResult(BaseModel):
    """Result of an agent routing decision."""
    selected_agent: str = Field(..., description="Name of the selected agent")
    confidence: RoutingConfidence = Field(..., description="Confidence level of the routing decision")
    reasoning: str = Field(..., description="Explanation for why this agent was selected")
    fallback_agents: Optional[List[str]] = Field(default=None, description="Alternative agents if primary fails")
    processing_time: Optional[float] = Field(default=None, description="Time taken to make routing decision")

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Reasoning must be at least 10 characters")
        return v

class LLMRoutingResponse(BaseModel):
    """Expected structure of LLM routing response."""
    selected_agent: str
    confidence: RoutingConfidence
    reasoning: str
    fallback_agents: Optional[List[str]] = None

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Reasoning must be at least 10 characters")
        return v

