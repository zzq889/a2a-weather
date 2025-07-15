#!/usr/bin/env python3
"""
Test script demonstrating parameter support in agent_router.
"""

import sys
import os

# Add the parent directory to the path so we can import agent_router
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_router import LLMAgentRouter, create_default_agents, Agent, AgentParameter, ParameterType


class MockLLM:
    """Mock LLM for testing parameter extraction."""
    
    def __init__(self):
        self.call_count = 0
    
    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        
        # Mock routing response
        if "routing system" in prompt.lower():
            return '''
            {
                "selected_agent": "data_analyst",
                "confidence": "high",
                "reasoning": "The user is asking for data analysis with specific parameters mentioned",
                "fallback_agents": ["general_assistant"]
            }
            '''
        
        # Mock parameter extraction response
        elif "parameter extraction" in prompt.lower():
            return '''
            {
                "extracted_parameters": {
                    "dataset_type": "sales",
                    "analysis_goal": "trend_analysis",
                    "time_period": "last_month"
                },
                "missing_required": [],
                "confidence": "high",
                "reasoning": "Successfully extracted dataset type as sales, analysis goal as trend analysis, and time period as last month from the user prompt"
            }
            '''
        
        return "Mock LLM response"


def test_parameter_extraction():
    """Test parameter extraction functionality."""
    print("🧪 Testing Parameter Extraction")
    print("=" * 50)
    
    # Create router with default agents (includes data_analyst with parameters)
    agents = create_default_agents()
    router = LLMAgentRouter(agents)
    llm = MockLLM()
    
    # Test routing with parameter extraction
    test_prompt = "Analyze my sales data from last month to identify trends"
    
    print(f"Test prompt: \"{test_prompt}\"")
    print()
    
    result = router.route_prompt(test_prompt, llm, extract_parameters=True)
    
    print(f"✅ Selected agent: {result.selected_agent}")
    print(f"   Confidence: {result.confidence}")
    print(f"   Reasoning: {result.reasoning}")
    print(f"   Processing time: {result.processing_time:.3f}s")
    
    if result.extracted_parameters:
        print(f"   📋 Extracted parameters:")
        for param_name, param_value in result.extracted_parameters.items():
            print(f"      - {param_name}: {param_value}")
    
    if result.missing_parameters:
        print(f"   ⚠️  Missing required parameters: {result.missing_parameters}")
    else:
        print(f"   ✅ All required parameters satisfied")
    
    print(f"\n📊 LLM calls made: {llm.call_count}")


def test_explicit_parameters():
    """Test routing with explicitly provided parameters."""
    print("\n🧪 Testing Explicit Parameters")
    print("=" * 50)
    
    agents = create_default_agents()
    router = LLMAgentRouter(agents)
    llm = MockLLM()
    
    test_prompt = "Help me analyze some data"
    explicit_params = {
        "dataset_type": "customer",
        "analysis_goal": "clustering",
        "time_period": "Q1_2024"
    }
    
    print(f"Test prompt: \"{test_prompt}\"")
    print(f"Explicit parameters: {explicit_params}")
    print()
    
    result = router.route_prompt(
        test_prompt, 
        llm, 
        provided_parameters=explicit_params,
        extract_parameters=True
    )
    
    print(f"✅ Selected agent: {result.selected_agent}")
    print(f"   Confidence: {result.confidence}")
    
    if result.extracted_parameters:
        print(f"   📋 Final parameters:")
        for param_name, param_value in result.extracted_parameters.items():
            print(f"      - {param_name}: {param_value}")


def test_parameter_validation():
    """Test parameter validation."""
    print("\n🧪 Testing Parameter Validation")
    print("=" * 50)
    
    agents = create_default_agents()
    router = LLMAgentRouter(agents)
    
    # Get data_analyst agent
    data_analyst = router.get_agent_info("data_analyst")
    
    # Test valid parameters
    valid_params = {
        "dataset_type": "sales",
        "analysis_goal": "prediction"
    }
    
    errors = router.validate_parameters(valid_params, data_analyst)
    print(f"Valid parameters: {valid_params}")
    print(f"Validation errors: {errors if errors else 'None'}")
    
    # Test invalid parameters
    invalid_params = {
        "unknown_param": "value",
        "dataset_type": "sales"
    }
    
    errors = router.validate_parameters(invalid_params, data_analyst)
    print(f"\nInvalid parameters: {invalid_params}")
    print(f"Validation errors: {errors}")


def test_agent_with_required_parameters():
    """Test agent with required parameters."""
    print("\n🧪 Testing Required Parameters")
    print("=" * 50)
    
    # Create a custom agent with required parameters
    custom_agent = Agent(
        name="weather_agent",
        description="Get weather information for specific locations",
        capabilities=["Current weather", "Weather forecasts", "Weather alerts"],
        examples=["What's the weather in New York?", "Will it rain tomorrow in London?"],
        priority=2,
        parameters=[
            AgentParameter(
                name="location",
                type=ParameterType.STRING,
                required=True,
                description="The city or location to get weather for",
                examples=["New York", "London", "Tokyo"]
            ),
            AgentParameter(
                name="forecast_days",
                type=ParameterType.INTEGER,
                required=False,
                description="Number of days for forecast",
                default=1,
                examples=["1", "3", "7"]
            )
        ]
    )
    
    # Create router with custom agent
    router = LLMAgentRouter([custom_agent])
    
    # Test without required parameter
    params_missing_required = {"forecast_days": 3}
    errors = router.validate_parameters(params_missing_required, custom_agent)
    print(f"Parameters missing required: {params_missing_required}")
    print(f"Validation errors: {errors}")
    
    # Test with all required parameters
    complete_params = {"location": "San Francisco", "forecast_days": 5}
    errors = router.validate_parameters(complete_params, custom_agent)
    print(f"\nComplete parameters: {complete_params}")
    print(f"Validation errors: {errors if errors else 'None'}")


def test_parameter_type_conversion():
    """Test parameter type conversion."""
    print("\n🧪 Testing Parameter Type Conversion")
    print("=" * 50)
    
    agents = create_default_agents()
    router = LLMAgentRouter(agents)
    
    # Test various type conversions
    test_cases = [
        ("123", ParameterType.INTEGER, 123),
        ("45.67", ParameterType.FLOAT, 45.67),
        ("true", ParameterType.BOOLEAN, True),
        ("false", ParameterType.BOOLEAN, False),
        ("apple,banana,cherry", ParameterType.LIST, ["apple", "banana", "cherry"]),
        ('["x", "y", "z"]', ParameterType.LIST, ["x", "y", "z"]),
        ('{"key": "value"}', ParameterType.DICT, {"key": "value"}),
        ("hello world", ParameterType.STRING, "hello world")
    ]
    
    for value, param_type, expected in test_cases:
        try:
            result = router._convert_parameter_value(value, param_type)
            status = "✅" if result == expected else "❌"
            print(f"{status} {value} ({param_type.value}) -> {result}")
        except ValueError as e:
            print(f"❌ {value} ({param_type.value}) -> Error: {e}")


def main():
    """Run all parameter tests."""
    print("🚀 Agent Router Parameter Support Tests")
    print("=" * 60)
    
    test_parameter_extraction()
    test_explicit_parameters()
    test_parameter_validation()
    test_agent_with_required_parameters()
    test_parameter_type_conversion()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()