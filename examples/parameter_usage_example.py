#!/usr/bin/env python3
"""
Example demonstrating parameter support in agent_router.
This shows how to create agents with parameters and use them effectively.
"""

import sys
import os

# Add the parent directory to the path so we can import agent_router
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_router import (
    LLMAgentRouter, 
    Agent, 
    AgentParameter, 
    ParameterType,
    RoutingConfidence
)


def create_agents_with_parameters():
    """Create example agents that demonstrate parameter support."""
    
    # Weather agent with required location parameter
    weather_agent = Agent(
        name="weather_agent",
        description="Get weather information and forecasts for specific locations",
        capabilities=[
            "Current weather conditions",
            "Weather forecasts up to 14 days",
            "Weather alerts and warnings",
            "Historical weather data"
        ],
        examples=[
            "What's the weather in New York?",
            "Will it rain tomorrow in London?",
            "Show me the 7-day forecast for Tokyo",
            "Are there any weather alerts for Miami?"
        ],
        priority=3,
        parameters=[
            AgentParameter(
                name="location",
                type=ParameterType.STRING,
                required=True,
                description="The city, state, or country to get weather for",
                examples=["New York, NY", "London, UK", "Tokyo, Japan", "Sydney, Australia"]
            ),
            AgentParameter(
                name="forecast_days",
                type=ParameterType.INTEGER,
                required=False,
                description="Number of days for forecast (1-14)",
                default=1,
                examples=["1", "3", "7", "14"]
            ),
            AgentParameter(
                name="include_alerts",
                type=ParameterType.BOOLEAN,
                required=False,
                description="Whether to include weather alerts",
                default=False,
                examples=["true", "false"]
            ),
            AgentParameter(
                name="units",
                type=ParameterType.STRING,
                required=False,
                description="Temperature units (celsius, fahrenheit, kelvin)",
                default="celsius",
                examples=["celsius", "fahrenheit", "kelvin"]
            )
        ]
    )
    
    # Code generator agent with various parameter types
    code_generator = Agent(
        name="code_generator",
        description="Generate code snippets, functions, and complete programs",
        capabilities=[
            "Generate functions in multiple languages",
            "Create data structures and algorithms",
            "Write unit tests",
            "Generate documentation",
            "Code optimization and refactoring"
        ],
        examples=[
            "Write a Python function to sort a list",
            "Create a REST API endpoint in Node.js",
            "Generate a React component for a login form",
            "Write unit tests for this function"
        ],
        priority=3,
        parameters=[
            AgentParameter(
                name="language",
                type=ParameterType.STRING,
                required=True,
                description="Programming language to use",
                examples=["python", "javascript", "java", "go", "rust", "typescript"]
            ),
            AgentParameter(
                name="function_name",
                type=ParameterType.STRING,
                required=False,
                description="Name of the function to generate",
                examples=["calculate_sum", "fetch_user_data", "validate_email"]
            ),
            AgentParameter(
                name="include_tests",
                type=ParameterType.BOOLEAN,
                required=False,
                description="Whether to include unit tests",
                default=False,
                examples=["true", "false"]
            ),
            AgentParameter(
                name="complexity",
                type=ParameterType.STRING,
                required=False,
                description="Code complexity level",
                default="medium",
                examples=["simple", "medium", "complex"]
            ),
            AgentParameter(
                name="dependencies",
                type=ParameterType.LIST,
                required=False,
                description="External libraries or frameworks to use",
                examples=["requests,pandas", "express,mongoose", "react,axios"]
            )
        ]
    )
    
    # Data processor agent with flexible parameters
    data_processor = Agent(
        name="data_processor",
        description="Process, transform, and analyze various data formats",
        capabilities=[
            "CSV and JSON data processing",
            "Data cleaning and validation",
            "Statistical analysis",
            "Data format conversion",
            "Batch processing operations"
        ],
        examples=[
            "Clean this CSV data and remove duplicates",
            "Convert JSON to CSV format",
            "Calculate statistics for this dataset",
            "Validate email addresses in this data"
        ],
        priority=2,
        parameters=[
            AgentParameter(
                name="input_format",
                type=ParameterType.STRING,
                required=True,
                description="Format of the input data",
                examples=["csv", "json", "xml", "excel", "txt"]
            ),
            AgentParameter(
                name="output_format",
                type=ParameterType.STRING,
                required=False,
                description="Desired output format",
                default="json",
                examples=["csv", "json", "xml", "excel"]
            ),
            AgentParameter(
                name="operations",
                type=ParameterType.LIST,
                required=False,
                description="Processing operations to perform",
                examples=["clean", "validate", "transform", "aggregate"]
            ),
            AgentParameter(
                name="config",
                type=ParameterType.DICT,
                required=False,
                description="Configuration options for processing",
                examples=['{"delimiter": ",", "encoding": "utf-8"}', '{"remove_nulls": true}']
            )
        ]
    )
    
    return [weather_agent, code_generator, data_processor]


class ExampleLLM:
    """Example LLM that provides realistic responses for demonstration."""
    
    def __call__(self, prompt: str) -> str:
        # Handle routing requests
        if "routing system" in prompt.lower():
            # Extract the user prompt from the routing prompt
            user_prompt_start = prompt.find('User Prompt: "') + len('User Prompt: "')
            user_prompt_end = prompt.find('"', user_prompt_start)
            user_prompt = prompt[user_prompt_start:user_prompt_end].lower() if user_prompt_start > 13 else ""
            if "weather" in user_prompt or "forecast" in user_prompt:
                return '''
                {
                    "selected_agent": "weather_agent",
                    "confidence": "high",
                    "reasoning": "User is asking for weather information which matches the weather agent's capabilities",
                    "fallback_agents": []
                }
                '''
            elif "code" in user_prompt or "function" in user_prompt or "python" in user_prompt or "fibonacci" in user_prompt:
                return '''
                {
                    "selected_agent": "code_generator",
                    "confidence": "high", 
                    "reasoning": "User is requesting code generation which is the specialty of the code generator agent",
                    "fallback_agents": []
                }
                '''
            elif "data" in user_prompt or "csv" in user_prompt or "clean" in user_prompt or "process" in user_prompt:
                return '''
                {
                    "selected_agent": "data_processor",
                    "confidence": "medium",
                    "reasoning": "User mentions data processing which matches the data processor agent",
                    "fallback_agents": ["code_generator"]
                }
                '''
            else:
                # Default routing based on first available agent
                return '''
                {
                    "selected_agent": "weather_agent",
                    "confidence": "low",
                    "reasoning": "No specific match found, using fallback agent",
                    "fallback_agents": []
                }
                '''
        
        # Handle parameter extraction requests
        elif "parameter extraction" in prompt.lower():
            if "weather_agent" in prompt:
                return '''
                {
                    "extracted_parameters": {
                        "location": "New York",
                        "forecast_days": "7",
                        "include_alerts": "true"
                    },
                    "missing_required": [],
                    "confidence": "high",
                    "reasoning": "Successfully extracted location as New York, forecast days as 7, and alerts flag as true"
                }
                '''
            elif "code_generator" in prompt:
                return '''
                {
                    "extracted_parameters": {
                        "language": "python",
                        "function_name": "fibonacci",
                        "include_tests": "true"
                    },
                    "missing_required": [],
                    "confidence": "high",
                    "reasoning": "Identified Python as language, fibonacci as function name, and tests requested"
                }
                '''
            elif "data_processor" in prompt:
                return '''
                {
                    "extracted_parameters": {
                        "input_format": "csv",
                        "operations": "clean,validate"
                    },
                    "missing_required": [],
                    "confidence": "medium",
                    "reasoning": "Found CSV as input format and cleaning/validation operations mentioned"
                }
                '''
        
        return "Example LLM response"


def demonstrate_weather_agent():
    """Demonstrate weather agent with parameters."""
    print("🌤️  Weather Agent Example")
    print("=" * 40)
    
    agents = create_agents_with_parameters()
    router = LLMAgentRouter(agents)
    llm = ExampleLLM()
    
    test_prompt = "What's the 7-day weather forecast for New York with alerts?"
    
    print(f"Prompt: \"{test_prompt}\"")
    result = router.route_prompt(test_prompt, llm)
    
    print(f"✅ Agent: {result.selected_agent}")
    print(f"📋 Parameters: {result.extracted_parameters}")
    
    # Demonstrate explicit parameter override
    print(f"\n🔧 With explicit parameters:")
    explicit_params = {
        "location": "London, UK", 
        "forecast_days": 3,
        "units": "fahrenheit"
    }
    
    result2 = router.route_prompt(
        "Get weather info", 
        llm, 
        provided_parameters=explicit_params
    )
    print(f"📋 Final parameters: {result2.extracted_parameters}")


def demonstrate_code_generator():
    """Demonstrate code generator with complex parameters."""
    print("\n💻 Code Generator Example")
    print("=" * 40)
    
    agents = create_agents_with_parameters()
    router = LLMAgentRouter(agents)
    llm = ExampleLLM()
    
    test_prompt = "Write a Python fibonacci function with unit tests"
    
    print(f"Prompt: \"{test_prompt}\"")
    result = router.route_prompt(test_prompt, llm)
    
    print(f"✅ Agent: {result.selected_agent}")
    print(f"📋 Parameters: {result.extracted_parameters}")
    
    # Test parameter validation
    print(f"\n🔍 Parameter validation:")
    agent = router.get_agent_info("code_generator")
    
    # Valid parameters
    valid_params = {
        "language": "javascript",
        "function_name": "calculate_sum",
        "include_tests": True,
        "dependencies": ["lodash", "moment"]
    }
    errors = router.validate_parameters(valid_params, agent)
    print(f"Valid params: {valid_params}")
    print(f"Errors: {errors if errors else 'None'}")
    
    # Invalid parameters
    invalid_params = {
        "language": "python",
        "invalid_param": "value"
    }
    errors = router.validate_parameters(invalid_params, agent)
    print(f"\nInvalid params: {invalid_params}")
    print(f"Errors: {errors}")


def demonstrate_data_processor():
    """Demonstrate data processor with dict parameters."""
    print("\n📊 Data Processor Example")
    print("=" * 40)
    
    agents = create_agents_with_parameters()
    router = LLMAgentRouter(agents)
    llm = ExampleLLM()
    
    test_prompt = "Clean and validate this CSV data"
    
    print(f"Prompt: \"{test_prompt}\"")
    result = router.route_prompt(test_prompt, llm)
    
    print(f"✅ Agent: {result.selected_agent}")
    print(f"📋 Parameters: {result.extracted_parameters}")
    
    # Demonstrate complex parameter types
    print(f"\n🛠️  Complex parameter example:")
    complex_params = {
        "input_format": "csv",
        "output_format": "json",
        "operations": ["clean", "validate", "transform"],
        "config": {
            "delimiter": ",",
            "encoding": "utf-8",
            "remove_nulls": True,
            "validation_rules": ["email", "phone"]
        }
    }
    
    result2 = router.route_prompt(
        "Process my data file",
        llm,
        provided_parameters=complex_params
    )
    print(f"📋 Complex parameters: {result2.extracted_parameters}")


def main():
    """Run parameter usage examples."""
    print("🚀 Agent Router Parameter Usage Examples")
    print("=" * 60)
    
    # Show available agents and their parameters
    agents = create_agents_with_parameters()
    print("📋 Available Agents:")
    for agent in agents:
        print(f"\n🤖 {agent.name}")
        print(f"   Description: {agent.description}")
        if agent.parameters:
            print(f"   Parameters:")
            for param in agent.parameters:
                required_str = " (required)" if param.required else f" (default: {param.default})"
                print(f"      - {param.name} ({param.type.value}){required_str}: {param.description}")
    
    print("\n" + "=" * 60)
    
    # Run demonstrations
    demonstrate_weather_agent()
    demonstrate_code_generator()
    demonstrate_data_processor()
    
    print(f"\n✅ Parameter usage examples completed!")


if __name__ == "__main__":
    main()