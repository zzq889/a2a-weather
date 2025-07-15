#!/usr/bin/env python3
"""
Test what happens when user prompts don't contain required parameters.
"""

import sys
import os

# Add the parent directory to the path so we can import agent_router
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_router import LLMAgentRouter, Agent, AgentParameter, ParameterType


def create_weather_agent():
    """Create weather agent with required location parameter."""
    return Agent(
        name="weather_agent",
        description="Get weather information for specific locations",
        capabilities=["Current weather", "Weather forecasts", "Weather alerts"],
        examples=["What's the weather in New York?", "Will it rain tomorrow in London?"],
        priority=3,
        parameters=[
            AgentParameter(
                name="location",
                type=ParameterType.STRING,
                required=True,  # This is required!
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


class TestLLM:
    """Mock LLM that simulates realistic parameter extraction."""
    
    def __call__(self, prompt: str) -> str:
        # Handle routing requests
        if "routing system" in prompt.lower():
            # Extract the user prompt from the routing prompt
            user_prompt_start = prompt.find('User Prompt: "') + len('User Prompt: "')
            user_prompt_end = prompt.find('"', user_prompt_start)
            user_prompt = prompt[user_prompt_start:user_prompt_end].lower() if user_prompt_start > 13 else ""
            
            if "weather" in user_prompt:
                return '''
                {
                    "selected_agent": "weather_agent",
                    "confidence": "high",
                    "reasoning": "User is asking for weather information",
                    "fallback_agents": []
                }
                '''
        
        # Handle parameter extraction requests
        elif "parameter extraction" in prompt.lower():
            user_prompt_start = prompt.find('User Prompt: "') + len('User Prompt: "')
            user_prompt_end = prompt.find('"', user_prompt_start)
            user_prompt = prompt[user_prompt_start:user_prompt_end] if user_prompt_start > 13 else ""
            
            # Simulate different scenarios based on the prompt
            if "how's the weather" in user_prompt.lower() and "new york" not in user_prompt.lower():
                # No location mentioned - can't extract required parameter
                return '''
                {
                    "extracted_parameters": {},
                    "missing_required": ["location"],
                    "confidence": "low",
                    "reasoning": "The user asked about weather but did not specify a location, which is required"
                }
                '''
            elif "weather in london" in user_prompt.lower():
                # Location is mentioned
                return '''
                {
                    "extracted_parameters": {
                        "location": "London"
                    },
                    "missing_required": [],
                    "confidence": "high",
                    "reasoning": "Successfully extracted location as London from the user prompt"
                }
                '''
            elif "3-day forecast" in user_prompt.lower():
                # Forecast mentioned but no location
                return '''
                {
                    "extracted_parameters": {
                        "forecast_days": "3"
                    },
                    "missing_required": ["location"],
                    "confidence": "medium",
                    "reasoning": "Found forecast duration but no location specified"
                }
                '''
        
        return "Mock LLM response"


def test_missing_required_parameters():
    """Test various scenarios with missing required parameters."""
    print("🧪 Testing Missing Required Parameters")
    print("=" * 60)
    
    weather_agent = create_weather_agent()
    router = LLMAgentRouter([weather_agent])
    llm = TestLLM()
    
    test_cases = [
        {
            "prompt": "How's the weather?",
            "description": "Generic weather question - no location"
        },
        {
            "prompt": "What's the weather in London?",
            "description": "Specific location mentioned"
        },
        {
            "prompt": "Give me a 3-day forecast",
            "description": "Forecast request without location"
        },
        {
            "prompt": "Tell me about the weather",
            "description": "Very vague weather question"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print(f"   Prompt: \"{test_case['prompt']}\"")
        
        try:
            result = router.route_prompt(test_case['prompt'], llm)
            
            print(f"   ✅ Routed to: {result.selected_agent}")
            print(f"   🎯 Confidence: {result.confidence}")
            
            if result.extracted_parameters:
                print(f"   📋 Extracted: {result.extracted_parameters}")
            else:
                print(f"   📋 Extracted: None")
            
            if result.missing_parameters:
                print(f"   ⚠️  Missing required: {result.missing_parameters}")
                print(f"   💡 Status: INCOMPLETE - Missing required parameters")
            else:
                print(f"   ✅ Status: COMPLETE - All required parameters available")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("-" * 50)


def test_parameter_validation_scenarios():
    """Test parameter validation with different scenarios."""
    print("\n🔍 Testing Parameter Validation Scenarios")
    print("=" * 60)
    
    weather_agent = create_weather_agent()
    router = LLMAgentRouter([weather_agent])
    
    validation_cases = [
        {
            "name": "Complete parameters",
            "params": {"location": "Paris", "forecast_days": 5},
            "expected": "Valid"
        },
        {
            "name": "Missing required parameter",
            "params": {"forecast_days": 3},
            "expected": "Invalid - missing location"
        },
        {
            "name": "Only required parameter",
            "params": {"location": "Tokyo"},
            "expected": "Valid - optional parameters get defaults"
        },
        {
            "name": "Unknown parameter",
            "params": {"location": "Berlin", "temperature_unit": "celsius"},
            "expected": "Invalid - unknown parameter"
        },
        {
            "name": "Empty parameters",
            "params": {},
            "expected": "Invalid - missing required"
        }
    ]
    
    for case in validation_cases:
        print(f"\n📋 {case['name']}")
        print(f"   Parameters: {case['params']}")
        print(f"   Expected: {case['expected']}")
        
        errors = router.validate_parameters(case['params'], weather_agent)
        
        if errors:
            print(f"   ❌ Validation errors:")
            for error in errors:
                print(f"      - {error}")
        else:
            print(f"   ✅ Validation passed")


def test_handling_strategies():
    """Show different strategies for handling missing parameters."""
    print("\n🛠️  Handling Strategies for Missing Parameters")
    print("=" * 60)
    
    weather_agent = create_weather_agent()
    router = LLMAgentRouter([weather_agent])
    llm = TestLLM()
    
    prompt = "How's the weather?"
    result = router.route_prompt(prompt, llm)
    
    print(f"User prompt: \"{prompt}\"")
    print(f"Selected agent: {result.selected_agent}")
    print(f"Missing parameters: {result.missing_parameters}")
    
    print(f"\n💡 Possible handling strategies:")
    
    if result.missing_parameters:
        print(f"   1. 🔄 PROMPT USER: Ask user to provide missing parameters")
        print(f"      \"I need to know the location. Which city's weather would you like?\"")
        
        print(f"   2. 🎯 DEFAULT FALLBACK: Use a default location")
        print(f"      \"Using default location: Current user location\"")
        
        print(f"   3. 🔀 AGENT FALLBACK: Route to a more general agent")
        print(f"      \"Routing to general_assistant for generic weather info\"")
        
        print(f"   4. ❌ GRACEFUL REJECTION: Explain what's needed")
        print(f"      \"I can help with weather, but I need a specific location.\"")
        
        print(f"   5. 🤖 INTERACTIVE MODE: Start a conversation to collect parameters")
        print(f"      \"Let me help you get weather information. What location?\"")


def demonstrate_parameter_completion():
    """Show how to handle incomplete parameters in practice."""
    print("\n🔧 Parameter Completion Example")
    print("=" * 60)
    
    weather_agent = create_weather_agent()
    router = LLMAgentRouter([weather_agent])
    llm = TestLLM()
    
    # Simulate a conversation flow
    print("👤 User: \"How's the weather?\"")
    result1 = router.route_prompt("How's the weather?", llm)
    
    if result1.missing_parameters:
        print(f"🤖 System: Missing required parameters: {result1.missing_parameters}")
        print("🤖 System: \"I can help with weather! Which location would you like?\"")
        print("👤 User: \"New York\"")
        
        # Complete the parameters with user input
        completed_params = result1.extracted_parameters.copy() if result1.extracted_parameters else {}
        completed_params["location"] = "New York"
        
        result2 = router.route_prompt(
            "How's the weather?", 
            llm, 
            provided_parameters=completed_params
        )
        
        print(f"🤖 System: Parameters completed: {result2.extracted_parameters}")
        print("🤖 System: \"Getting weather for New York...\"")
        
        # Validate final parameters
        errors = router.validate_parameters(result2.extracted_parameters, weather_agent)
        if not errors:
            print("✅ Ready to execute weather query!")
        else:
            print(f"❌ Still have issues: {errors}")


def main():
    """Run all missing parameter tests."""
    print("🚀 Testing Missing Required Parameters")
    print("=" * 80)
    
    test_missing_required_parameters()
    test_parameter_validation_scenarios()
    test_handling_strategies()
    demonstrate_parameter_completion()
    
    print(f"\n✅ Missing parameter testing completed!")


if __name__ == "__main__":
    main()