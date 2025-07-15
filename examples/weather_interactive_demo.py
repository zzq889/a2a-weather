#!/usr/bin/env python3
"""
Interactive demo combining agent_router parameter collection with the real weather_agent.
This shows how to use agent_router to collect parameters, then call the actual weather service.
"""

import asyncio
import uuid
import sys
import os
import time
import json
from typing import Dict, Any, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_router import LLMAgentRouter, Agent, AgentParameter, ParameterType, RoutingError

# A2A client for weather agent
try:
    import httpx
    from a2a.client import A2ACardResolver, A2AClient
    from a2a.types import (
        Role,
        Message,
        Part,
        SendMessageRequest,
        SendStreamingMessageRequest,
        TextPart,
        MessageSendParams,
        TaskStatusUpdateEvent,
        TaskArtifactUpdateEvent,
        DataPart,
    )
except ImportError:
    print("❌ A2A client not installed. This demo requires the a2a package.")
    print("Please install it to use this demo.")
    sys.exit(1)

# OpenAI SDK for OpenRouter
try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI SDK not installed. Install with: pip install openai")
    sys.exit(1)


class OpenRouterLLM:
    """OpenRouter LLM client for parameter collection and routing."""

    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def __call__(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RoutingError(f"LLM API call failed: {str(e)}", e)


class SimpleDisplay:
    """Simple terminal display with typewriter effect."""
    
    @staticmethod
    def typewriter_effect(text: str, delay: float = 0.05):
        """Display text with typewriter effect."""
        for char in text:
            print(char, end="", flush=True)
            time.sleep(delay)
        print()  # New line at the end


class WeatherAgentClient:
    """Client for interacting with the real weather agent."""
    
    def __init__(self, weather_url: str = "http://localhost:7777", debug: bool = False):
        self.weather_url = weather_url
        self._agent_card = None
        self.debug = debug
        
    async def get_weather(self, location: str, forecast_days: int = 1, units: str = "celsius") -> str:
        """
        Get weather from the actual weather agent.
        
        Args:
            location: City or location name
            forecast_days: Number of forecast days (currently not used by weather agent)
            units: Temperature units (currently not used by weather agent)
            
        Returns:
            Weather response string
        """
        try:
            async with httpx.AsyncClient() as httpx_client:
                # Get agent card if not cached
                if not self._agent_card:
                    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.weather_url)
                    self._agent_card = await resolver.get_agent_card()
                
                # Create client
                client = A2AClient(httpx_client=httpx_client, agent_card=self._agent_card)
                
                # Prepare message with location
                message = Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.user,
                    parts=[Part(root=TextPart(text=location))],
                )
                
                # Send request
                if self._agent_card.capabilities.streaming:
                    # Use streaming for real-time response
                    request = SendStreamingMessageRequest(
                        id=str(uuid.uuid4()),
                        params=MessageSendParams(message=message),
                    )
                    
                    print(f"🌤️ Getting weather for {location}...")
                    
                    response_text = ""
                    weather_data = None
                    stream_response = client.send_message_streaming(request)
                    previous_text = ""
                    
                    async for chunk in stream_response:
                        if self.debug:
                            print(f"\n[DEBUG] Chunk: {chunk}")
                        
                        # Types are already imported at the top
                        
                        # Check if this is a status update with streaming text
                        if hasattr(chunk, 'root') and hasattr(chunk.root, 'result') and isinstance(chunk.root.result, TaskStatusUpdateEvent):
                            status_update = chunk.root.result
                            if status_update.status and status_update.status.message:
                                message = status_update.status.message
                                if message.parts:
                                    for part in message.parts:
                                        if hasattr(part.root, 'text'):
                                            new_text = part.root.text
                                            if new_text != previous_text:
                                                # Show only new characters with typewriter effect
                                                new_chars = new_text[len(previous_text):]
                                                for char in new_chars:
                                                    print(char, end="", flush=True)
                                                    time.sleep(0.05)  # Typewriter delay
                                                
                                                previous_text = new_text
                                                response_text = new_text
                        
                        # Check if this is an artifact update with weather data and final text
                        elif hasattr(chunk, 'root') and hasattr(chunk.root, 'result') and hasattr(chunk.root.result, 'kind') and chunk.root.result.kind == 'artifact-update':
                            artifact_event = chunk.root.result
                            if self.debug:
                                print(f"\n[DEBUG] Processing artifact update")
                            
                            if artifact_event.artifact and artifact_event.artifact.parts:
                                # Extract text and data from artifact parts
                                for part in artifact_event.artifact.parts:
                                    if hasattr(part, 'root'):
                                        # Get final complete text: 'Weather in uk: 🌞 Sunny, 25 °C'
                                        if hasattr(part.root, 'text'):
                                            final_text = part.root.text
                                            if self.debug:
                                                print(f"[DEBUG] Found final text: '{final_text}'")
                                            # Complete any remaining characters from streaming
                                            if len(final_text) > len(response_text):
                                                remaining_chars = final_text[len(response_text):]
                                                for char in remaining_chars:
                                                    print(char, end="", flush=True)
                                                    time.sleep(0.05)
                                            response_text = final_text
                                        
                                        # Get weather data: {'type': 'widget', 'payload': {...}}
                                        elif hasattr(part.root, 'data'):
                                            weather_data = part.root.data
                                            if self.debug:
                                                print(f"[DEBUG] Found weather data: {weather_data}")
                        
                        # Debug: catch anything else
                        else:
                            if self.debug and hasattr(chunk, 'root') and hasattr(chunk.root, 'result'):
                                print(f"[DEBUG] ❌ Unhandled chunk type: {type(chunk.root.result)} with kind: {getattr(chunk.root.result, 'kind', 'no kind')}")
                    
                    print()  # New line after streaming
                    
                    # Show structured weather data if available
                    if weather_data and isinstance(weather_data, dict) and "payload" in weather_data:
                        payload = weather_data["payload"]
                        print("\n📊 Weather Details:")
                        if "temperature" in payload:
                            print(f"🌡️  Temperature: {payload['temperature']}{payload.get('unit', '')}")
                        if "condition" in payload:
                            print(f"☀️  Condition: {payload['condition']}")
                        if "icon" in payload:
                            print(f"🎨 Icon: {payload['icon']}")
                    
                    # Debug: Show what we actually got
                    if self.debug:
                        print(f"\n[DEBUG] Final response_text: '{response_text}'")
                        print(f"[DEBUG] Final weather_data: {weather_data}")
                    
                    return response_text or f"Weather data received for {location}"
                
                else:
                    # Non-streaming request
                    request = SendMessageRequest(
                        id=str(uuid.uuid4()),
                        params=MessageSendParams(message=message),
                    )
                    response = await client.send_message(request)
                    return f"Weather response: {response}"
                    
        except Exception as e:
            return f"❌ Error getting weather: {str(e)}"


class WeatherChatSession:
    """Chat session that combines parameter collection with weather agent execution."""
    
    def __init__(self, router: LLMAgentRouter, llm_function, weather_client: WeatherAgentClient):
        self.router = router
        self.llm_function = llm_function
        self.weather_client = weather_client
        
        # Session state for parameter collection
        self.collecting_for_agent = None
        self.partial_parameters = {}
        self.missing_parameters = []
    
    async def handle_message(self, user_message: str) -> str:
        """Handle user message with parameter collection and weather execution."""
        
        # If we're not currently collecting parameters, start fresh
        if not self.collecting_for_agent:
            return await self._start_new_request(user_message)
        
        # If we are collecting parameters, process the additional input
        else:
            return await self._continue_parameter_collection(user_message)
    
    async def _start_new_request(self, user_message: str) -> str:
        """Handle a new user request - route and check for missing parameters."""
        
        # Step 1: Route the request
        result = self.router.route_prompt(user_message, self.llm_function)
        
        print(f"[DEBUG] Routed to: {result.selected_agent}")
        print(f"[DEBUG] Extracted: {result.extracted_parameters}")
        print(f"[DEBUG] Missing: {result.missing_parameters}")
        
        # Only handle weather agent for this demo
        if result.selected_agent != "weather_agent":
            return f"This demo only supports weather requests. Try asking about weather!"
        
        # Step 2: Check if we have everything we need
        if not result.missing_parameters:
            # We have everything! Execute the weather agent
            return await self._execute_weather_agent(result.extracted_parameters)
        
        # Step 3: We need to collect missing parameters
        self.collecting_for_agent = result.selected_agent
        self.partial_parameters = result.extracted_parameters.copy() if result.extracted_parameters else {}
        self.missing_parameters = result.missing_parameters.copy()
        
        # Ask for the first missing parameter
        return self._ask_for_next_parameter()
    
    async def _continue_parameter_collection(self, user_message: str) -> str:
        """Continue collecting parameters from user input."""
        
        if not self.missing_parameters:
            # This shouldn't happen, but handle gracefully
            return await self._complete_request()
        
        # Get the parameter we're currently collecting
        current_param_name = self.missing_parameters[0]
        agent = self.router.get_agent_info(self.collecting_for_agent)
        param_def = next(p for p in agent.parameters if p.name == current_param_name)
        
        # Try to extract the parameter value from user's response
        try:
            param_value = self.router._convert_parameter_value(user_message.strip(), param_def.type)
            self.partial_parameters[current_param_name] = param_value
            self.missing_parameters.pop(0)  # Remove this parameter from missing list
            
            print(f"[DEBUG] Collected {current_param_name}: {param_value}")
            
        except ValueError as e:
            return f"Sorry, that's not a valid {param_def.type.value}. {param_def.description}. Please try again."
        
        # Check if we still need more parameters
        if self.missing_parameters:
            return self._ask_for_next_parameter()
        else:
            return await self._complete_request()
    
    def _ask_for_next_parameter(self) -> str:
        """Ask user for the next missing parameter."""
        
        if not self.missing_parameters:
            return "All parameters collected!"
        
        param_name = self.missing_parameters[0]
        agent = self.router.get_agent_info(self.collecting_for_agent)
        param_def = next(p for p in agent.parameters if p.name == param_name)
        
        # Create a friendly prompt
        prompt = f"I need to know the {param_name.replace('_', ' ')}. {param_def.description}"
        
        if param_def.examples:
            prompt += f" (e.g., {', '.join(param_def.examples[:2])})"
        
        return prompt
    
    async def _complete_request(self) -> str:
        """Complete the request with all parameters collected."""
        
        # Final validation
        agent = self.router.get_agent_info(self.collecting_for_agent)
        validation_errors = self.router.validate_parameters(self.partial_parameters, agent)
        
        if validation_errors:
            # Reset state and report error
            self._reset_state()
            return f"Sorry, there was an error: {'; '.join(validation_errors)}"
        
        # Execute the agent
        parameters = self.partial_parameters.copy()
        
        # Reset state for next request
        self._reset_state()
        
        return await self._execute_weather_agent(parameters)
    
    async def _execute_weather_agent(self, parameters: Dict[str, Any]) -> str:
        """Execute the real weather agent with the collected parameters."""
        
        location = parameters.get('location', 'unknown')
        forecast_days = parameters.get('forecast_days', 1)
        units = parameters.get('units', 'celsius')
        
        print(f"[DEBUG] Executing weather agent: location={location}, days={forecast_days}, units={units}")
        
        # Call the real weather agent
        try:
            weather_result = await self.weather_client.get_weather(location, forecast_days, units)
            return weather_result
        except Exception as e:
            return f"❌ Error getting weather: {str(e)}"
    
    def _reset_state(self):
        """Reset the parameter collection state."""
        self.collecting_for_agent = None
        self.partial_parameters = {}
        self.missing_parameters = []


def create_weather_agent_definition():
    """Create the weather agent definition for routing."""
    
    return Agent(
        name="weather_agent",
        description="Get current weather and forecasts for any location",
        capabilities=[
            "Current weather conditions",
            "Weather forecasts",
            "Temperature information",
            "Weather data visualization"
        ],
        examples=[
            "What's the weather in New York?",
            "How's the weather today?",
            "Give me the weather forecast for London",
            "Will it rain in Tokyo tomorrow?"
        ],
        priority=3,
        parameters=[
            AgentParameter(
                name="location",
                type=ParameterType.STRING,
                required=True,
                description="The city, state, or country to get weather for",
                examples=["New York", "London", "Paris", "Tokyo", "San Francisco"]
            ),
            AgentParameter(
                name="forecast_days",
                type=ParameterType.INTEGER,
                required=False,
                description="Number of days for forecast (1-7)",
                default=1,
                examples=["1", "3", "7"]
            ),
            AgentParameter(
                name="units",
                type=ParameterType.STRING,
                required=False,
                description="Temperature units (celsius, fahrenheit)",
                default="celsius",
                examples=["celsius", "fahrenheit"]
            )
        ]
    )


def get_api_key() -> str:
    """Get OpenRouter API key from environment."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OpenRouter API key not found!")
        print("\n🔧 Setup Instructions:")
        print("1. Get your API key from: https://openrouter.ai/keys")
        print("2. Set environment variable:")
        print("   export OPENROUTER_API_KEY='your-key-here'")
        print("3. Or create a .env file:")
        print("   echo 'OPENROUTER_API_KEY=your-key-here' > .env")
        sys.exit(1)
    
    return api_key


def select_model() -> str:
    """Let user select a model."""
    default_model = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-flash")
    
    models = {
        "1": ("google/gemini-2.5-flash", "Gemini 2.5 Flash (fast, cheap)"),
        "2": ("openai/gpt-4o-mini", "GPT-4o Mini (balanced)"),
        "3": ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet (smart)")
    }

    print("🤖 Available Models:")
    for key, (model_id, name) in models.items():
        default_marker = " ⭐ (Default)" if model_id == default_model else ""
        print(f"{key}. {name}{default_marker}")
    print()

    choice = input("Select model (1-3) or Enter for default: ").strip()
    
    if not choice:
        print(f"✅ Using default: {default_model}")
        return default_model
    
    if choice in models:
        selected_model = models[choice][0]
        print(f"✅ Selected: {models[choice][1]}")
        return selected_model
    
    print("⚠️ Invalid choice, using default")
    return default_model


async def check_weather_agent_available(weather_url: str = "http://localhost:7777") -> bool:
    """Check if the weather agent is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{weather_url}/.well-known/agent.json")
            return response.status_code == 200
    except:
        return False


async def interactive_weather_demo():
    """Main interactive demo."""
    
    print("🌤️ Interactive Weather Agent Demo")
    print("=" * 60)
    print("This demo combines agent_router parameter collection")
    print("with the real weather_agent service.")
    print("=" * 60)
    
    # Check if weather agent is running
    print("🔌 Checking weather agent...")
    weather_available = await check_weather_agent_available()
    
    if not weather_available:
        print("❌ Weather agent not available at http://localhost:7777")
        print("\n🚀 To start the weather agent:")
        print("1. Open a new terminal")
        print("2. cd to this project directory")
        print("3. Run: python -m weather_agent")
        print("4. Wait for 'Uvicorn running on http://0.0.0.0:7777'")
        print("5. Then run this demo again")
        return
    
    print("✅ Weather agent is running!")
    
    # Setup API
    api_key = get_api_key()
    model = select_model()
    print()
    
    # Initialize components
    print("🔧 Initializing...")
    weather_agent = create_weather_agent_definition()
    router = LLMAgentRouter([weather_agent])
    llm = OpenRouterLLM(api_key, model)
    weather_client = WeatherAgentClient(debug=False)
    session = WeatherChatSession(router, llm, weather_client)
    
    print(f"✅ Connected to {model}")
    print("✅ Weather agent definition loaded")
    print()
    
    # Test LLM connection
    try:
        print("🔌 Testing LLM connection...")
        test_response = llm("Say 'Ready for weather!'", max_tokens=50)
        print(f"✅ LLM test: {test_response}")
    except Exception as e:
        print(f"❌ LLM connection failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🎮 Interactive Weather Chat")
    print("Ask about weather! The system will collect missing parameters.")
    print("Commands: 'quit' to exit, 'reset' to start over")
    print("=" * 60)
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() in ['reset', 'new']:
                session._reset_state()
                print("🔄 Starting new conversation")
                continue
            
            if not user_input:
                continue
            
            print("🤔 Thinking...")
            response = await session.handle_message(user_input)
            print(f"🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Resetting session...")
            session._reset_state()


async def demo_test_cases():
    """Run automated test cases."""
    
    print("🧪 Automated Test Cases")
    print("=" * 60)
    
    # Check weather agent
    weather_available = await check_weather_agent_available()
    if not weather_available:
        print("❌ Weather agent not available. Please start it first.")
        return
    
    # Setup
    api_key = get_api_key()
    model = select_model()
    
    weather_agent = create_weather_agent_definition()
    router = LLMAgentRouter([weather_agent])
    llm = OpenRouterLLM(api_key, model)
    weather_client = WeatherAgentClient(debug=False)
    
    test_cases = [
        "What's the weather in London?",
        "How's the weather?",
        "Give me the weather for Tokyo for 3 days",
        "Weather forecast please"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case}")
        print("-" * 40)
        
        try:
            session = WeatherChatSession(router, llm, weather_client)
            response = await session.handle_message(test_case)
            print(f"Response: {response}")
            
            # Show if more input is needed
            if session.collecting_for_agent:
                print(f"💬 Waiting for: {session.missing_parameters}")
        
        except Exception as e:
            print(f"❌ Test failed: {e}")


def main():
    """Main function with mode selection."""
    print("🚀 Weather Agent + Parameter Collection Demo")
    print("=" * 60)
    
    print("Select mode:")
    print("1. 🎮 Interactive Demo (recommended)")
    print("2. 🧪 Automated Tests") 
    print()
    
    while True:
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            asyncio.run(interactive_weather_demo())
            break
        elif choice == "2":
            asyncio.run(demo_test_cases())
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()