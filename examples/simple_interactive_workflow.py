#!/usr/bin/env python3
"""
Simple example showing how to use agent_router for interactive parameter collection
with real OpenRouter API integration.
"""

import sys
import os
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

# OpenAI SDK for OpenRouter
try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI SDK not installed. Install with: pip install openai")
    sys.exit(1)


class ChatSession:
    """Simulates a chat session with persistent state."""
    
    def __init__(self, router: LLMAgentRouter, llm_function):
        self.router = router
        self.llm_function = llm_function
        
        # Session state for parameter collection
        self.collecting_for_agent = None
        self.partial_parameters = {}
        self.missing_parameters = []
    
    def handle_message(self, user_message: str) -> str:
        """
        Main handler - this is what you'd call in your application.
        
        Returns:
            Response message to send back to user
        """
        
        # If we're not currently collecting parameters, start fresh
        if not self.collecting_for_agent:
            return self._start_new_request(user_message)
        
        # If we are collecting parameters, process the additional input
        else:
            return self._continue_parameter_collection(user_message)
    
    def _start_new_request(self, user_message: str) -> str:
        """Handle a new user request - route and check for missing parameters."""
        
        # Step 1: Route the request
        result = self.router.route_prompt(user_message, self.llm_function)
        
        print(f"[DEBUG] Routed to: {result.selected_agent}")
        print(f"[DEBUG] Extracted: {result.extracted_parameters}")
        print(f"[DEBUG] Missing: {result.missing_parameters}")
        
        # Step 2: Check if we have everything we need
        if not result.missing_parameters:
            # We have everything! Execute the agent
            return self._execute_agent(result.selected_agent, result.extracted_parameters)
        
        # Step 3: We need to collect missing parameters
        self.collecting_for_agent = result.selected_agent
        self.partial_parameters = result.extracted_parameters.copy() if result.extracted_parameters else {}
        self.missing_parameters = result.missing_parameters.copy()
        
        # Ask for the first missing parameter
        return self._ask_for_next_parameter()
    
    def _continue_parameter_collection(self, user_message: str) -> str:
        """Continue collecting parameters from user input."""
        
        if not self.missing_parameters:
            # This shouldn't happen, but handle gracefully
            return self._complete_request()
        
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
            return self._complete_request()
    
    def _ask_for_next_parameter(self) -> str:
        """Ask user for the next missing parameter."""
        
        if not self.missing_parameters:
            return self._complete_request()
        
        param_name = self.missing_parameters[0]
        agent = self.router.get_agent_info(self.collecting_for_agent)
        param_def = next(p for p in agent.parameters if p.name == param_name)
        
        # Create a friendly prompt
        prompt = f"I need to know the {param_name.replace('_', ' ')}. {param_def.description}"
        
        if param_def.examples:
            prompt += f" (e.g., {', '.join(param_def.examples[:2])})"
        
        return prompt
    
    def _complete_request(self) -> str:
        """Complete the request with all parameters collected."""
        
        # Final validation
        agent = self.router.get_agent_info(self.collecting_for_agent)
        validation_errors = self.router.validate_parameters(self.partial_parameters, agent)
        
        if validation_errors:
            # Reset state and report error
            self._reset_state()
            return f"Sorry, there was an error: {'; '.join(validation_errors)}"
        
        # Execute the agent
        agent_name = self.collecting_for_agent
        parameters = self.partial_parameters.copy()
        
        # Reset state for next request
        self._reset_state()
        
        return self._execute_agent(agent_name, parameters)
    
    def _execute_agent(self, agent_name: str, parameters: Dict[str, Any]) -> str:
        """Execute the agent with the collected parameters."""
        
        # In a real application, this would call your actual agent execution logic
        print(f"[DEBUG] Executing {agent_name} with {parameters}")
        
        if agent_name == "weather_agent":
            location = parameters.get('location', 'unknown')
            days = parameters.get('forecast_days', 1)
            units = parameters.get('units', 'celsius')
            return f"🌤️ Here's the {days}-day weather forecast for {location} in {units}..."
        
        elif agent_name == "booking_agent":
            service = parameters.get('service_type', 'service')
            destination = parameters.get('destination', 'destination')
            date = parameters.get('checkin_date', 'date')
            return f"✈️ Booking {service} to {destination} for {date}..."
        
        else:
            return f"✅ Executed {agent_name} with parameters: {parameters}"
    
    def _reset_state(self):
        """Reset the parameter collection state."""
        self.collecting_for_agent = None
        self.partial_parameters = {}
        self.missing_parameters = []


def create_simple_agents():
    """Create a few simple agents for demonstration."""
    
    weather_agent = Agent(
        name="weather_agent",
        description="Get weather information",
        capabilities=["Weather forecasts"],
        examples=["What's the weather?"],
        priority=3,
        parameters=[
            AgentParameter(
                name="location",
                type=ParameterType.STRING,
                required=True,
                description="The city to get weather for",
                examples=["New York", "London"]
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
    
    booking_agent = Agent(
        name="booking_agent",
        description="Book travel services",
        capabilities=["Flight booking", "Hotel booking"],
        examples=["Book a flight"],
        priority=3,
        parameters=[
            AgentParameter(
                name="service_type",
                type=ParameterType.STRING,
                required=True,
                description="Type of service to book",
                examples=["flight", "hotel"]
            ),
            AgentParameter(
                name="destination",
                type=ParameterType.STRING,
                required=True,
                description="Where to go",
                examples=["Paris", "Tokyo"]
            ),
            AgentParameter(
                name="checkin_date",
                type=ParameterType.STRING,
                required=True,
                description="When to travel",
                examples=["2024-04-15", "next week"]
            )
        ]
    )
    
    return [weather_agent, booking_agent]


class OpenRouterLLM:
    """OpenRouter LLM client using OpenAI SDK."""

    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash"):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model to use (see https://openrouter.ai/models)
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def __call__(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """
        Call the LLM with a prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLM response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RoutingError(f"LLM API call failed: {str(e)}", e)


def get_api_key() -> str:
    """Get OpenRouter API key from environment variable or .env file."""
    # Try to get from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("❌ OpenRouter API key not found!")
        print("\n🔧 Setup Instructions:")
        print("1. Get your API key from: https://openrouter.ai/keys")
        print("2. Set environment variable:")
        print("   export OPENROUTER_API_KEY='your-key-here'")
        print("3. Or create a .env file:")
        print("   echo 'OPENROUTER_API_KEY=your-key-here' > .env")
        if not DOTENV_AVAILABLE:
            print("4. Install python-dotenv for .env support:")
            print("   pip install python-dotenv")
        sys.exit(1)

    return api_key


def select_model() -> str:
    """Let user select a model or use default."""
    default_model = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-flash")
    
    models = {
        "1": ("google/gemini-2.5-flash", "Gemini 2.5 Flash (fast, cheap)"),
        "2": ("openai/gpt-4o-mini", "GPT-4o Mini (balanced)"),
        "3": ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet (smart)"),
        "4": ("meta-llama/llama-3.1-8b-instruct:free", "Llama 3.1 8B (free)")
    }

    print("🤖 Available Models:")
    for key, (model_id, name) in models.items():
        default_marker = " ⭐ (Default)" if model_id == default_model else ""
        print(f"{key}. {name}{default_marker}")
    print()

    choice = input("Select model (1-4) or Enter for default: ").strip()
    
    if not choice:
        print(f"✅ Using default: {default_model}")
        return default_model
    
    if choice in models:
        selected_model = models[choice][0]
        print(f"✅ Selected: {models[choice][1]}")
        return selected_model
    
    print("⚠️ Invalid choice, using default")
    return default_model


def demo_conversation():
    """Demonstrate a realistic conversation flow with real OpenRouter API."""
    
    print("🤖 Interactive Parameter Collection with OpenRouter")
    print("=" * 60)
    
    # Setup API
    api_key = get_api_key()
    model = select_model()
    print()
    
    # Initialize
    print("🔧 Initializing...")
    agents = create_simple_agents()
    router = LLMAgentRouter(agents)
    llm = OpenRouterLLM(api_key, model)
    session = ChatSession(router, llm)
    
    print(f"✅ Connected to {model}")
    print(f"✅ Loaded {len(agents)} agents: {[a.name for a in agents]}")
    print()
    
    # Test connection
    try:
        print("🔌 Testing connection...")
        test_response = llm("Say 'Hello, I am working correctly!'", max_tokens=50)
        print(f"✅ Connection test: {test_response}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🎮 Interactive Chat Session")
    print("Type 'quit' to exit, 'reset' to start new conversation")
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
            response = session.handle_message(user_input)
            print(f"🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Resetting session...")
            session._reset_state()


def demo_automated_tests():
    """Run some automated tests with real API."""
    
    print("\n🧪 Automated Test Mode")
    print("=" * 60)
    
    # Get API setup
    api_key = get_api_key()
    model = select_model()
    
    # Initialize
    agents = create_simple_agents()
    router = LLMAgentRouter(agents)
    llm = OpenRouterLLM(api_key, model)
    
    # Test cases
    test_cases = [
        {
            "name": "Complete Weather Request",
            "message": "What's the weather in London for 3 days?"
        },
        {
            "name": "Incomplete Weather Request", 
            "message": "How's the weather?"
        },
        {
            "name": "Complete Booking Request",
            "message": "Book a flight to Paris for tomorrow"
        },
        {
            "name": "Incomplete Booking Request",
            "message": "I want to book something"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test['name']}")
        print("-" * 40)
        print(f"Input: \"{test['message']}\"")
        
        try:
            session = ChatSession(router, llm)
            response = session.handle_message(test['message'])
            print(f"Response: {response}")
            
            # Show session state
            if session.collecting_for_agent:
                print(f"State: Collecting for {session.collecting_for_agent}")
                print(f"Missing: {session.missing_parameters}")
                print(f"Partial: {session.partial_parameters}")
        
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        print("-" * 40)


def show_integration_code():
    """Show how to integrate this into different frameworks."""
    
    print("\n🔧 Integration Examples")
    print("=" * 60)
    
    integration_examples = '''
# Flask Web App Integration:
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    session_id = request.json['session_id']
    
    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = ChatSession(router, llm)
    
    response = sessions[session_id].handle_message(user_message)
    return {"response": response}

# Discord Bot Integration:
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Get or create session for user
    user_id = message.author.id
    if user_id not in user_sessions:
        user_sessions[user_id] = ChatSession(router, llm)
    
    response = user_sessions[user_id].handle_message(message.content)
    await message.channel.send(response)

# CLI Integration:
def main():
    session = ChatSession(router, llm)
    
    print("Chat Assistant - Type 'quit' to exit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = session.handle_message(user_input)
        print(f"Bot: {response}")
'''
    
    print(integration_examples)


def main():
    """Main function with mode selection."""
    print("🚀 Agent Router with OpenRouter API")
    print("=" * 50)
    
    if not DOTENV_AVAILABLE:
        print("💡 Tip: Install python-dotenv for .env file support")
        print("   pip install python-dotenv")
        print()
    
    print("Select mode:")
    print("1. 🎮 Interactive Chat (recommended)")
    print("2. 🧪 Automated Tests")
    print("3. 📖 Integration Examples")
    print()
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            demo_conversation()
            break
        elif choice == "2":
            demo_automated_tests()
            break
        elif choice == "3":
            show_integration_code()
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()