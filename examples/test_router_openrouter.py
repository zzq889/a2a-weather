# File: examples/openrouter_demo.py
"""
OpenRouter demo for the agent router package using OpenAI SDK.
"""

import sys
import os
from typing import Optional
import asyncio

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads .env file automatically
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Add the parent directory to the path so we can import agent_router
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_router import LLMAgentRouter, create_default_agents, RoutingError, LLMResponseError

# OpenAI SDK imports
try:
    from openai import OpenAI
    import openai
except ImportError:
    print("❌ OpenAI SDK not installed. Install with: pip install openai")
    sys.exit(1)

class OpenRouterLLM:
    """OpenRouter LLM client using OpenAI SDK."""

    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.1-8b-instruct:free"):
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
    # Try to get from environment variable (works with .env if python-dotenv is installed)
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        # If not found and dotenv is not available, try to load .env manually
        if not DOTENV_AVAILABLE:
            env_file = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_file):
                print("📄 Found .env file, loading manually...")
                api_key = load_env_file_manually(env_file).get("OPENROUTER_API_KEY")

    if not api_key:
        print("❌ OpenRouter API key not found!")
        print("\n🔧 Setup Options:")
        print("1. Create a .env file with your API key:")
        print("   echo 'OPENROUTER_API_KEY=your-key-here' > .env")
        print("\n2. Install python-dotenv for automatic .env loading:")
        print("   pip install python-dotenv")
        print("\n3. Or export manually:")
        print("   export OPENROUTER_API_KEY='your-key-here'")
        print("\n4. Get your API key from: https://openrouter.ai/keys")
        sys.exit(1)

    return api_key

def load_env_file_manually(env_file: str) -> dict:
    """Manually load .env file without python-dotenv."""
    env_vars = {}
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('\'"')
                    env_vars[key.strip()] = value
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
    return env_vars

def select_model() -> str:
    """Let user select a model to test with."""
    # Check if default model is set in environment
    default_model = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-flash")
    return default_model

    models = {
        "1": ("google/gemini-2.5-flash", "Gemini 2.5 Flash"),
        "2": ("openai/gpt-4o-mini", "GPT-4o-mini"),
    }

    print("🤖 Available Models:")
    print("=" * 50)
    for key, (model_id, name) in models.items():
        default_marker = " ⭐ (Default)" if model_id == default_model else ""
        print(f"{key}. {name}{default_marker}")
    print()

    while True:
        choice = input("Select a model (1-2) or press Enter for default: ").strip()
        if not choice:
            return default_model
        if choice in models:
            selected_model = models[choice][0]
            print(f"✅ Selected: {models[choice][1]}")
            return selected_model
        print("❌ Invalid choice. Please select 1-2.")

def run_routing_tests(router: LLMAgentRouter, llm: OpenRouterLLM):
    """Run comprehensive routing tests."""

    test_cases = [
        {
            "prompt": "Write a Python function to implement binary search",
            "expected_agent": "code_assistant",
            "description": "Programming task"
        },
        {
            "prompt": "Analyze customer churn data and identify key factors",
            "expected_agent": "data_analyst",
            "description": "Data analysis task"
        },
        {
            "prompt": "Write a compelling story about a time-traveling detective",
            "expected_agent": "creative_writer",
            "description": "Creative writing task"
        },
        {
            "prompt": "Research the impact of renewable energy on job markets",
            "expected_agent": "research_assistant",
            "description": "Research task"
        },
        {
            "prompt": "How should I price my SaaS product for maximum growth?",
            "expected_agent": "business_advisor",
            "description": "Business strategy"
        },
        {
            "prompt": "What's the capital of France?",
            "expected_agent": "general_assistant",
            "description": "General knowledge"
        },
        {
            "prompt": "Help me debug this React component that won't render",
            "expected_agent": "code_assistant",
            "description": "Technical debugging"
        },
        {
            "prompt": "Create a marketing campaign for eco-friendly products",
            "expected_agent": "creative_writer",  # Could also be business_advisor
            "description": "Marketing/creative task"
        }
    ]

    print("🧪 Running Routing Tests")
    print("=" * 60)

    correct_routes = 0
    total_tests = len(test_cases)

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}/{total_tests}: {test['description']}")
        print(f"Prompt: \"{test['prompt']}\"")
        print(f"Expected: {test['expected_agent']}")

        try:
            result = router.route_prompt(test['prompt'], llm)

            print(f"✅ Routed to: {result.selected_agent}")
            print(f"   Confidence: {result.confidence.value}")
            print(f"   Reasoning: {result.reasoning}")
            print(f"   Processing time: {result.processing_time:.3f}s")

            if result.fallback_agents:
                print(f"   Fallbacks: {result.fallback_agents}")
            
            if result.extracted_parameters:
                print(f"   📋 Parameters: {result.extracted_parameters}")
            
            if result.missing_parameters:
                print(f"   ⚠️  Missing required: {result.missing_parameters}")

            # Check if routing is correct
            if result.selected_agent == test['expected_agent']:
                print("   ✅ CORRECT routing")
                correct_routes += 1
            else:
                print(f"   ⚠️  Different than expected (not necessarily wrong)")

        except (RoutingError, LLMResponseError) as e:
            print(f"   ❌ Routing error: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")

        print("-" * 60)

    # Summary
    accuracy = (correct_routes / total_tests) * 100
    print(f"\n📊 Test Summary:")
    print(f"   Correct routes: {correct_routes}/{total_tests} ({accuracy:.1f}%)")
    print(f"   Note: 'Different than expected' doesn't mean wrong - some prompts could go to multiple agents")

def interactive_mode(router: LLMAgentRouter, llm: OpenRouterLLM):
    """Interactive mode for testing custom prompts."""

    print("\n🎮 Interactive Mode")
    print("=" * 30)
    print("Enter prompts to test routing (type 'quit' to exit)")
    print()

    while True:
        try:
            prompt = input("Your prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            if not prompt:
                continue

            print("🤔 Thinking...")
            result = router.route_prompt(prompt, llm)

            print(f"➡️  Routed to: {result.selected_agent}")
            print(f"   Confidence: {result.confidence.value}")
            print(f"   Reasoning: {result.reasoning}")
            print(f"   Time: {result.processing_time:.3f}s")

            if result.fallback_agents:
                print(f"   Fallbacks: {result.fallback_agents}")
            
            if result.extracted_parameters:
                print(f"   📋 Parameters: {result.extracted_parameters}")
            
            if result.missing_parameters:
                print(f"   ⚠️  Missing required: {result.missing_parameters}")
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

def main():
    """Main demo function."""
    print("🚀 OpenRouter Multi-Agent Router Demo")
    print("=" * 50)

    # Show environment status
    if DOTENV_AVAILABLE:
        print("✅ python-dotenv available - .env files will be loaded automatically")
    else:
        print("⚠️  python-dotenv not installed - .env files will be loaded manually")
    print()

    # Get API key
    api_key = get_api_key()

    # Select model
    model = select_model()
    print()

    try:
        # Initialize LLM and router
        print("🔧 Initializing...")
        llm = OpenRouterLLM(api_key, model)
        agents = create_default_agents()
        router = LLMAgentRouter(agents)

        print(f"✅ Router initialized with {len(agents)} agents:")
        for agent_name in router.list_agents():
            print(f"   - {agent_name}")
        print()

        # Test LLM connection
        print("🔌 Testing LLM connection...")
        test_response = llm("Say 'Hello, I am working correctly!'", max_tokens=50)
        print(f"✅ LLM Response: {test_response}")
        print()

        # Run automated tests
        run_routing_tests(router, llm)

        # Interactive mode
        interactive_mode(router, llm)

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

