"""
Basic usage examples for the agent router package.
"""

import sys
import os

# Add the parent directory to the path so we can import agent_router
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_router import LLMAgentRouter, create_default_agents, RoutingError

def mock_llm_function(prompt: str) -> str:
    """Mock LLM function for testing purposes."""
    # This would be replaced with your actual LLM API call
    return '''
    {
        "selected_agent": "code_assistant",
        "confidence": "high",
        "reasoning": "The user is asking for help with a Python programming task, which directly matches the code_assistant's capabilities for writing and explaining code.",
        "fallback_agents": ["general_assistant"]
    }
    '''

def main():
    """Demonstrate basic usage of the agent router."""
    try:
        # Create router with default agents
        agents = create_default_agents()
        router = LLMAgentRouter(agents)

        print("🤖 Multi-Agent Router Demo")
        print("=" * 50)
        print(f"Available agents: {', '.join(router.list_agents())}")
        print()

        # Test prompts
        test_prompts = [
            "Help me write a Python function to sort a list",
            "Analyze this CSV file and find correlations",
            "Write a poem about the ocean",
            "Research the history of the Internet",
            "What's my best business strategy for launching a startup?",
            "What's 2 + 2?",
            ""  # Test empty prompt
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"Test {i}: '{prompt}'")
            try:
                result = router.route_prompt(prompt, mock_llm_function)
                print(f"  → Routed to: {result.selected_agent}")
                print(f"  → Confidence: {result.confidence.value}")
                print(f"  → Reasoning: {result.reasoning}")
                if result.fallback_agents:
                    print(f"  → Fallbacks: {result.fallback_agents}")
                if result.processing_time:
                    print(f"  → Processing time: {result.processing_time:.3f}s")
                print()
            except RoutingError as e:
                print(f"  ❌ Routing error: {e}")
                print()
            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
                print()

        # Demonstrate agent management
        print("🔧 Agent Management Demo")
        print("=" * 30)

        # Get agent info
        agent_info = router.get_agent_info("code_assistant")
        print(f"Code Assistant capabilities: {', '.join(agent_info.capabilities[:3])}...")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()

