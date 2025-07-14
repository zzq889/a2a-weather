"""
Pre-defined agent configurations for common use cases.
"""

from typing import List
from .models import Agent

def create_default_agents() -> List[Agent]:
    """Create a set of default agents for common use cases."""
    return [
        Agent(
            name="code_assistant",
            description="Specialized in programming, code review, debugging, and technical implementation",
            capabilities=[
                "Write code in multiple programming languages",
                "Debug and fix code issues",
                "Explain programming concepts",
                "Code review and optimization",
                "API documentation and integration",
                "Software architecture advice"
            ],
            examples=[
                "Write a Python function to calculate fibonacci numbers",
                "Debug this JavaScript code that's not working",
                "How do I implement authentication in my web app?",
                "Review my code for best practices",
                "Explain how this algorithm works"
            ],
            priority=3
        ),
        Agent(
            name="data_analyst",
            description="Expert in data analysis, statistics, visualization, and machine learning",
            capabilities=[
                "Analyze datasets and generate insights",
                "Create data visualizations",
                "Statistical analysis and hypothesis testing",
                "Machine learning model recommendations",
                "Data cleaning and preprocessing advice",
                "SQL query optimization"
            ],
            examples=[
                "Analyze this sales data and find trends",
                "What's the best way to visualize customer demographics?",
                "Help me choose the right ML algorithm for classification",
                "How do I handle missing values in my dataset?",
                "Write a SQL query to find top customers"
            ],
            priority=3
        ),
        Agent(
            name="creative_writer",
            description="Specialized in creative writing, storytelling, and content creation",
            capabilities=[
                "Write creative stories and narratives",
                "Develop characters and plot lines",
                "Create marketing copy and content",
                "Poetry and creative expression",
                "Edit and improve writing style",
                "Social media content creation"
            ],
            examples=[
                "Write a short story about time travel",
                "Help me create a compelling character for my novel",
                "Write marketing copy for my new product",
                "Improve the flow of this paragraph",
                "Create a social media post for our campaign"
            ],
            priority=2
        ),
        Agent(
            name="research_assistant",
            description="Expert in research, fact-checking, analysis, and academic writing",
            capabilities=[
                "Conduct comprehensive research on topics",
                "Fact-check information and claims",
                "Synthesize information from multiple sources",
                "Academic writing and citation help",
                "Market research and competitive analysis",
                "Literature reviews and summaries"
            ],
            examples=[
                "Research the history of renewable energy",
                "Fact-check these claims about climate change",
                "Help me write a literature review on AI ethics",
                "What are the latest trends in electric vehicles?",
                "Summarize recent papers on quantum computing"
            ],
            priority=2
        ),
        Agent(
            name="business_advisor",
            description="Expert in business strategy, entrepreneurship, and professional development",
            capabilities=[
                "Business strategy and planning",
                "Market analysis and competitive intelligence",
                "Financial planning and analysis",
                "Professional development advice",
                "Leadership and management guidance",
                "Startup and entrepreneurship support"
            ],
            examples=[
                "Help me create a business plan for my startup",
                "What's the best pricing strategy for my product?",
                "How do I improve team productivity?",
                "Analyze the competitive landscape for my industry",
                "What skills should I develop for career growth?"
            ],
            priority=2
        ),
        Agent(
            name="general_assistant",
            description="General-purpose assistant for everyday tasks and broad inquiries",
            capabilities=[
                "Answer general knowledge questions",
                "Provide explanations on various topics",
                "Help with everyday tasks and planning",
                "Basic math and calculations",
                "General conversation and advice",
                "Multi-topic support when other agents don't fit"
            ],
            examples=[
                "What's the weather like today?",
                "Help me plan my daily schedule",
                "Explain how photosynthesis works",
                "What are some good recipes for dinner?",
                "Convert 100 degrees Fahrenheit to Celsius"
            ],
            priority=1  # Lowest priority - fallback agent
        )
    ]

def create_coding_agents() -> List[Agent]:
    """Create agents specialized for software development."""
    return [
        Agent(
            name="frontend_developer",
            description="Specialized in frontend development, UI/UX, and web technologies",
            capabilities=[
                "React, Vue, Angular development",
                "HTML, CSS, JavaScript",
                "UI/UX design principles",
                "Responsive web design",
                "Frontend build tools and optimization"
            ],
            examples=[
                "Create a responsive navbar component",
                "Help with CSS flexbox layout",
                "Optimize my React app performance"
            ],
            priority=3
        ),
        Agent(
            name="backend_developer",
            description="Expert in backend development, APIs, and server-side technologies",
            capabilities=[
                "REST API and GraphQL development",
                "Database design and optimization",
                "Server architecture and deployment",
                "Authentication and security",
                "Microservices and cloud technologies"
            ],
            examples=[
                "Design a user authentication system",
                "Help me optimize my database queries",
                "Create a REST API for my application"
            ],
            priority=3
        ),
        Agent(
            name="devops_engineer",
            description="Specialized in DevOps, infrastructure, and deployment automation",
            capabilities=[
                "CI/CD pipeline setup",
                "Docker and containerization",
                "Cloud infrastructure (AWS, GCP, Azure)",
                "Monitoring and logging",
                "Infrastructure as Code"
            ],
            examples=[
                "Set up a CI/CD pipeline for my project",
                "Help me containerize my application",
                "Configure monitoring for my services"
            ],
            priority=2
        )
    ]

