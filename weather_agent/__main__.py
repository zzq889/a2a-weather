import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from agent_executor import WeatherAgentExecutor

if __name__ == "__main__":
    skill = AgentSkill(
        id="get_weather",
        name="Get Weather",
        description="Returns a fake weather string.",
        tags=['weather'],
        examples=['Paris']
    )

    card = AgentCard(
        name="Weather Agent",
        description="Mock weather service",
        url="http://localhost:7777/",
        version="0.0.1",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    handler = DefaultRequestHandler(WeatherAgentExecutor(), InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=7777)

