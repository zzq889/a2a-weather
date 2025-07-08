from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_parts_message
from a2a.types import TextPart, DataPart, Part

class WeatherAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # INPUT PARAM IS NOW  “location”
        location = context.get_user_input() or "somewhere"
        text = f"Weather in {location}: 🌞 Sunny, 25 °C"
        artifact = {
            "type": "widget",
            "data": {
                "temperature": 25,
                "unit": "°C",
                "condition": "Sunny",
                "icon": "sunny.png",
            }
        }
        data_part = Part(root=DataPart(data=artifact))
        parts = [
            Part(root=TextPart(text=text)),
            data_part,
        ]
        await event_queue.enqueue_event(
            new_agent_parts_message(
                parts,
                context_id=context.context_id,
                task_id=context.task_id,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")

