import asyncio
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from a2a.types import TaskState, TextPart, DataPart, Part, UnsupportedOperationError

class WeatherAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 1) If there's no existing Task, start one using the incoming Message
        task = context.current_task
        if not task:
            assert context.message is not None, "No incoming message!"
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        # 2) Binder for sending updates
        updater = TaskUpdater(event_queue, task.id, task.contextId)

        # 3) Fetch weather and build a data artifact
        location = context.get_user_input() or "somewhere"
        full_text = f"Weather in {location}: 🌞 Sunny, 25 °C"
        artifact = {
            "type": "widget",
            "payload": {
                "temperature": 25,
                "unit": "°C",
                "condition": "Sunny",
                "icon": "sunny.png",
            }
        }
        data_part = Part(root=DataPart(data=artifact))

        # 4) Stream word-by-word (cumulative like LLM streaming)
        words = full_text.split()
        chunk = ""
        for i, word in enumerate(words):
            is_last = (i == len(words) - 1)
            chunk += word + (" " if not is_last else "")
            if not is_last:
                # interim update with cumulative text (like LLM streaming)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(chunk, task.contextId, task.id)
                )
                await asyncio.sleep(0.15)   # pause between chunks
            else:
                # attach final artifact and complete
                await updater.add_artifact(
                    [Part(root=TextPart(text=chunk)), data_part],
                    name="weather",
                )
                await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # No cancellation support
        raise ServerError(error=UnsupportedOperationError())

