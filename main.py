import asyncio, uuid, httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Role,
    Message,
    Part,
    TextPart,
    MessageSendParams,
    SendMessageRequest
)

async def main() -> None:
    location = "Paris"
    weather_url = "http://localhost:7777"

    async with httpx.AsyncClient() as httpx_client:
        # 1) Fetch Weather Agent’s card ---------------------------
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=weather_url)
        weather_card = await resolver.get_agent_card()

        # 2) Send the location string ------------------------------
        client = A2AClient(httpx_client=httpx_client, agent_card=weather_card)

        message = Message(
            messageId=uuid.uuid4().hex,
            role=Role.user,
            parts=[Part(root=TextPart(text=location))],
        )
        request = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(message=message, configuration=None),
        )
        response = await client.send_message(request)

        # 3) Print the Weather Agent’s answer ----------------------
        print(response.model_dump(mode='json', exclude_none=True))

if __name__ == "__main__":
    asyncio.run(main())

