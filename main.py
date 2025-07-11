import asyncio, uuid, httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Role,
    Message,
    Part,
    SendMessageRequest,
    SendStreamingMessageRequest,
    TextPart,
    MessageSendParams,
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
            messageId=str(uuid.uuid4()),
            role=Role.user,
            parts=[Part(root=TextPart(text=location))],
        )

        if weather_card.capabilities.streaming:
            request = SendStreamingMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(message=message),
            )
            stream_response = client.send_message_streaming(request)

            async for chunk in stream_response:
                print(chunk.model_dump(mode='json', exclude_none=True))
        else:
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(message=message),
            )
            response = await client.send_message(request)
            print(response.model_dump(mode='json', exclude_none=True))


if __name__ == "__main__":
    asyncio.run(main())

