from typing import AsyncGenerator

from google.genai import types
from google.adk import runners
from google.adk.sessions import session


async def run_prompt(runner: runners.Runner, prompt: str, user_id: str, app_name: str) -> str:
    content = types.Content(parts=[types.Part(text=prompt)], role="user")

    session: session.Session = await runner.session_service.create_session(user_id=user_id, app_name=app_name)

    events = runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    )
    text_contents = []
    async for event in events:  # type: ignore
        if event.content.parts and event.content.parts[0].text:
            text_contents.append(event.content.parts[0].text)

    response = "\n".join(text_contents)
    # if event.content.parts and event.content.parts[0].text:
    #     response = event.content.parts[0].text

    return response
