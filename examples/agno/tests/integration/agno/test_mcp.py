import pytest

from mcp import ClientSession
from mcp.client.sse import sse_client

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools


server_url = "http://0.0.0.0:8888/sse"

async def get_agent():
    async with sse_client(server_url) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            async with MCPTools(
                # transport="sse",
                session=session,
            ) as mcp_tools:
                agent = Agent(
                    model=OpenAIChat(id="gpt-4o-mini"),
                    tools=[mcp_tools],
                    markdown=True,
                    add_name_to_instructions=True,
                    show_tool_calls=True,
                    use_json_mode=True,
                )
                return agent

async def run_agent():
    async with sse_client(server_url) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            async with MCPTools(
                # transport="sse",
                session=session,
            ) as mcp_tools:
                agent = Agent(
                    model=OpenAIChat(id="gpt-4o-mini"),
                    tools=[mcp_tools],
                    markdown=True,
                    add_name_to_instructions=True,
                    show_tool_calls=True,
                    use_json_mode=True,
                )
                await agent.aprint_response(
                    message="Find me a accommodation for cathedrals and museums in Europe.",
                    stream=True,
                    markdown=True,
                )
                return agent.run_response



if __name__ == "__main__":
    import asyncio
    run_response = asyncio.run(run_agent())
    
    for message in run_response.messages:
        if message.content_is_valid():
            print(f"Role: {message.role} => Message: {message.content}")
        if message.tool_calls:
            print(f"Role: {message.role} => Tool Calls:")
            for tool_call in message.tool_calls:
                print(f"- Tool Call: {tool_call['function']}")
        if message.tool_call_id:
            print(f"Tool Call ID: {message.tool_call_id}")
            if message.tool_name:
                print(f"Tool Name: {message.tool_name}")
                if message.tool_args:
                    print(f"Tool Args: {message.tool_args}")
        print("-" * 50)