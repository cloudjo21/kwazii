"""
Show how to connect to MCP servers that use either SSE or Streamable HTTP transport using our MCPTools and MultiMCPTools classes.

Check the README.md file for instructions on how to run these examples.
"""

import asyncio

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools

# This is the URL of the MCP server we want to use.
server_url = "http://0.0.0.0:8888/sse"


async def run_agent(message: str) -> None:
    # async with MCPTools(transport="streamable-http", url=server_url) as mcp_tools:
    async with MCPTools(transport="sse", url=server_url) as mcp_tools:
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[mcp_tools],
            markdown=True,
            add_name_to_context=True,
            # show_tool_calls=True,
            use_json_mode=True,
        )
        await agent.aprint_response(input=message, stream=True, markdown=True)
        # for message in agent.run_messages.messages:
        #     if message.content_is_valid():
        #         print(f"Role: {message.role} => Message: {message.content}")
        #     if message.tool_calls:
        #         print(f"Role: {message.role} => Tool Calls:")
        #         for tool_call in message.tool_calls:
        #             print(
        #                 f"- Tool Call: {tool_call['function']}"
        #             )
        #     if message.tool_call_id:
        #         print(f"Tool Call ID: {message.tool_call_id}")
        #         if message.tool_name:
        #             print(f"Tool Name: {message.tool_name}")
        #             if message.tool_args:
        #                 print(f"Tool Args: {message.tool_args}")
        #     print("-" * 50)


if __name__ == "__main__":
    asyncio.run(
        run_agent("Find me a accommodation for cathedrals and museums in Europe.")
    )
