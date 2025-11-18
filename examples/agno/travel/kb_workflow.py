import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.run.response import RunResponse
from agno.team.team import Team
from agno.tools.mcp import MCPTools


llm_name = "gpt-4o-mini"
llm = OpenAIChat(id=llm_name)

# This is the URL of the MCP server we want to use.
server_url = "http://0.0.0.0:8888/sse"


async def get_search_agent(session) -> Agent:
    """Create and return an Agent with MCPTools."""
    # Initialize the MCPTools
    mcp_tools = MCPTools(session=session)
    await mcp_tools.initialize()

    # Create the Agent with the MCPTools
    search_agent = Agent(
        model=llm,
        name="Accommodation Search Agent",
        role="Search for accommodations based on user queries",
        tools=[mcp_tools],
        instructions=[
            "Your task is to find accommodations based on user queries.",
            "Use the provided tools to search for relevant accommodations.",
            "If you find a accommodation that matches the query, return it with its details.",
            "If no accommodation matches, inform the user that no accommodations were found.",
        ],
        add_name_to_instructions=True,
        show_tool_calls=True,
        use_json_mode=True,
    )
    return search_agent


async def run_search_agent(message: str, mcp_server_url: str) -> RunResponse:

    async with sse_client(mcp_server_url) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            agent = await get_search_agent(session)

            # response = await agent.arun(
            #     message=message,
            #     # stream=True, # return AsyncIterator
            # )
            # return response

            # Debug mode to print the agent's response
            await agent.aprint_response(
                message=message,
                stream=True,
            )
            return agent.run_response


async def run_team(message: str, mcp_server_url: str) -> RunResponse:
    """Run a Team with the search agent."""
    async with sse_client(mcp_server_url) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            search_agent = await get_search_agent(session)

            accommodation_snd_team = Team(
                name="Accommodation SND Team",
                description="You are an accommodation recommender that searches and discovers accommodations",
                members=[search_agent],
                show_tool_calls=True,
                show_members_responses=True,
            )
            await accommodation_snd_team.aprint_response(
                message=message,
                stream=True,
            )
            return accommodation_snd_team.run_response


async def main():
    message = "Find me a accommodation for cathedrals and museums in Bern."

    # receive response by arun
    response = await run_team(message, server_url)
    print(response.content)

    # # receive response by arun
    # response = await run_search_agent(message)
    # print(response.content)

    # # ? receive response by arun with stream=True
    # async_generator = await run_search_agent(message)
    # async for response in async_generator:
    #     print(response.content)

    # a = await result
    # result = await asyncio.gather(run_search_agent(message))

    # response = asyncio.run(run_search_agent(message))
    # print(response.content)
    # asyncio.run(
    #     search_agent.aprint_response(
    #         message=message,
    #         stream=True,
    #     )
    # )


if __name__ == "__main__":
    asyncio.run(main())
