"""
Travel Agency Agent Team for searching attractions and POIs for specific regions or cities using MCP server.
"""

import asyncio

from textwrap import dedent

from mcp import ClientSession
from mcp.client.sse import sse_client

from agno.agent import Agent
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.openai import OpenAIChat
from agno.run.response import RunResponse
from agno.storage.sqlite import SqliteStorage
from agno.team.team import Team
from agno.tools.mcp import MCPTools

from travel.db.models import (
    AccommodationSearchResponse,
    PointOfInterestSearchResponse,
    RegionSearchResponse,
    TravelSearchResponse,
)


user_id = "user_12345"  # Example user ID, can be any unique identifier for the user
session_id = (
    "session_67890"  # Example session ID, can be any unique identifier for the session
)

# llm_name = "gpt-4o-mini"
llm_name = "o4-mini"
llm = OpenAIChat(id=llm_name)

# This is the URL of the MCP server we want to use.
server_url = "http://0.0.0.0:8888/sse"

team_storage = SqliteStorage(
    table_name="team_memory",
)
memory_db = SqliteMemoryDb(table_name="team_memory", db_file="tmp/memory.db")
memory = Memory(
    db=memory_db, debug_mode=True, delete_memories=True, clear_memories=True
)


search_agents = [
    {
        "agent_id": "accommodation_search_agent",
        "name": "Accommodation Search Agent",
        "role": "Search for accommodations based on user query",
        "description": dedent(
            """
            You are a accommodation search agent that searches for accommodations for specific regions or cities using MCP server.
            Always respond from the results through search_accommodations.
            Don't use external knowledge base, use only MCP server to search accommodations.
            """
        ),
        "resopnse_model": AccommodationSearchResponse,
    },
    {
        "agent_id": "point_of_interst_search_agent",
        "name": "Point-Of-Interest Search Agent",
        "role": "Search for Point-Of-Interests based on user query",
        "description": dedent(
            """
            You are a point-of-interest search agent that searches for specific regions or cities using MCP server.
            Always respond from the results through search_point_of_interests.
            Don't use external knowledge base, use only MCP server to search Point-Of-Interests.
            """
        ),
        "resopnse_model": PointOfInterestSearchResponse,
    },
    {
        "agent_id": "region_search_agent",
        "name": "Region Search Agent",
        "role": "Search for region as city based on user query",
        "description": dedent(
            """
            You are a city search agent that searches for specific regions or cities using MCP server.
            Always respond from the results through search_cities.
            Don't use external knowledge base, use only MCP server to search regions.
            """
        ),
        "resopnse_model": RegionSearchResponse,
    },
]


def get_search_agent_config(agent_id: str):
    """Get the configuration for the search agent based on agent_id."""
    for agent in search_agents:
        if agent["agent_id"] == agent_id:
            return agent
    raise ValueError(f"Agent with id {agent_id} not found.")


async def get_search_agent(session, agent_id) -> Agent:
    """Create and return an Agent with MCPTools."""

    agent_config = get_search_agent_config(agent_id)

    # Initialize the MCPTools
    mcp_tools = MCPTools(session=session)
    await mcp_tools.initialize()

    # Create the Agent with the MCPTools
    search_agent = Agent(
        agent_id=agent_config["agent_id"],
        user_id=user_id,
        session_id=session_id,
        model=llm,
        name=agent_config["name"],
        role=agent_config["role"],
        tools=[mcp_tools],
        description=agent_config["description"],
        response_model=agent_config["resopnse_model"],
        structured_outputs=True,
        use_json_mode=True,
        add_datetime_to_instructions=True,
        add_name_to_instructions=True,
        show_tool_calls=True,
        retries=1,
        # debugging
        debug_mode=True,
        debug_level=2,
    )
    return search_agent


async def run_team(message: str, mcp_server_url: str) -> RunResponse:
    """Run a Team with the search agent."""
    async with sse_client(mcp_server_url) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            accomodation_search_agent = await get_search_agent(
                session, "accommodation_search_agent"
            )
            poi_search_agent = await get_search_agent(
                session, "point_of_interst_search_agent"
            )
            city_search_agent = await get_search_agent(session, "region_search_agent")

            team = Team(
                reasoning_max_steps=1,
                user_id=user_id,
                session_id=session_id,
                name="Travel Agency Team",
                description="You are an travel agency that searches and discovers accommodations or Point-Of-Interests for specific regions or cities with Travel Search Agent.",
                instructions=[
                    # "Use search_cities tool to search for regions or cities if you don't have enough information, and use city names to re-write message to search accommodation or point-of-interest.",
                    "At first, use search_cities tool to search for regions or cities based on user query and memory."
                    "Use only city names from region search results to re-write message to search accommodation or point-of-interest.",
                    "Read memory if user's memory is available and pass it to the corresponding search agent.",
                    "Use memory tool to remember user queries and responses, and use it to improve your responses",
                    "Add or update memory to remember accommodations just search requested with topics=[travel, search, accommodations] if you successfully get accommodation search results.",
                    "Add or update memory to remember point-of-interest just search requested with topics=[travel, search, pois] if you successfully get point-of-interests search results.",
                    "Delete memory to forget user queries and responses if user asks to do so",
                ],
                members=[
                    accomodation_search_agent,
                    poi_search_agent,
                    city_search_agent,
                ],
                response_model=TravelSearchResponse,
                # memory
                memory=memory,
                storage=team_storage,
                enable_agentic_memory=True,
                enable_user_memories=True,
                add_memory_references=True,
                show_tool_calls=True,
                show_members_responses=True,
                # debugging
                debug_mode=True,
                debug_level=2,
            )
            await team.aprint_response(
                message=message,
                user_id=user_id,
                session_id=session_id,
                show_full_reasoning=True,
            )
            return team.run_response


async def main():
    # message1 = "지금까지 검색했던 여행기록 삭제해줘."
    message1 = "Find me a accommodation for cathedrals and museums in Barcelona."
    message2 = "Find me a accommodation for cathedrals and museums in Europe and and please exclude the cities as I've searched before."

    response = await run_team(message1, server_url)
    response = await run_team(message2, server_url)

    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
