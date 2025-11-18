import asyncio

from google.adk.tools.mcp_tool import mcp_toolset
from google.adk.tools.mcp_tool import mcp_session_manager
from google.adk.runners import InMemoryRunner
from google.adk.agents import llm_agent
from google.adk.models.lite_llm import LiteLlm

from phoenix.otel import register

# Configure the Phoenix tracer
tracer_provider = register(
  project_name="my-travel-app", # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed OI dependencies
)

from travel import adk_utils
from travel import env

mcp_toolset = mcp_toolset.McpToolset(
    connection_params=env.MCP_CONNECTION["mcp_connection_params_class"](
        url=env.MCP_CONNECTION["url"],
        timeout=10.0,
        sse_read_timeout=300.0,
    ),
    tool_filter=[
        "search_accommodations", "search_point_of_interests", "search_regions"
    ],
)

root_agent = llm_agent.LlmAgent(
    model=LiteLlm(model='openai/gpt-4o'),
    name="TravelAgent",
    description=
    "An agent that helps users find travel accommodations, points of interest, and regions.",
    tools=[mcp_toolset],
)
runner = InMemoryRunner(
    agent=root_agent,
    app_name="TravelAdkAgent",
)


async def main():
    user_input = "Find me a accommodation for cathedrals and museums in Europe."
    response = await adk_utils.run_prompt(
        runner=runner,
        prompt=user_input,
        user_id="user_123",
        app_name="TravelAdkAgent",
    )
    return response


if __name__ == "__main__":

    print("Agent Response:")
    response = asyncio.run(main())
    print(response)
