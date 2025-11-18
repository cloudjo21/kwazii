"""Start an MCP server that uses the Streamable HTTP transport."""

# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP

from agno.knowledge.document import Document
from agno.knowledge.knowledge import Knowledge

from travel import env
from travel.db import fetch_kb_utils


_DOMAIN_NAME = env.DOMAIN_NAME
# init_knowledge_base: bool = False

search_mcp = FastMCP("travel_info_search_assistant")
main_mcp = FastMCP("main", port=env.MCP_PORT)

kb_configs = {
    "region":
    fetch_kb_utils.KnowledgeBaseConfig(
        domain_name=_DOMAIN_NAME,
        collection="region",
        collection_keyname="name",
        local_vector_db_path=env.VECTOR_DB_PATH,
    ),
    "accommodation":
    fetch_kb_utils.KnowledgeBaseConfig(
        domain_name=_DOMAIN_NAME,
        collection="accommodation",
        collection_keyname="city_name",
        local_vector_db_path=env.VECTOR_DB_PATH,
    ),
    "poi":
    fetch_kb_utils.KnowledgeBaseConfig(
        domain_name=_DOMAIN_NAME,
        collection="poi",
        collection_keyname="city_name",
        local_vector_db_path=env.VECTOR_DB_PATH,
    ),
}

region_knowledge_base: Knowledge = (fetch_kb_utils.fetch_knowledge_base(
    kb_configs["region"]))
accommodation_knowledge_base: Knowledge = (fetch_kb_utils.fetch_knowledge_base(
    kb_configs["accommodation"]))
poi_knowledge_base: fetch_kb_utils.Knowledge = (
    fetch_kb_utils.fetch_knowledge_base(kb_configs["poi"]))

# region_knowledge_base.load(
#     recreate=init_knowledge_base, skip_existing=True, upsert=False
# )
# accommodation_knowledge_base.load(
#     recreate=init_knowledge_base, skip_existing=True, upsert=False
# )
# poi_knowledge_base.load(recreate=init_knowledge_base, skip_existing=True, upsert=False)


@search_mcp.tool()
def search_accommodations(query: str) -> str:
    """
    search accommodations based on the query.
    """
    print(f"Searching accommodations for query: {query}")

    global accommodation_knowledge_base
    documents: list[Document] = accommodation_knowledge_base.search(
        query, max_results=3)

    if documents:
        results = "\n".join(f"- Accommodation: {doc.content}"
                            for doc in documents)
        print(f"Here are some accommodations found:\n{results}")
        return f"Here are some accommodations found:\n{results}"

    return "No accommodations found for your query."


@search_mcp.tool()
def search_point_of_interests(query: str) -> str:
    """
    search Point-Of-Interests based on the query.
    """
    print(f"Searching POIs for query: {query}")

    global poi_knowledge_base
    documents: list[Document] = poi_knowledge_base.search(query, max_results=3)

    if documents:
        results = "\n".join(f"- POI: {doc.content}" for doc in documents)
        print(f"Here are some Point-of-Interests found:\n{results}")
        return f"Here are some Point-of-Interests found:\n{results}"

    return "No POIs found for your query."


@search_mcp.tool()
def search_regions(query: str) -> str:
    """
    search regions based on the query.
    """
    print(f"Searching regions for query: {query}")

    global region_knowledge_base
    documents: list[Document] = region_knowledge_base.search(query,
                                                             max_results=3)

    if documents:
        results = "\n".join(f"- Region: {doc.content}" for doc in documents)
        print(f"Here are some regions found:\n{results}")
        return f"Here are some regions found:\n{results}"

    return "No regions found for your query."


if __name__ == "__main__":
    main_mcp.mount(server=search_mcp)  # , prefix="search")
    main_mcp.run(transport=env.MCP_CONNECTION["transport"])
