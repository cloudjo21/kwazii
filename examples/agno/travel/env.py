from google.adk.tools.mcp_tool import mcp_session_manager


DOMAIN_NAME = "travel"

VECTOR_DB_PATH = "resources/travel/vector_db"
CONTENTS_DB_PATH = "resources/travel/contents_db"

MCP_PORT = 8888

_MCP_CONNECTION_SETTINGS = {
    "STREAMABLE_HTTP": {
        "transport": "streamable-http",
        "url": "http://localhost:8888/mcp",
        "mcp_connection_params_class": mcp_session_manager.StreamableHTTPConnectionParams
    },
    "SSE": {
        "transport": "sse",
        "url": "http://localhost:8888/sse",
        "mcp_connection_params_class": mcp_session_manager.SseConnectionParams
    },
}

_MCP_CONNECTION = "STREAMABLE_HTTP"
# _MCP_CONNECTION = "SSE"

MCP_CONNECTION = _MCP_CONNECTION_SETTINGS[_MCP_CONNECTION]
