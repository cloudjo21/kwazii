from agno.agent import Agent
from agno.knowledge.document import DocumentKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.team.team import Team

from chromadb.config import Settings as ChromaSettings
from mem0 import Memory

from travel.db.fetch_utils import KnowledgeBaseConfig, fetch_knowledge_base


llm_name = "gpt-4o-mini"
embedder_llm_name = "text-embedding-3-small"

# ────────────────────────────────
# 1. ChromaDB 메모리 관리 설정 (선택)
#    - LRU 캐시 정책으로 2 GB 를 넘기면 오래된 segment 를 자동 언로드
# ────────────────────────────────
chroma_settings = ChromaSettings(  # :contentReference[oaicite:0]{index=0}
    chroma_segment_cache_policy="LRU", chroma_memory_limit_bytes=2 * 1024**3  # 2 GB
)


user_memory_config = {
    "vector_store": {  # Chroma 연결부
        "provider": "chroma",
        "config": {
            "collection_name": "csv_memory",
            "path": "./chromadb",  # 폴더만 있으면 자동 생성
            "client_settings": chroma_settings,  # 메모리 관리 옵션 주입
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "gemma-2-embed",
            "ollama_base_url": "http://localhost:11434/api/chat",
        },
    },
}


llm = OpenAIChat(id=llm_name)

domain_name = "travel"


kb_configs = {
    "region": KnowledgeBaseConfig(
        domain_name=domain_name,
        collection="region",
        collection_keyname="name",
        local_vector_db_path="tmp/chromadb",
    ),
    "accommodation": KnowledgeBaseConfig(
        domain_name=domain_name,
        collection="accommodation",
        collection_keyname="city_name",
        local_vector_db_path="tmp/chromadb",
    ),
    "poi": KnowledgeBaseConfig(
        domain_name=domain_name,
        collection="poi",
        collection_keyname="city_name",
        local_vector_db_path="tmp/chromadb",
    ),
}


region_knowledge_base: DocumentKnowledgeBase = fetch_knowledge_base(
    kb_configs["region"]
)
accommodation_knowledge_base: DocumentKnowledgeBase = fetch_knowledge_base(
    kb_configs["accommodation"]
)
poi_knowledge_base: DocumentKnowledgeBase = fetch_knowledge_base(kb_configs["poi"])

region_knowledge_base.load(recreate=False, skip_existing=True, upsert=False)
accommodation_knowledge_base.load(recreate=False, skip_existing=True, upsert=False)
poi_knowledge_base.load(recreate=False, skip_existing=True, upsert=False)


#### Agent Team Definition ####

search_agent = Agent(
    model=llm,
    name="Accommodation Search Agent",
    role="Search for accommodations based on user queries",
    instructions=[
        # "You are a recipe search agent.", # add this similar sentence by add_name_to_instructions=True
        "Your task is to find accommodations based on user queries.",
        "Use the provided knowledge base to search for relevant accommodations.",
        "If you find a accommodation that matches the query, return it with its details.",
        "If no accommodation matches, inform the user that no accommodations were found.",
    ],
    knowledge=accommodation_knowledge_base,
    add_name_to_instructions=True,
    show_tool_calls=True,
    use_json_mode=True,
)

accommodation_snd_team = Team(
    name="Accommodation SND Team",
    description="You are a accommodation recommender that search and discover accommodations",
    members=[search_agent],
    show_members_responses=True,
)


#### Example Usage ####

# message = "Find me a accommodation for calm leisure that can played children on the beach."
message = "Find me a accommodation for cathedrals and museums in Europe."

accommodation_snd_team.print_response(
    message=message,
    stream=True,
)
