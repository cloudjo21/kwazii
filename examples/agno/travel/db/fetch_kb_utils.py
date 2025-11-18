import pydantic

from agno.knowledge.document import Document
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.chroma import ChromaDb
from agno.db.sqlite import SqliteDb

from travel.utils import read_csv_to_documents


class KnowledgeBaseConfig(pydantic.BaseModel):
    """
    Configuration for the knowledge base.
    """

    domain_name: str
    collection: str
    collection_keyname: str
    local_vector_db_path: str = "resources/travel/vector_db"
    local_contents_db_path: str = "resources/travel/contents_db"
    skip_if_exists: bool = True


def fetch_knowledge_base(kb_config: KnowledgeBaseConfig) -> Knowledge:
    """
    Fetch the knowledge base for the given collection.
    """

    # 1) Vector DB 생성
    vector_db = ChromaDb(
        collection=kb_config.collection,
        path=f"{kb_config.local_vector_db_path}/{kb_config.collection}",
        persistent_client=True,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    )

    contents_db = SqliteDb(
        db_file=f"{kb_config.local_vector_db_path}/{kb_config.collection}.db"
    )

    # 2) Knowledge 인스턴스 생성
    knowledge = Knowledge(
        name=f"{kb_config.domain_name}-{kb_config.collection}",
        description=f"Knowledge for {kb_config.domain_name}/{kb_config.collection}",
        vector_db=vector_db,
        contents_db=contents_db,
    )

    # 3) CSV → Document 리스트 로딩
    documents: list[Document] = read_csv_to_documents(
        csv_filepath=f"resources/{kb_config.domain_name}/{kb_config.collection}.csv",
        key=kb_config.collection_keyname,
    )

    # 4) Document 리스트를 add_contents로 한 번에 적재
    contents: list[dict] = []

    for doc in documents:
        # content가 비어 있으면 스킵
        if not doc.content:
            continue

        metadata = doc.meta_data or {}
        name = None

        if isinstance(metadata, dict):
            name = metadata.get(kb_config.collection_keyname)

        contents.append(
            {
                "name": name,
                "text_content": doc.content,
                "meta_data": metadata,
            }
        )

    if contents:
        knowledge.add_contents(contents, skip_if_exists=kb_config.skip_if_exists)

    return knowledge
