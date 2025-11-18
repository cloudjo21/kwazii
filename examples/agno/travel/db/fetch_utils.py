import pydantic

from agno.document import Document
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.document import DocumentKnowledgeBase
from agno.vectordb.chroma.chromadb import ChromaDb

from travel.utils import read_csv_to_documents


class KnowledgeBaseConfig(pydantic.BaseModel):
    """
    Configuration for the knowledge base.
    """

    domain_name: str
    collection: str
    collection_keyname: str
    local_vector_db_path: str = "tmp/chromadb"



def fetch_knowledge_base(kb_config: KnowledgeBaseConfig) -> DocumentKnowledgeBase:
    """
    Fetch the knowledge base for the given collection.
    """
    vector_db = ChromaDb(
        collection=kb_config.collection,
        path=f"{kb_config.local_vector_db_path}/{kb_config.collection}",
        persistent_client=True,
        embedder=OpenAIEmbedder(),
    )
    documents: list[Document] = read_csv_to_documents(
        csv_filepath=f"{kb_config.domain_name}/resources/{kb_config.collection}.csv",
        key=kb_config.collection_keyname,
    )
    return DocumentKnowledgeBase(
        documents=documents,
        vector_db=vector_db,
    )
