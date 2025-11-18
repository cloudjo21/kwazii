
from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.embedder.openai import OpenAIEmbedder

KNOWLEDGE_BASE_FILE = 'travel/resources/accommodation.csv'


embedder = OpenAIEmbedder(id="text-embedding-3-small", dimensions=1536)

vector_db = ChromaDb(collection="recipes", path="tmp/chromadb", persistent_client=True, embedder=embedder)
knowledge_base = CSVKnowledgeBase(
    path=KNOWLEDGE_BASE_FILE,
    vector_db=vector_db,
    formats=["csv"],
    metadata={
        "source": "travel",
        "description": "A collection of accommodation from my travels.",
    },
)

knowledge_base.load(recreate=False)

for documents in knowledge_base.document_lists:
    for document in documents:
        print(document.__dict__)
        print("-----")
    # print(document.metadata)
    # print(document.text)
    # print("-----")