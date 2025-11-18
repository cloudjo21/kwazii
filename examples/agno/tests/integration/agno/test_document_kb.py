import json
import pandas as pd
import pytest

from agno.knowledge.document import Document
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb

cases = [
    ('accommodation', "travel/resources/accommodation.csv", 'city_name', '도쿄 호텔'),
    ('poi', "travel/resources/poi.csv", 'city_name', '정적인 관광지'),
    ('region', "travel/resources/region.csv", 'name', '바다와 해변의 도시')
]
RECREATE = False


@pytest.mark.parametrize("collection, csv_filepath, key, query", cases)
def test_document_knowledge_base(collection, csv_filepath, key, query):
    """
    Test reading a CSV file and converting it to a list of Document objects.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_filepath, nrows=13)

    rows = [row.to_dict() for _, row in df.iterrows()]
    # Convert the DataFrame to a list of Document objects
    documents = [
        Document(
            content=json.dumps(row, ensure_ascii=False),
            meta_data={key: row.get(key)},
        )
        for row in rows
    ]

    # Print the first few documents
    print()
    for doc in documents[:2]:
        print(doc.content)

    vector_db = ChromaDb(
        collection=collection,
        path=f"tmp/chromadb/{collection}",
        persistent_client=True,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    )
    kb = Knowledge(
        name=f"test-{collection}",
        description=f"Knowledge for test/{collection}",
        vector_db=vector_db,
    )

    search_documents = kb.search(query=query, max_results=3)
    print()
    for doc in search_documents:
        print(doc.content)

    assert len(search_documents) == 3, "No documents were found in the knowledge base."
