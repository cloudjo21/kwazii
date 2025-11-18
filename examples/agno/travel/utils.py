import pandas as pd
import json

from agno.knowledge.document import Document


def read_csv_to_documents(csv_filepath, key):
    """
    Read a CSV file and convert it to a list of Document objects.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_filepath)

    rows = [row.to_dict() for _, row in df.iterrows()]
    # Convert the DataFrame to a list of Document objects
    documents = [
        Document(
            content=json.dumps(row, ensure_ascii=False),
            meta_data={key: row.get(key)},
        )
        for row in rows
    ]

    return documents
