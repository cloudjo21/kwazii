import pytest

import pathlib

from agno.document.reader.csv_reader import CSVReader
from agno.document.chunking.recursive import RecursiveChunking

@pytest.fixture
def reader():
    return CSVReader()

def test_read(reader):
    """
    just a simple test to read a CSV file to observe odd working of CSVReader
    """
    # csv_filepath = pathlib.Path("travel/resources/accommodation.csv")
    csv_filepath = pathlib.Path("travel/resources/small/sample.csv")

    documents = reader.read(csv_filepath)
    # documents = reader.read(csv_filepath, delimiter='\n')
    # assert len(documents) > 0, "No documents were read from the CSV file."
    print(f"#### {len(documents)} documents read from {csv_filepath}")
    print(documents)