import json
import pytest

from agno.tools.csv_toolkit import CsvTools

CSV_FILEPATH = "travel/resources/accommodation.csv"

@pytest.fixture
def csv_tools():
    return CsvTools(csvs=[CSV_FILEPATH])

def test_csv_tools(csv_tools):
    print(csv_tools.get_columns('accommodation'))
    print(csv_tools.read_csv_file('accommodation', row_limit=3))
    assert len(json.loads(csv_tools.read_csv_file('accommodation', row_limit=3))) == 3
