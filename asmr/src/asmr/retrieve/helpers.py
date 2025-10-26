from asmr.retrieve import aggregate
from asmr.retrieve import retrievers


class DocumentRetriever:

    def __init__(self, fields_retriever: retrievers.FieldComplexRetriever):
        self.fields_retriever = fields_retriever

    async def retrieve(self, query: str, k: int):
        return await aggregate.retrieve_documents(
            query,
            self.fields_retriever.fields,
            self.fields_retriever,
            k
        )
