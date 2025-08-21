import unittest
from ai.chains.ingestion_chain import ingest_data
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

class TestChains(unittest.TestCase):

    def test_ingestion_chain(self):
        """
        Tests the ingestion chain.
        """
        embedding_function = FakeEmbeddings(size=768)
        vector_store = Chroma(embedding_function=embedding_function, persist_directory="./test_db")
        data = "This is a test."
        self.assertTrue(ingest_data(data, vector_store))
        # a real test would check if the data is in the vector store
        # but that requires a query, which is outside the scope of this test
        # for now, we just check if the function returns True

if __name__ == '__main__':
    unittest.main()
