import unittest
from ai.agents.retrieval_agent import get_retrieval_agent
from langchain_community.vectorstores import Chroma
from langchain.llms.fake import FakeLLM
from langchain_community.embeddings import FakeEmbeddings

class TestAgents(unittest.TestCase):

    def test_retrieval_agent(self):
        """
        Tests the retrieval agent.
        """
        embedding_function = FakeEmbeddings(size=768)
        vector_store = Chroma(embedding_function=embedding_function, persist_directory="./test_db")

        queries = ["What is the service history of my car?"]
        responses = ["The service history of your car is..."]
        llm = FakeLLM(responses=responses, queries=queries)

        agent = get_retrieval_agent(llm, vector_store)

        result = agent.run("What is the service history of my car?")
        self.assertEqual(result, "The service history of your car is...")

if __name__ == '__main__':
    unittest.main()
