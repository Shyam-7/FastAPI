from langchain.agents import initialize_agent
from ..tools.vectorstore_tool import get_vectorstore_tool

def get_retrieval_agent(llm, vector_store):
    """
    Returns a retrieval agent.
    """
    tools = [get_vectorstore_tool(vector_store, llm)]

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    return agent
