from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
from langchain_openai import OpenAI
from ..vectorstores.chroma_store import get_chroma_vector_store

def get_vectorstore_tool(vector_store, llm):
    """
    Returns a vectorstore tool.
    """
    vectorstore_info = VectorStoreInfo(
        name="vehicle_service_history",
        description="Contains information about vehicle service history",
        vectorstore=vector_store
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    agent = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)
    return agent
