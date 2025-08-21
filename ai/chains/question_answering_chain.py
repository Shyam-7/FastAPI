from langchain.chains import RetrievalQA
from ..vectorstores.chroma_store import get_chroma_vector_store
from langchain_openai import OpenAI

def get_qa_chain(llm, vector_store):
    """
    Returns a question answering chain.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa_chain
