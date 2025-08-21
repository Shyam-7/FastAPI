from langchain_community.vectorstores import Chroma
from ..embeddings.hf_embeddings import get_hf_embeddings

def get_chroma_vector_store(path: str, embedding_function=None):
    """
    Returns a Chroma vector store.
    """
    if embedding_function is None:
        embedding_function = get_hf_embeddings()
    return Chroma(persist_directory=path, embedding_function=embedding_function)
