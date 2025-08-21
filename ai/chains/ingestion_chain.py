from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..vectorstores.chroma_store import get_chroma_vector_store

def ingest_data(data, vector_store):
    """
    Ingests data into the vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(data)
    vector_store.add_texts(texts)
    return True
