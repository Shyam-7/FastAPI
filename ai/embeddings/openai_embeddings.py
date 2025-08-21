from langchain_openai import OpenAIEmbeddings

def get_openai_embeddings():
    """
    Returns an OpenAI embedding model.
    """
    return OpenAIEmbeddings()
