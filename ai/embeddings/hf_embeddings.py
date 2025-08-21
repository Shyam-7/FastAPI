from langchain_community.embeddings import HuggingFaceEmbeddings

def get_hf_embeddings():
    """
    Returns a Hugging Face embedding model.
    """
    return HuggingFaceEmbeddings()
