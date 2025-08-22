import os
from typing import List
import docx2txt
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

def parse_file(file_path: str) -> List[Document]:
    """
    Parses a file and returns its content as a list of LangChain Documents.
    Supports .pdf, .txt, and .docx files.
    """
    _, extension = os.path.splitext(file_path)
    if extension.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif extension.lower() == ".txt":
        loader = TextLoader(file_path)
        return loader.load()
    elif extension.lower() == ".docx":
        text = docx2txt.process(file_path)
        return [Document(page_content=text, metadata={"source": file_path})]
    else:
        raise ValueError(f"Unsupported file type: {extension}")
