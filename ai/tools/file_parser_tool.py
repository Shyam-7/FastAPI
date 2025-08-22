from langchain.tools import BaseTool
import pypdf
from typing import Type, Optional
from pydantic import BaseModel, Field
import os

class FileParserInput(BaseModel):
    """Input for the FileParserTool."""
    file_path: str = Field(description="The path to the file to be parsed.")

class FileParserTool(BaseTool):
    """A tool to parse text from PDF and TXT files."""
    name: str = "file_parser"
    description: str = "Use this tool to parse text content from a file. It supports .pdf and .txt files."
    args_schema: Type[BaseModel] = FileParserInput

    def _run(self, file_path: str) -> str:
        """Use the tool."""
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"

        _, extension = os.path.splitext(file_path)

        if extension.lower() == ".pdf":
            return self._parse_pdf(file_path)
        elif extension.lower() == ".txt":
            return self._parse_txt(file_path)
        else:
            return f"Error: Unsupported file type '{extension}'. Only .pdf and .txt files are supported."

    def _parse_pdf(self, file_path: str) -> str:
        """Parses text from a PDF file."""
        try:
            reader = pypdf.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            return f"Error parsing PDF file: {e}"

    def _parse_txt(self, file_path: str) -> str:
        """Parses text from a TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading TXT file: {e}"
