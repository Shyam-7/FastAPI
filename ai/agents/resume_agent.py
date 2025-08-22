import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='postgres_FastAPI/.env')

class ResumeSummary(BaseModel):
    """Structured data for a resume summary."""
    summary: str = Field(description="A concise summary of the candidate's profile.")
    work_experience: list = Field(description="A list of professional experiences.")
    education: list = Field(description="A list of educational qualifications.")
    skills: list = Field(description="A list of key skills.")

def create_resume_summary(resume_content: str) -> Dict[str, Any]:
    """
    Uses an LLM to create a structured summary of a resume.

    Args:
        resume_content: The text content of the resume.

    Returns:
        A dictionary containing the structured resume summary.
    """
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at summarizing resumes. Extract the following information from the provided resume content and format it as a JSON object: summary, work_experience, education, and skills."),
            ("user", "{resume_content}"),
        ]
    )

    # Create the chain with a JSON output parser
    parser = "json_object"
    chain = prompt_template | llm.with_structured_output(parser)

    # Invoke the chain with the resume content
    response = chain.invoke({"resume_content": resume_content})

    return response
