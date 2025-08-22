import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='postgres_FastAPI/.env')

class JobDescriptionSummary(BaseModel):
    """Structured data for a job description summary."""
    job_title: str = Field(description="The title of the job.")
    company: str = Field(description="The name of the company.")
    key_responsibilities: list = Field(description="A list of key responsibilities for the role.")
    required_skills: list = Field(description="A list of required skills for the role.")

def create_job_summary(job_description: str) -> Dict[str, Any]:
    """
    Uses an LLM to create a structured summary of a job description.

    Args:
        job_description: The text content of the job description.

    Returns:
        A dictionary containing the structured job description summary.
    """
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at analyzing job descriptions. Extract the following information from the provided job description and format it as a JSON object: job_title, company, key_responsibilities, and required_skills."),
            ("user", "{job_description}"),
        ]
    )

    # Create the chain with a JSON output parser
    parser = "json_object"
    chain = prompt_template | llm.with_structured_output(parser)

    # Invoke the chain with the job description
    response = chain.invoke({"job_description": job_description})

    return response
