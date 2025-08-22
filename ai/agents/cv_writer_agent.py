import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='postgres_FastAPI/.env')

class TailoredCV(BaseModel):
    """A tailored CV in markdown format."""
    cv_markdown: str = Field(description="The tailored CV in markdown format.")

def create_tailored_cv(resume_summary: Dict[str, Any], job_summary: Dict[str, Any]) -> str:
    """
    Uses an LLM to generate a tailored CV based on a resume and job description.

    Args:
        resume_summary: A structured summary of the resume.
        job_summary: A structured summary of the job description.

    Returns:
        The tailored CV in markdown format.
    """
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature=0.2, model="gpt-4")

    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
You are an expert CV writer. Your task is to create a tailored CV in markdown format.
You will be given a resume summary and a job description summary.
Your goal is to rewrite the resume to perfectly match the job description.

Instructions:
1.  Start with the candidate's summary and tailor it to the job title and company.
2.  Highlight the most relevant work experience. For each role, rewrite the description to emphasize the skills and achievements that are most relevant to the new job's responsibilities and required skills. Use action verbs and quantify achievements where possible.
3.  List the skills from the resume that are most relevant to the job.
4.  Include the education section.
5.  The final output should be a single markdown string, professionally formatted.
"""),
            ("user", """
Here is the resume summary:
{resume_summary}

Here is the job description summary:
{job_summary}

Now, please generate the tailored CV in markdown format.
"""),
        ]
    )

    # Create the chain
    chain = prompt_template | llm

    # Invoke the chain
    response = chain.invoke({
        "resume_summary": resume_summary,
        "job_summary": job_summary,
    })

    return response.content
