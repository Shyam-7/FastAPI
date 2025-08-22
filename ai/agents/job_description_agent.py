from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM

def create_job_description_analysis_chain(llm: BaseLLM) -> LLMChain:
    """
    Creates a LangChain chain that analyzes a job description to extract key info.

    Args:
        llm: The language model to use for the chain.

    Returns:
        An LLMChain that takes job_description_text as input and returns a
        structured analysis.
    """
    prompt_template = """
You are an expert HR analyst specializing in parsing job descriptions. Your task is to carefully read the following job description and extract the most critical information that a candidate should address in their CV.

Job Description:
---
{job_description_text}
---

Please extract the information and format your output as a concise summary covering these key areas:

**Core Responsibilities:**
- A bulleted list of the top 3-5 most important responsibilities or duties.

**Essential Skills & Technologies:**
- A bulleted list of the mandatory skills, programming languages, and technologies mentioned.

**Experience & Qualifications:**
- A bulleted list of the key qualifications, such as years of experience, educational background, or specific certifications required.
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["job_description_text"]
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_key="job_description_summary")
    return chain
