from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM

def create_cv_writing_chain(llm: BaseLLM) -> LLMChain:
    """
    Creates a LangChain chain that generates a tailored CV by matching a resume
    to a job description.

    Args:
        llm: The language model to use for the chain.

    Returns:
        An LLMChain that takes resume_summary, job_description_summary, and
        original_resume as input, and returns the generated_cv text.
    """
    prompt_template = """
You are an expert professional CV writer and career coach. Your task is to create a new, tailored CV for a candidate by strategically aligning their existing resume with a specific job description.

**1. Target Job Requirements (from Job Description):**
---
{job_description_summary}
---

**2. Candidate's Profile (from Resume):**
---
{resume_summary}
---

**3. Candidate's Full Original Resume (for context and detail):**
---
{original_resume}
---

**Instructions:**
Based on all the information provided, generate a complete, new CV in plain text format. The new CV must:
- **Rewrite the Professional Summary:** Create a new, compelling summary that directly mirrors the language and key requirements of the job description, using the candidate's experience.
- **Tailor Work Experience:** For each role in the candidate's history, rephrase the bullet points to highlight achievements and responsibilities that are most relevant to the target job's responsibilities and required skills. Use strong action verbs.
- **Emphasize Key Skills:** Ensure that skills mentioned in the job description are prominently featured in the CV's skills section if they are present in the candidate's original resume.
- **Maintain Professional Format:** The output should be a clean, well-formatted, and professional CV.

**Important:** Do not invent any new skills or experiences. All content in the generated CV must be derived from the candidate's original resume.

**Generated CV:**
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["job_description_summary", "resume_summary", "original_resume"]
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_key="generated_cv")
    return chain
