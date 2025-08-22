from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM

def create_resume_summary_chain(llm: BaseLLM) -> LLMChain:
    """
    Creates a LangChain chain that summarizes a resume into a structured format.

    Args:
        llm: The language model to use for the chain.

    Returns:
        An LLMChain that takes resume_text as input and returns a structured summary.
    """
    prompt_template = """
You are an expert resume parser and summarizer. Your task is to analyze the following resume text and extract the key information in a structured and concise format.

Resume Text:
---
{resume_text}
---

Please extract the information and format your output as follows. If a section is not found, omit it.

**Summary:**
A brief, 2-3 sentence summary of the candidate's professional profile.

**Work Experience:**
- **[Job Title] at [Company Name]** ([Start Date] - [End Date])
  - Key responsibility or achievement 1.
  - Key responsibility or achievement 2.
- ...

**Skills:**
- Technical Skills: [List of technical skills]
- Soft Skills: [List of soft skills]

**Education:**
- **[Degree] in [Major]** - [University Name] ([Graduation Year])
- ...
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["resume_text"]
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_key="resume_summary")
    return chain
