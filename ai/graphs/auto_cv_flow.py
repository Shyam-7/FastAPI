from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.language_models.llms import BaseLLM
import operator

from ..agents.resume_agent import create_resume_summary_chain
from ..agents.job_description_agent import create_job_description_analysis_chain
from ..agents.cv_writer_agent import create_cv_writing_chain

class AutoCVState(TypedDict):
    """
    Represents the state of the auto CV generation graph.
    """
    resume_text: str
    job_description_text: str
    resume_summary: str
    job_description_summary: str
    generated_cv: str

def create_auto_cv_graph(llm: BaseLLM):
    """
    Creates and compiles the LangGraph for the auto CV generation flow.

    Args:
        llm: The language model to be used by the agents.

    Returns:
        A compiled LangGraph application.
    """
    # 1. Instantiate the agent chains
    resume_chain = create_resume_summary_chain(llm)
    jd_chain = create_job_description_analysis_chain(llm)
    cv_chain = create_cv_writing_chain(llm)

    # 2. Define the graph nodes
    def summarize_resume(state: AutoCVState) -> dict:
        """Node to summarize the resume."""
        print("---SUMMARIZING RESUME---")
        resume_text = state["resume_text"]
        response = resume_chain.invoke({"resume_text": resume_text})
        return {"resume_summary": response["resume_summary"]}

    def analyze_job_description(state: AutoCVState) -> dict:
        """Node to analyze the job description."""
        print("---ANALYZING JOB DESCRIPTION---")
        job_description_text = state["job_description_text"]
        response = jd_chain.invoke({"job_description_text": job_description_text})
        return {"job_description_summary": response["job_description_summary"]}

    def generate_cv(state: AutoCVState) -> dict:
        """Node to generate the tailored CV."""
        print("---GENERATING CV---")
        response = cv_chain.invoke({
            "resume_summary": state["resume_summary"],
            "job_description_summary": state["job_description_summary"],
            "original_resume": state["resume_text"]
        })
        return {"generated_cv": response["generated_cv"]}

    # 3. Build the graph
    workflow = StateGraph(AutoCVState)

    # Add nodes
    workflow.add_node("summarize_resume", summarize_resume)
    workflow.add_node("analyze_job_description", analyze_job_description)
    workflow.add_node("generate_cv", generate_cv)

    # Define edges for a sequential flow
    # While resume and JD analysis could be parallel, a sequential flow is simpler to start.
    workflow.set_entry_point("summarize_resume")
    workflow.add_edge("summarize_resume", "analyze_job_description")
    workflow.add_edge("analyze_job_description", "generate_cv")
    workflow.add_edge("generate_cv", END)

    # 4. Compile the graph
    app = workflow.compile()
    return app
