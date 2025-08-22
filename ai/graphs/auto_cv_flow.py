from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END
from ai.tools.file_parser_tool import parse_file
from ai.agents.resume_agent import create_resume_summary
from ai.agents.job_description_agent import create_job_summary
from ai.agents.cv_writer_agent import create_tailored_cv

class AutoCVState(TypedDict):
    resume_path: str
    job_description: str
    resume_content: Optional[str]
    resume_summary: Optional[Any]
    job_summary: Optional[Any]
    tailored_cv: Optional[str]

def ingest_resume_node(state: AutoCVState):
    """
    Parses the resume file and updates the state.
    """
    docs = parse_file(state["resume_path"])
    resume_content = "\n".join([doc.page_content for doc in docs])
    return {"resume_content": resume_content}

def ingest_job_description_node(state: AutoCVState):
    """
    This node is a placeholder, as the job description is passed in the initial state.
    In a real-world scenario, this might involve fetching the job description from a URL.
    """
    return {}

def summarize_resume_node(state: AutoCVState):
    """
    Summarizes the resume and updates the state.
    """
    summary = create_resume_summary(state["resume_content"])
    return {"resume_summary": summary}

def summarize_job_description_node(state: AutoCVState):
    """
    Summarizes the job description and updates the state.
    """
    summary = create_job_summary(state["job_description"])
    return {"job_summary": summary}

def generate_cv_node(state: AutoCVState):
    """
    Generates the tailored CV and updates the state.
    """
    cv = create_tailored_cv(state["resume_summary"], state["job_summary"])
    return {"tailored_cv": cv}

def create_auto_cv_graph():
    """
    Creates the LangGraph for the auto CV generation flow.
    """
    graph = StateGraph(AutoCVState)

    graph.add_node("ingest_resume", ingest_resume_node)
    graph.add_node("ingest_job_description", ingest_job_description_node)
    graph.add_node("summarize_resume", summarize_resume_node)
    graph.add_node("summarize_job_description", summarize_job_description_node)
    graph.add_node("generate_cv", generate_cv_node)

    graph.set_entry_point("ingest_resume")
    graph.add_edge("ingest_resume", "ingest_job_description")
    graph.add_edge("ingest_job_description", "summarize_resume")
    graph.add_edge("summarize_resume", "summarize_job_description")
    graph.add_edge("summarize_job_description", "generate_cv")
    graph.add_edge("generate_cv", END)

    return graph.compile()

if __name__ == '__main__':
    # Example usage:
    # This demonstrates how to run the graph.
    # In a real application, this would be triggered by a FastAPI endpoint.

    # 1. Define the inputs
    resume_path = "path/to/your/resume.pdf"  # Replace with a real path
    job_description = "We are looking for a software engineer with experience in Python and LangChain."

    # Create a dummy resume file for testing
    with open("dummy_resume.txt", "w") as f:
        f.write("John Doe\nSoftware Engineer\n- 5 years of experience in Python\n- 2 years of experience with LangChain")

    resume_path = "dummy_resume.txt"

    # 2. Compile the graph
    app = create_auto_cv_graph()

    # 3. Define the initial state
    initial_state = {
        "resume_path": resume_path,
        "job_description": job_description,
    }

    # 4. Run the graph
    # Note: This will make real calls to the OpenAI API if you have set up your API key.
    # final_state = app.invoke(initial_state)

    # 5. Print the final CV
    # print(final_state.get("tailored_cv"))
    print("Graph created successfully. To run, uncomment the lines in the if __name__ == '__main__': block and provide a valid resume file and OpenAI API key.")
