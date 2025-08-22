# Auto CV Generation Flow

This directory contains a LangGraph-based flow for automatically generating a tailored CV based on a resume and a job description.

## Overview

The flow takes a user's resume (in PDF, DOCX, or TXT format) and a job description as input. It then uses a series of LLM-powered agents to:

1.  Parse and summarize the resume.
2.  Analyze the job description to extract key requirements.
3.  Generate a new CV in markdown format that is tailored to the specific job.

## File Structure

-   `agents/`: Contains the core logic for the different AI agents.
    -   `resume_agent.py`: Summarizes the resume.
    -   `job_description_agent.py`: Analyzes the job description.
    -   `cv_writer_agent.py`: Generates the tailored CV.
-   `graphs/`: Contains the LangGraph flow.
    -   `auto_cv_flow.py`: Defines the graph that connects the agents and tools.
-   `tools/`: Contains utility tools.
    -   `file_parser_tool.py`: Parses uploaded files.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r ../postgres_FastAPI/requirements.txt
    ```

2.  **Set up your environment variables:**
    Create a `.env` file in the `postgres_FastAPI` directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## How to Run

The CV generation flow is exposed as a FastAPI endpoint.

### Endpoint: `POST /generate-cv/`

This endpoint accepts a multipart/form-data request with the following fields:

-   `file`: The resume file (PDF, DOCX, or TXT).
-   `job_description`: A string containing the job description.

**Example using `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/generate-cv/" \
-F "file=@/path/to/your/resume.pdf" \
-F "job_description=We are looking for a software engineer..."
```

The endpoint will return a JSON response with the tailored CV:

```json
{
  "tailored_cv": "# Your Tailored CV in Markdown..."
}
```

### Running the FastAPI Server

To run the FastAPI server, navigate to the `postgres_FastAPI` directory and run:

```bash
uvicorn main:app --reload
```
