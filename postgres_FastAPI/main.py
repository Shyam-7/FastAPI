from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Annotated
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session

app = FastAPI()
models.Base.metadata.create_all(bind=engine)

class ChoiceBase(BaseModel):
    choice_text: str
    is_correct: bool
    
class QuestionBase(BaseModel):
    question_text: str
    choices: List[ChoiceBase]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
@app.get("/questions/{question_id}")
async def read_question(question_id: int, db: db_dependency):
    result = db.query(models.Questions).filter(models.Questions.id == question_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Question not found")
    return result

@app.get("/choices/{question_id}")
async def read_choices(question_id: int, db: db_dependency):
    result = db.query(models.Choices).filter(models.Choices.question_id == question_id).all()
    if not result:
           raise HTTPException(status_code=404, detail="Choices not found")
    return result

@app.post("/questions/")
async def create_question(question: QuestionBase, db: db_dependency): # type: ignore
    db_question = models.Questions(question_text=question.question_text)
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    for choice in question.choices:
        db_choice = models.Choices(choice_text=choice.choice_text, is_correct=choice.is_correct, question_id = db_question.id)
        db.add(db_choice)
    db.commit()

# --- New imports for AI CV Generation ---
from fastapi import File, UploadFile, Form
from fastapi.responses import StreamingResponse
import io
import os
from docx import Document
from ai.graphs.auto_cv_flow import create_auto_cv_graph
from ai.tools.file_parser_tool import FileParserTool
from langchain_openai import OpenAI

# --- AI Model and Graph Initialization ---
# In a production app, use a proper config and secrets management.
# You should also add OPENAI_API_KEY to your environment variables.
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cv_graph = create_auto_cv_graph(llm)
file_parser = FileParserTool()

# --- New Endpoint for CV Generation ---
@app.post("/generate-cv/")
async def generate_cv_endpoint(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    Accepts a resume file and a job description, then returns a tailored CV
    as a downloadable DOCX file.
    """
    # Use a temporary directory for file uploads
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    # Sanitize filename to prevent directory traversal issues
    safe_filename = os.path.basename(resume_file.filename)
    temp_resume_path = os.path.join(temp_dir, safe_filename)

    try:
        with open(temp_resume_path, "wb") as buffer:
            buffer.write(await resume_file.read())

        # Parse the resume file to get text
        resume_text = file_parser._run(temp_resume_path)

        if resume_text.startswith("Error:"):
            raise HTTPException(status_code=400, detail=resume_text)

        # Run the LangGraph flow
        inputs = {
            "resume_text": resume_text,
            "job_description_text": job_description
        }
        result = cv_graph.invoke(inputs)
        generated_cv_text = result.get("generated_cv")

        if not generated_cv_text:
            raise HTTPException(status_code=500, detail="Failed to generate CV. The AI model may have returned an empty response.")

        # Create a DOCX file in memory
        document = Document()
        document.add_paragraph(generated_cv_text)

        file_stream = io.BytesIO()
        document.save(file_stream)
        file_stream.seek(0)

        # Return the DOCX file as a response
        return StreamingResponse(
            file_stream,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename=Generated_CV_for_{safe_filename}.docx"}
        )
    finally:
        # Clean up the temp file
        if os.path.exists(temp_resume_path):
            os.remove(temp_resume_path)
