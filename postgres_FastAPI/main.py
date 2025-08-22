from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Annotated
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import shutil
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.graphs.auto_cv_flow import create_auto_cv_graph

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

@app.post("/generate-cv/")
async def generate_cv_endpoint(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    This endpoint generates a tailored CV.
    It accepts a resume file and a job description.
    """
    # Create a temporary directory to store the uploaded file
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)

    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save the uploaded file to a temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Compile the graph
        app_graph = create_auto_cv_graph()

        # 2. Define the initial state
        initial_state = {
            "resume_path": temp_file_path,
            "job_description": job_description,
        }

        # 3. Run the graph
        final_state = app_graph.invoke(initial_state)

        # 4. Return the tailored CV
        return {"tailored_cv": final_state.get("tailored_cv")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
