import unittest
from langchain_core.language_models.llms import FakeLLM
from ai.agents.resume_agent import create_resume_summary_chain
from ai.agents.job_description_agent import create_job_description_analysis_chain
from ai.agents.cv_writer_agent import create_cv_writing_chain

class TestAIAgents(unittest.TestCase):

    def test_resume_summary_chain(self):
        """Test the resume summary chain with a fake LLM."""
        # Arrange
        expected_summary = "This is a resume summary."
        queries = {"resume_text": "A long resume..."}
        responses = {"resume_summary": expected_summary}
        fake_llm = FakeLLM(queries=queries, responses=responses)

        chain = create_resume_summary_chain(llm=fake_llm)

        # Act
        result = chain.invoke({"resume_text": "A long resume..."})

        # Assert
        self.assertEqual(result["resume_summary"], expected_summary)

    def test_job_description_analysis_chain(self):
        """Test the job description analysis chain with a fake LLM."""
        # Arrange
        expected_analysis = "This is a job description analysis."
        queries = {"job_description_text": "A long job description..."}
        responses = {"job_description_summary": expected_analysis}
        fake_llm = FakeLLM(queries=queries, responses=responses)

        chain = create_job_description_analysis_chain(llm=fake_llm)

        # Act
        result = chain.invoke({"job_description_text": "A long job description..."})

        # Assert
        self.assertEqual(result["job_description_summary"], expected_analysis)

    def test_cv_writing_chain(self):
        """Test the CV writing chain with a fake LLM."""
        # Arrange
        expected_cv = "This is a generated CV."
        queries = {
            "resume_summary": "Resume summary",
            "job_description_summary": "JD summary",
            "original_resume": "Original resume text"
        }
        responses = {"generated_cv": expected_cv}
        fake_llm = FakeLLM(queries=queries, responses=responses)

        chain = create_cv_writing_chain(llm=fake_llm)

        # Act
        result = chain.invoke({
            "resume_summary": "Resume summary",
            "job_description_summary": "JD summary",
            "original_resume": "Original resume text"
        })

        # Assert
        self.assertEqual(result["generated_cv"], expected_cv)

if __name__ == '__main__':
    unittest.main()
