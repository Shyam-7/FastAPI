import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import io
import os

# Set an environment variable for the tests
os.environ['OPENAI_API_KEY'] = 'test_key'

from postgres_FastAPI.main import app

class TestFastAPIEndpoints(unittest.TestCase):

    def setUp(self):
        """Set up the test client."""
        self.client = TestClient(app)

    @patch('postgres_FastAPI.main.file_parser._run')
    @patch('postgres_FastAPI.main.cv_graph.invoke')
    def test_generate_cv_success(self, mock_graph_invoke, mock_parser_run):
        """Test the /generate-cv/ endpoint for a successful case."""
        # Arrange
        mock_parser_run.return_value = "This is the resume text."
        mock_graph_invoke.return_value = {"generated_cv": "This is the generated CV."}

        job_description = "We are looking for a great developer."
        resume_content = b"dummy resume content"

        # Act
        response = self.client.post(
            "/generate-cv/",
            files={"resume_file": ("resume.txt", io.BytesIO(resume_content), "text/plain")},
            data={"job_description": job_description}
        )

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        self.assertTrue(response.headers['content-disposition'].startswith('attachment; filename=Generated_CV_for_'))

    @patch('postgres_FastAPI.main.file_parser._run')
    def test_generate_cv_parser_error(self, mock_parser_run):
        """Test the /generate-cv/ endpoint when the file parser returns an error."""
        # Arrange
        mock_parser_run.return_value = "Error: Unsupported file type"

        job_description = "A job description."
        resume_content = b"dummy content"

        # Act
        response = self.client.post(
            "/generate-cv/",
            files={"resume_file": ("resume.unsupported", io.BytesIO(resume_content), "application/octet-stream")},
            data={"job_description": job_description}
        )

        # Assert
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])

    @patch('postgres_FastAPI.main.file_parser._run')
    @patch('postgres_FastAPI.main.cv_graph.invoke')
    def test_generate_cv_graph_failure(self, mock_graph_invoke, mock_parser_run):
        """Test the /generate-cv/ endpoint when the graph fails to generate a CV."""
        # Arrange
        mock_parser_run.return_value = "This is the resume text."
        mock_graph_invoke.return_value = {"generated_cv": None} # Simulate graph failure

        job_description = "A job description."
        resume_content = b"dummy content"

        # Act
        response = self.client.post(
            "/generate-cv/",
            files={"resume_file": ("resume.txt", io.BytesIO(resume_content), "text/plain")},
            data={"job_description": job_description}
        )

        # Assert
        self.assertEqual(response.status_code, 500)
        self.assertIn("Failed to generate CV", response.json()["detail"])

if __name__ == '__main__':
    unittest.main()
