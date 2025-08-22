import unittest
from unittest.mock import patch, MagicMock
import os
from ai.tools.file_parser_tool import FileParserTool

class TestFileParserTool(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        self.parser = FileParserTool()
        self.test_dir = "test_assets"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Tear down after the tests."""
        for f in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f))
        os.rmdir(self.test_dir)

    def test_parse_txt_file(self):
        """Test parsing a .txt file."""
        file_path = os.path.join(self.test_dir, "test.txt")
        expected_content = "This is a test text file."
        with open(file_path, "w") as f:
            f.write(expected_content)

        result = self.parser._run(file_path)
        self.assertEqual(result, expected_content)

    @patch('pypdf.PdfReader')
    def test_parse_pdf_file(self, mock_pdf_reader):
        """Test parsing a .pdf file using a mock."""
        # Arrange
        file_path = os.path.join(self.test_dir, "test.pdf")
        with open(file_path, "w") as f: # create a dummy file to exist
            f.write("")

        expected_content = "This is a test PDF content."
        mock_page = MagicMock()
        mock_page.extract_text.return_value = expected_content
        mock_instance = mock_pdf_reader.return_value
        mock_instance.pages = [mock_page]

        # Act
        result = self.parser._run(file_path)

        # Assert
        self.assertEqual(result.strip(), expected_content)

    def test_file_not_found(self):
        """Test error handling for a file that does not exist."""
        file_path = "non_existent_file.txt"
        result = self.parser._run(file_path)
        self.assertTrue(result.startswith("Error: File not found"))

    def test_unsupported_file_type(self):
        """Test error handling for an unsupported file type."""
        file_path = os.path.join(self.test_dir, "test.docx")
        with open(file_path, "w") as f:
            f.write("dummy content")

        result = self.parser._run(file_path)
        self.assertTrue(result.startswith("Error: Unsupported file type"))

if __name__ == '__main__':
    unittest.main()
