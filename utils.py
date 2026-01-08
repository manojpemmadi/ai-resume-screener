"""
Utility functions for text processing and PDF parsing
"""
import PyPDF2
from typing import Optional


def extract_text_from_pdf(pdf_input) -> Optional[str]:
    """
    Extract text from a PDF file or file-like object
    
    Args:
        pdf_input: Path to PDF file (str) or file-like object (Streamlit upload)
        
    Returns:
        Extracted text as string, or None if error
    """
    try:
        text = ""
        # Handle both file paths and file-like objects (Streamlit uploads)
        if isinstance(pdf_input, str):
            # File path
            with open(pdf_input, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        else:
            # File-like object (Streamlit upload)
            pdf_reader = PyPDF2.PdfReader(pdf_input)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def read_text_file(file_path: str) -> Optional[str]:
    """
    Read text from a file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File contents as string, or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None
