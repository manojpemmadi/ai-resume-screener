"""
Text extraction, cleaning, and resume chunking.
"""
from typing import List, Optional

import PyPDF2

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # Older LangChain layouts
    from langchain.text_splitter import RecursiveCharacterTextSplitter

import config


def extract_text_from_pdf(pdf_input) -> Optional[str]:
    """
    Extract text from a PDF path or a file-like object (e.g. Streamlit upload).
    Returns None if extraction fails.
    """
    try:
        pages = []
        if isinstance(pdf_input, str):
            with open(pdf_input, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
        else:
            reader = PyPDF2.PdfReader(pdf_input)
            for page in reader.pages:
                pages.append(page.extract_text() or "")
        text = "\n".join(pages).strip()
        return text or None
    except Exception as exc:
        print(f"Error extracting text from PDF: {exc}")
        return None


def clean_text(text: str) -> str:
    """
    Normalize extracted text while keeping paragraph structure for chunking.
    """
    if not text:
        return ""

    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = []
    previous_blank = False
    for raw_line in text.split("\n"):
        line = " ".join(raw_line.split())
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        previous_blank = False
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def split_resume_into_chunks(
    resume_text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """
    Split a resume into overlapping chunks so retrieval can return
    the most relevant sections instead of one oversized embedding.
    """
    text = clean_text(resume_text)
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
    return chunks


def read_text_file(file_path: str) -> Optional[str]:
    """Read a UTF-8 text file. Returns None on error."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as exc:
        print(f"Error reading file: {exc}")
        return None


def load_text_from_upload(uploaded_file) -> Optional[str]:
    """Load text from a Streamlit UploadedFile (PDF or TXT)."""
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    name = (uploaded_file.name or "").lower()
    if uploaded_file.type == "application/pdf" or name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    try:
        return uploaded_file.read().decode("utf-8")
    except Exception as exc:
        print(f"Error reading uploaded file: {exc}")
        return None
