# AI Resume Screener

A Python application that evaluates resumes against a job description using semantic search and Google Gemini.

## Features

- Upload PDF/TXT resumes
- Extract and clean resume text
- Split resumes into smaller sections
- Generate resume embeddings
- Store embeddings in ChromaDB
- Retrieve relevant resume sections based on the job description
- Use Gemini to evaluate candidate-job matching
- Get score, matched skills, missing skills, reasoning, and recommendation
- Support single and batch resume screening

## How It Works

Resume → Text Extraction → Chunking → Embeddings → ChromaDB  
Job Description → Semantic Search → Relevant Resume Sections → Gemini → Evaluation

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Google Gemini
- PyPDF2

## Project Structure

```text
ai-resume-screener/
├── app.py
├── main.py
├── rag_pipeline.py
├── vector_store.py
├── llm_evaluator.py
├── utils.py
├── config.py
├── requirements.txt
└── .env.example
