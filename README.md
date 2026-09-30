# AI Resume Screener

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/streamlit-1.40%2B-red)](https://streamlit.io/)

An intelligent resume screening application powered by semantic search and Google Gemini AI. It helps recruiters and hiring teams evaluate candidates against a job description by combining retrieval-based matching with LLM-based reasoning.

## Overview

AI Resume Screener processes resumes and job descriptions in a RAG-style pipeline:

- Extracts text from uploaded PDF or TXT files
- Splits resumes into overlapping chunks
- Embeds chunks with Sentence Transformers
- Stores them in ChromaDB
- Retrieves the most relevant resume sections for a job description
- Uses Google Gemini to score the match and explain the result

This creates a more transparent and explainable resume screening workflow than pure keyword filtering.

## Features

- PDF and TXT resume support
- PDF and TXT job description upload support
- Text cleaning and normalization
- Resume chunking for semantic retrieval
- Embedding generation with Sentence Transformers
- Vector storage with ChromaDB
- Top-k retrieval against a job description
- Gemini-powered scoring and recommendations
- Structured output with matched and missing skills
- Batch screening across multiple resumes
- Streamlit web UI for quick usage
- CLI mode for terminal-based evaluation

## How It Works

1. Upload a resume and a job description.
2. The app cleans both texts and splits the resume into chunks.
3. Each chunk is embedded and stored in ChromaDB.
4. The job description is embedded and used to retrieve the most relevant chunks.
5. Gemini evaluates the retrieved excerpts against the job description.
6. The final score blends:
   - Retrieval similarity score (40%)
   - LLM evaluation score (60%)
7. The app shows the final recommendation, matched skills, missing skills, and reasoning.

## Scoring Model

The final score is computed as:

final_score = retrieval_score * 0.4 + llm_score * 0.6

Where:

- retrieval_score: average cosine similarity of retrieved resume chunks
- llm_score: Gemini-based match score from 0.0 to 1.0

## Tech Stack

- Python 3.8+
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Google Gemini / Google GenAI
- PyPDF2
- Python-dotenv

## Project Structure

```text
.
├── app.py                  # Streamlit web application
├── main.py                 # CLI entry point
├── run_app.py              # Launch Streamlit app
├── rag_pipeline.py         # Core RAG screening pipeline
├── llm_evaluator.py        # Gemini evaluation logic
├── vector_store.py         # ChromaDB vector store wrapper
├── utils.py                # Text extraction, cleaning, and chunking
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
├── LICENSE                 # MIT license
├── README.md               # Project documentation
└── vector_db/              # Persistent vector database directory (generated at runtime)
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/manojpemmadi/ai-resume-screener.git
cd ai-resume-screener
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

Create a `.env` file in the project root using the example:

```bash
cp .env.example .env
```

Then update the file:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

You may also configure the model if needed:

```dotenv
LLM_MODEL=gemini-2.0-flash
VECTOR_DB_PATH=./vector_db
```

## Run the App

### Streamlit web app

```bash
python run_app.py
```

Or directly:

```bash
streamlit run app.py
```

### Command-line mode

```bash
python main.py
```

The CLI prompts for:

- Job description
- Resume text

Then it prints the screening result and saves it to `screening_result.json`.

## Usage

### Single resume screening

- Open the app in a browser
- Choose between uploading files or pasting text for the job description and resume
- Click "Screen resume"
- Review:
  - final score
  - retrieval score
  - LLM score
  - recommendation
  - matched and missing skills
  - reasoning
  - retrieved excerpts

### Batch screening

- Open the "Batch screening" tab
- Upload one or more resumes and either paste or upload a job description
- Click "Screen batch"
- Review the ranked results table and detailed per-candidate evaluation

## Configuration

The main settings are in `config.py`:

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
RETRIEVAL_WEIGHT = 0.4
LLM_WEIGHT = 0.6
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
COLLECTION_NAME = "resume_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
TEMPERATURE = 0.2
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K_CHUNKS = 5
```

These values can be tuned depending on the size of the resumes and the desired strictness of matching.

## Important Notes

- The vector store persists to `./vector_db` by default.
- The app expects a valid Google Gemini API key.
- If a resume or job description is empty after cleaning, the app raises a validation error.
- The system uses only retrieved excerpts for the LLM evaluation, which helps ground decisions in candidate evidence rather than memory alone.

## Example Output

A typical result includes:

- final_score
- retrieval_score
- llm_score
- weights
- recommendation
- llm_details with reasoning, matched_skills, missing_skills
- retrieved_chunks with similarity values

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. If you want to enhance the app, consider improving:

- retrieval quality
- scoring calibration
- resume parsing for more file types
- better UI for recruiter workflows
- batch export/reporting

## Troubleshooting

### Gemini API errors

- Ensure `GEMINI_API_KEY` is set in `.env`
- Verify the API key is valid and has access
- Confirm the selected model name is available in your environment

### Missing packages

```bash
pip install -r requirements.txt
```

### ChromaDB or embeddings issues

Delete the persistent database if needed:

```bash
rm -rf vector_db
```

Then rerun the app and it will recreate the collection.

## Summary

AI Resume Screener is a practical resume-ranking application that combines semantic retrieval with LLM-based assessment. It is designed for quick candidate evaluation and transparent scoring, making it suitable for recruiters, HR teams, and AI-assisted hiring workflows.


