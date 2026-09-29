# AI Resume Screener

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python resume screening app that uses a real RAG pipeline: resume text is chunked, embedded, and stored in ChromaDB; the job description retrieves the most relevant resume sections; Gemini then returns a structured evaluation.

This is a local Streamlit application for an AI Engineer portfolio. It is not a hosted API or production recruiting platform.

## What this project demonstrates

- Document loading and cleaning (PDF/TXT)
- Resume chunking with LangChain text splitters
- Sentence Transformer embeddings
- Vector storage and similarity search with ChromaDB
- Retrieval-augmented generation: the LLM sees retrieved resume excerpts, not an unused vector store
- Structured Gemini output (score, skills, reasoning, recommendation)
- A Streamlit screening workflow for one resume or a batch of resumes

## Architecture

```text
Resume PDF/TXT ──► extract/clean ──► chunk ──► embed ──► ChromaDB
                                                          │
Job description ──► extract/clean ──► embed query ────────┘
                                                          │
                                              retrieve top-k chunks
                                                          │
                          retrieved excerpts + job description
                                                          │
                                              Gemini structured eval
                                                          │
                         final_score = 0.4 * retrieval + 0.6 * llm
```

1. **Extract and clean** resume and job-description text. Paragraph breaks are kept so chunking still has structure.
2. **Split** the resume into overlapping chunks (`RecursiveCharacterTextSplitter`).
3. **Embed** those chunks with `all-MiniLM-L6-v2` and **upsert** them into a ChromaDB collection keyed by `resume_id`.
4. **Retrieve** the chunks most similar to the job description (`where={"resume_id": ...}`). The vector database is queried in the main screening path, not as an unused side index.
5. **Generate** a structured Gemini evaluation from the retrieved excerpts plus the job description.
6. **Score** with a documented hybrid: retrieval similarity 40%, LLM score 60%.

If Gemini/API fails, the app reports an explicit error. It does **not** treat that failure as a candidate score of `0.0`.

### Why this is RAG

The LLM is not asked to score the full raw resume in isolation while a vector database sits idle. Screening always:

- writes this candidate's chunks into ChromaDB
- queries those chunks with the job description
- passes the retrieved context into Gemini

Retrieval similarity is a supporting signal (how closely stored resume sections match the JD). The LLM score is the grounded evaluation of that context.

## Tech stack

- **Python 3.10+**
- **LangChain** — prompt + Gemini client + text splitting
- **ChromaDB** — persistent local vector store
- **Sentence Transformers** — `all-MiniLM-L6-v2` embeddings
- **Google Gemini** (`gemini-2.0-flash` by default) — structured evaluation
- **Streamlit** — UI
- **PyPDF2** — PDF text extraction

## Project structure

```text
ai-resume-screener/
├── app.py                 # Streamlit UI (single + batch)
├── main.py                # CLI screening flow
├── run_app.py             # Helper to launch Streamlit
├── rag_pipeline.py        # Index → retrieve → evaluate
├── vector_store.py        # ChromaDB embeddings and search
├── llm_evaluator.py       # Gemini structured output
├── utils.py               # PDF/text loading, cleaning, chunking
├── config.py              # Models, weights, chunk settings
├── test_setup.py          # Import + retrieval checks (no API key)
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/manojpemmadi/ai-resume-screener.git
   cd ai-resume-screener
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. Add a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey):

   ```bash
   copy .env.example .env   # Windows
   cp .env.example .env     # macOS/Linux
   ```

   Then set `GEMINI_API_KEY` in `.env`. Do not commit that file.

The embedding model (~80MB) downloads automatically on first run. ChromaDB data is stored locally in `./vector_db/` and is gitignored.

## Run

Web app:

```bash
streamlit run app.py
```

Or:

```bash
python run_app.py
```

Open `http://localhost:8501`.

CLI:

```bash
python main.py
```

Import/retrieval check (no Gemini key required):

```bash
python test_setup.py
```

## Usage

```python
from rag_pipeline import RAGResumeScreener
from utils import extract_text_from_pdf

screener = RAGResumeScreener()

resume_text = extract_text_from_pdf("path/to/resume.pdf")
result = screener.screen_resume(resume_text, job_description)

if result["status"] == "success":
    print(result["final_score"], result["recommendation"])
    print(result["llm_details"]["matched_skills"])
else:
    print("System error:", result["error"])
```

Batch screening indexes every resume, then retrieves and evaluates each one against the same job description:

```python
resumes = [
    {"id": "alice.pdf", "text": alice_text},
    {"id": "bob.pdf", "text": bob_text},
]
results = screener.batch_screen_resumes(resumes, job_description)
```

The Streamlit **Batch screening** tab does the same with multiple uploaded files.

## Output

A successful result looks like:

```json
{
  "status": "success",
  "resume_id": "resume_ab12cd34ef56",
  "final_score": 0.81,
  "retrieval_score": 0.74,
  "llm_score": 0.86,
  "recommendation": "Recommended",
  "llm_details": {
    "score": 0.86,
    "reasoning": "Retrieved excerpts show Python, embeddings, and vector search experience aligned with the role.",
    "matched_skills": ["Python", "embeddings", "vector search"],
    "missing_skills": ["production MLOps"],
    "recommendation": "Recommended"
  },
  "retrieved_chunks": [
    {
      "id": "resume_ab12cd34ef56::chunk::2",
      "text": "Built a retrieval-augmented search service...",
      "similarity": 0.78,
      "metadata": {"resume_id": "resume_ab12cd34ef56", "chunk_index": 2, "doc_type": "resume_chunk"}
    }
  ],
  "chunk_count": 6,
  "weights": {
    "retrieval": 0.4,
    "llm": 0.6
  },
  "error": null
}
```

`status: "error"` means indexing, retrieval, or Gemini failed. `final_score` is `null` in that case so a failed API call is not shown as a poor candidate.

## Configuration

Edit `config.py` (or override some values with environment variables):

| Setting | Default | Purpose |
| --- | --- | --- |
| `RETRIEVAL_WEIGHT` | `0.4` | Weight of Chroma retrieval similarity |
| `LLM_WEIGHT` | `0.6` | Weight of Gemini evaluation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model (`LLM_MODEL` env var) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `120` | Resume splitting |
| `TOP_K_CHUNKS` | `5` | Chunks retrieved per candidate |
| `VECTOR_DB_PATH` | `./vector_db` | Chroma persistence directory |

## Requirements

- Python 3.10+
- A Gemini API key
- Network access on first run to download the embedding model

## License

MIT. See [LICENSE](LICENSE).
