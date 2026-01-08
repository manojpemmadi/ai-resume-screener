# AI Resume Screener

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent resume screening system that uses Retrieval-Augmented Generation (RAG) to automatically evaluate candidate resumes against job descriptions. The system combines semantic search (60% weight) and LLM-based evaluation (40% weight) using Google's Gemini model.

## 🚀 Features

- **🌐 Web Interface**: Beautiful Streamlit-based web application for easy use
- **📄 File Upload**: Upload PDF or text files for job descriptions and resumes
- **Semantic Search (60% weight)**: Uses vector embeddings to find semantic similarities between resumes and job descriptions
- **LLM Evaluation (40% weight)**: Leverages Google Gemini for context-aware matching and detailed analysis
- **RAG Pipeline**: Combines retrieval and generation for accurate candidate assessment
- **Batch Processing**: Screen multiple resumes at once (via API)
- **Detailed Analysis**: Provides matched skills, missing skills, and reasoning
- **PDF Support**: Extract and process text from PDF resumes
- **Visualizations**: Interactive charts showing score breakdowns


## Tech Stack

- **Python**: Core programming language
- **LangChain**: Framework for LLM applications
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: For generating embeddings
- **Google Gemini**: LLM for intelligent evaluation
- **NLP**: Natural language processing for text analysis

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ai-resume-screener.git
   cd ai-resume-screener
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Gemini API Key**:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root:
     ```bash
     echo "GEMINI_API_KEY=your_api_key_here" > .env
     ```
   - Edit `.env` and add your Gemini API key

## 🎯 Quick Start

### Option 1: Web Application (Recommended)

Launch the interactive web interface:

```bash
# Install Streamlit (if not already installed)
pip install streamlit

# Run the web app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: Command Line Script

Run the CLI script:

```bash
python main.py
```

## 📖 Usage

### Basic Usage

```python
from rag_pipeline import RAGResumeScreener

# Initialize the screener
screener = RAGResumeScreener()

# Screen the resume
result = screener.screen_resume(resume_text, job_description)

# View results
print(f"Final Score: {result['final_score']}")
print(f"Recommendation: {result['recommendation']}")
print(f"Semantic Score: {result['semantic_score']}")
print(f"LLM Score: {result['llm_score']}")
```


### Batch Screening

```python
# Screen multiple resumes
resumes = [
    {'id': 'resume1', 'text': '...'},
    {'id': 'resume2', 'text': '...'},
]

results = screener.batch_screen_resumes(resumes, job_description)
# Results are sorted by final_score (highest first)
```

### Using PDF Files

```python
from utils import extract_text_from_pdf

# Extract text from PDF resume
resume_text = extract_text_from_pdf('path/to/resume.pdf')
result = screener.screen_resume(resume_text, job_description)
```

## Configuration

Edit `config.py` to customize:

- **Weights**: Adjust `SEMANTIC_SEARCH_WEIGHT` and `LLM_WEIGHT`
- **Embedding Model**: Change `EMBEDDING_MODEL` for different embeddings
- **LLM Model**: Modify `LLM_MODEL` (default: "gemini-pro")
- **Temperature**: Adjust `TEMPERATURE` for LLM consistency

## How It Works

1. **Semantic Search (60%)**:
   - Converts resume and job description to vector embeddings
   - Calculates cosine similarity between embeddings
   - Provides a similarity score (0-1)

2. **LLM Evaluation (40%)**:
   - Uses Gemini to analyze resume-job match
   - Evaluates skills, experience, and qualifications
   - Provides detailed reasoning and skill analysis

3. **Final Score**:
   - Weighted combination: `final_score = semantic_score * 0.6 + llm_score * 0.4`
   - Generates recommendation based on final score

## Output Format

The screening result includes:

```json
{
  "final_score": 0.85,
  "semantic_score": 0.82,
  "llm_score": 0.89,
  "recommendation": "Strongly Recommended - Excellent match",
  "llm_details": {
    "score": 0.89,
    "reasoning": "Strong match with required skills...",
    "matched_skills": ["Python", "NLP", "Machine Learning"],
    "missing_skills": ["Docker"]
  },
  "weights": {
    "semantic_search": 0.6,
    "llm": 0.4
  }
}
```

## 📁 Project Structure

```
ai-resume-screener/
├── app.py                 # Streamlit web application
├── config.py              # Configuration settings
├── vector_store.py        # Vector database and semantic search
├── llm_evaluator.py       # Gemini LLM evaluation
├── rag_pipeline.py        # Main RAG pipeline
├── main.py                # CLI script
├── utils.py               # Utility functions (PDF parsing, etc.)
├── run_app.py             # Quick launcher for web app
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Weights**: Adjust `SEMANTIC_SEARCH_WEIGHT` (default: 0.6) and `LLM_WEIGHT` (default: 0.4)
- **Embedding Model**: Change `EMBEDDING_MODEL` for different embeddings (default: "all-MiniLM-L6-v2")
- **LLM Model**: Modify `LLM_MODEL` (default: "gemini-pro")
- **Temperature**: Adjust `TEMPERATURE` for LLM consistency (default: 0.3)

## 🔧 Requirements

- Python 3.8+
- Gemini API key ([Get it here](https://makersuite.google.com/app/apikey) - free tier available)
- Internet connection (for downloading embedding models on first run)

## 📝 Notes

- The embedding model (`all-MiniLM-L6-v2`) will be downloaded automatically on first use (~80MB)
- Vector database is stored locally in `./vector_db/` directory
- Ensure you have sufficient Gemini API credits for LLM evaluations
- The `.env` file is gitignored - never commit your API keys!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the LLM framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Google Gemini](https://ai.google.dev/) for LLM capabilities
- [Sentence Transformers](https://www.sbert.net/) for embeddings
