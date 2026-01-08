"""
Streamlit Web Application for AI Resume Screener
"""
import streamlit as st
from rag_pipeline import RAGResumeScreener
from utils import extract_text_from_pdf, clean_text
import json
import time

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'screener' not in st.session_state:
    st.session_state.screener = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.markdown('<h1 class="main-header">📄 AI Resume Screener</h1>', unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Intelligent resume screening using RAG (Retrieval-Augmented Generation) with 
        Semantic Search (60%) and LLM Evaluation (40%)
    </div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("""
    **Scoring Weights:**
    - Semantic Search: 60%
    - LLM Evaluation: 40%
    """)
    
    st.markdown("---")
    st.subheader("📋 Instructions")
    st.markdown("""
    1. Upload or paste job description
    2. Upload or paste resume
    3. Click 'Screen Resume' button
    4. View detailed results
    """)
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    This tool uses:
    - **ChromaDB** for vector storage
    - **Sentence Transformers** for embeddings
    - **Google Gemini** for LLM evaluation
    """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Job Description")
    job_input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Paste Text"],
        key="job_method"
    )
    
    job_description = ""
    
    if job_input_method == "Upload File":
        job_file = st.file_uploader(
            "Upload Job Description (PDF or TXT)",
            type=['pdf', 'txt'],
            key="job_file"
        )
        if job_file:
            if job_file.type == "application/pdf" or job_file.name.endswith('.pdf'):
                # Reset file pointer
                job_file.seek(0)
                job_description = extract_text_from_pdf(job_file)
                if job_description:
                    st.success("✅ Job description loaded from PDF")
                else:
                    st.error("❌ Failed to extract text from PDF")
            else:
                job_file.seek(0)
                job_description = job_file.read().decode('utf-8')
                st.success("✅ Job description loaded from text file")
    else:
        job_description = st.text_area(
            "Paste Job Description:",
            height=200,
            placeholder="Enter the job description here...",
            key="job_text"
        )
    
    if job_description:
        with st.expander("Preview Job Description"):
            st.text(job_description[:500] + "..." if len(job_description) > 500 else job_description)

with col2:
    st.subheader("👤 Resume")
    resume_input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Paste Text"],
        key="resume_method"
    )
    
    resume_text = ""
    
    if resume_input_method == "Upload File":
        resume_file = st.file_uploader(
            "Upload Resume (PDF or TXT)",
            type=['pdf', 'txt'],
            key="resume_file"
        )
        if resume_file:
            if resume_file.type == "application/pdf" or resume_file.name.endswith('.pdf'):
                # Reset file pointer
                resume_file.seek(0)
                resume_text = extract_text_from_pdf(resume_file)
                if resume_text:
                    st.success("✅ Resume loaded from PDF")
                else:
                    st.error("❌ Failed to extract text from PDF")
            else:
                resume_file.seek(0)
                resume_text = resume_file.read().decode('utf-8')
                st.success("✅ Resume loaded from text file")
    else:
        resume_text = st.text_area(
            "Paste Resume:",
            height=200,
            placeholder="Enter the resume text here...",
            key="resume_text"
        )
    
    if resume_text:
        with st.expander("Preview Resume"):
            st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)

# Screen button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    screen_button = st.button(
        "🚀 Screen Resume",
        type="primary",
        use_container_width=True,
        disabled=(not job_description or not resume_text)
    )

# Processing and Results
if screen_button and job_description and resume_text:
    # Clean the text
    job_description = clean_text(job_description)
    resume_text = clean_text(resume_text)
    
    # Initialize screener if not already done
    if st.session_state.screener is None:
        with st.spinner("Initializing AI Resume Screener..."):
            try:
                st.session_state.screener = RAGResumeScreener()
                st.success("✅ Screener initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing screener: {str(e)}")
                st.stop()
    
    # Perform screening
    with st.spinner("🔄 Screening resume... This may take a moment."):
        try:
            result = st.session_state.screener.screen_resume(resume_text, job_description)
            st.session_state.results = result
        except Exception as e:
            st.error(f"❌ Error during screening: {str(e)}")
            st.stop()
    
    # Display Results
    st.markdown("---")
    st.header("📊 Screening Results")
    
    # Overall Score
    col_score1, col_score2, col_score3 = st.columns(3)
    
    with col_score1:
        final_score = result['final_score']
        score_color = "green" if final_score >= 0.7 else "orange" if final_score >= 0.5 else "red"
        st.metric(
            "Final Score",
            f"{final_score:.1%}",
            delta=result['recommendation']
        )
    
    with col_score2:
        st.metric(
            "Semantic Score",
            f"{result['semantic_score']:.1%}",
            delta=f"Weight: {result['weights']['semantic_search']:.0%}"
        )
    
    with col_score3:
        st.metric(
            "LLM Score",
            f"{result['llm_score']:.1%}",
            delta=f"Weight: {result['weights']['llm']:.0%}"
        )
    
    # Recommendation
    st.markdown("---")
    recommendation_color = {
        "Strongly Recommended": "🟢",
        "Recommended": "🟡",
        "Consider": "🟠",
        "Weak Match": "🔴",
        "Not Recommended": "⚫"
    }
    
    rec_icon = "🟢" if "Strongly" in result['recommendation'] else \
               "🟡" if "Recommended" in result['recommendation'] else \
               "🟠" if "Consider" in result['recommendation'] else \
               "🔴" if "Weak" in result['recommendation'] else "⚫"
    
    st.info(f"{rec_icon} **Recommendation:** {result['recommendation']}")
    
    # Detailed Analysis
    st.markdown("---")
    st.subheader("🔍 Detailed Analysis")
    
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        if result['llm_details'].get('matched_skills'):
            st.success("✅ **Matched Skills:**")
            for skill in result['llm_details']['matched_skills']:
                st.write(f"  • {skill}")
        else:
            st.info("No matched skills identified")
    
    with col_analysis2:
        if result['llm_details'].get('missing_skills'):
            st.warning("❌ **Missing Skills:**")
            for skill in result['llm_details']['missing_skills']:
                st.write(f"  • {skill}")
        else:
            st.info("No missing skills identified")
    
    # LLM Reasoning
    if result['llm_details'].get('reasoning') and not result['llm_details']['reasoning'].startswith('Error'):
        st.markdown("---")
        st.subheader("🤖 LLM Reasoning")
        st.write(result['llm_details']['reasoning'])
    
    # Score Visualization
    st.markdown("---")
    st.subheader("📈 Score Breakdown")
    
    import pandas as pd
    
    score_data = pd.DataFrame({
        'Component': ['Semantic Search', 'LLM Evaluation'],
        'Score': [result['semantic_score'], result['llm_score']],
        'Weight': [result['weights']['semantic_search'], result['weights']['llm']],
        'Weighted Score': [
            result['semantic_score'] * result['weights']['semantic_search'],
            result['llm_score'] * result['weights']['llm']
        ]
    })
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.bar_chart(score_data.set_index('Component')['Score'])
    
    with col_viz2:
        st.bar_chart(score_data.set_index('Component')['Weighted Score'])
    
    # Download Results
    st.markdown("---")
    results_json = json.dumps(result, indent=2)
    st.download_button(
        label="📥 Download Results (JSON)",
        data=results_json,
        file_name=f"resume_screening_result_{int(time.time())}.json",
        mime="application/json"
    )
    
    # Store results in session state
    st.session_state.results = result

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #999; padding: 2rem;'>
        <p>AI Resume Screener | Powered by LangChain, ChromaDB, and Google Gemini</p>
    </div>
""", unsafe_allow_html=True)
