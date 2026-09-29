"""
Streamlit app for the AI Resume Screener.
"""
import json
import time

import pandas as pd
import streamlit as st

from rag_pipeline import RAGResumeScreener
from utils import clean_text, load_text_from_upload


st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "screener" not in st.session_state:
    st.session_state.screener = None
if "results" not in st.session_state:
    st.session_state.results = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


def get_screener() -> RAGResumeScreener:
    if st.session_state.screener is None:
        with st.spinner("Loading embedding model, ChromaDB, and Gemini client..."):
            st.session_state.screener = RAGResumeScreener()
    return st.session_state.screener


def result_to_json(result) -> str:
    return json.dumps(result, indent=2, default=str)


def show_single_result(result: dict) -> None:
    if result.get("status") != "success":
        st.error(
            "Screening failed because of a system or API error. "
            "This is not a candidate score of 0%."
        )
        st.write(result.get("error") or "Unknown error")
        if result.get("retrieval_score") is not None:
            st.info(f"Retrieval still succeeded. Similarity: {result['retrieval_score']:.1%}")
        return

    col_score1, col_score2, col_score3 = st.columns(3)
    with col_score1:
        st.metric("Final Score", f"{result['final_score']:.1%}")
    with col_score2:
        st.metric(
            "Retrieval Score",
            f"{result['retrieval_score']:.1%}",
            delta=f"Weight: {result['weights']['retrieval']:.0%}",
        )
    with col_score3:
        st.metric(
            "LLM Score",
            f"{result['llm_score']:.1%}",
            delta=f"Weight: {result['weights']['llm']:.0%}",
        )

    rec = result.get("recommendation") or ""
    rec_icon = (
        "🟢" if "Strongly" in rec else
        "🟡" if rec.startswith("Recommended") else
        "🟠" if "Consider" in rec else
        "🔴" if "Weak" in rec else "⚫"
    )
    st.info(f"{rec_icon} **Recommendation:** {result['recommendation']}")

    st.markdown("---")
    st.subheader("Detailed analysis")
    col_analysis1, col_analysis2 = st.columns(2)
    details = result.get("llm_details") or {}

    with col_analysis1:
        matched = details.get("matched_skills") or []
        if matched:
            st.success("Matched skills")
            for skill in matched:
                st.write(f"• {skill}")
        else:
            st.info("No matched skills identified")

    with col_analysis2:
        missing = details.get("missing_skills") or []
        if missing:
            st.warning("Missing skills")
            for skill in missing:
                st.write(f"• {skill}")
        else:
            st.info("No missing skills identified")

    if details.get("reasoning"):
        st.markdown("---")
        st.subheader("LLM reasoning")
        st.write(details["reasoning"])

    retrieved = result.get("retrieved_chunks") or []
    if retrieved:
        st.markdown("---")
        st.subheader("Retrieved resume excerpts")
        st.caption(
            f"ChromaDB returned {len(retrieved)} of {result.get('chunk_count', len(retrieved))} "
            "indexed chunks, ranked by similarity to the job description."
        )
        for index, chunk in enumerate(retrieved, start=1):
            with st.expander(f"Excerpt {index} · similarity {chunk.get('similarity', 0):.1%}"):
                st.write(chunk.get("text", ""))

    st.markdown("---")
    st.subheader("Score breakdown")
    score_data = pd.DataFrame(
        {
            "Component": ["Retrieval", "LLM evaluation"],
            "Score": [result["retrieval_score"], result["llm_score"]],
            "Weighted Score": [
                result["retrieval_score"] * result["weights"]["retrieval"],
                result["llm_score"] * result["weights"]["llm"],
            ],
        }
    )
    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        st.bar_chart(score_data.set_index("Component")["Score"])
    with col_viz2:
        st.bar_chart(score_data.set_index("Component")["Weighted Score"])

    st.download_button(
        label="Download results (JSON)",
        data=result_to_json(result),
        file_name=f"resume_screening_result_{int(time.time())}.json",
        mime="application/json",
    )


st.markdown('<h1 class="main-header">AI Resume Screener</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #666; margin-bottom: 1.5rem;'>
        RAG screening: resume chunks are embedded in ChromaDB, the job description retrieves
        the most relevant sections, and Gemini evaluates that retrieved context.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("How scoring works")
    st.info(
        "**Retrieval 40%** — average cosine similarity of "
        "the resume chunks retrieved for this job description.\n\n"
        "**LLM 60%** — Gemini structured evaluation of those retrieved excerpts."
    )
    st.markdown("---")
    st.subheader("Pipeline")
    st.markdown(
        """
        1. Extract and clean text
        2. Split the resume into chunks
        3. Store embeddings in ChromaDB
        4. Retrieve top matching sections
        5. Evaluate with Gemini
        """
    )
    st.markdown("---")
    st.subheader("Stack")
    st.markdown(
        """
        - **ChromaDB** — vector store
        - **Sentence Transformers** — embeddings
        - **LangChain** — chunking and Gemini client
        - **Google Gemini** — structured evaluation
        """
    )

single_tab, batch_tab = st.tabs(["Single resume", "Batch screening"])

with single_tab:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job description")
        job_input_method = st.radio(
            "Job description input",
            ["Upload File", "Paste Text"],
            key="job_method",
        )
        job_description = ""
        if job_input_method == "Upload File":
            job_file = st.file_uploader(
                "Upload job description (PDF or TXT)",
                type=["pdf", "txt"],
                key="job_file",
            )
            if job_file:
                job_description = load_text_from_upload(job_file) or ""
                if job_description:
                    st.success("Job description loaded")
                else:
                    st.error("Could not extract text from the job description file")
        else:
            job_description = st.text_area(
                "Paste job description",
                height=220,
                placeholder="Enter the job description here...",
                key="job_text",
            )
        if job_description:
            with st.expander("Preview job description"):
                preview = job_description[:800]
                st.text(preview + ("..." if len(job_description) > 800 else ""))

    with col2:
        st.subheader("Resume")
        resume_input_method = st.radio(
            "Resume input",
            ["Upload File", "Paste Text"],
            key="resume_method",
        )
        resume_text = ""
        if resume_input_method == "Upload File":
            resume_file = st.file_uploader(
                "Upload resume (PDF or TXT)",
                type=["pdf", "txt"],
                key="resume_file",
            )
            if resume_file:
                resume_text = load_text_from_upload(resume_file) or ""
                if resume_text:
                    st.success("Resume loaded")
                else:
                    st.error("Could not extract text from the resume file")
        else:
            resume_text = st.text_area(
                "Paste resume",
                height=220,
                placeholder="Enter the resume text here...",
                key="resume_text",
            )
        if resume_text:
            with st.expander("Preview resume"):
                preview = resume_text[:800]
                st.text(preview + ("..." if len(resume_text) > 800 else ""))

    st.markdown("---")
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        screen_button = st.button(
            "Screen resume",
            type="primary",
            use_container_width=True,
            disabled=(not job_description or not resume_text),
        )

    if screen_button and job_description and resume_text:
        job_description = clean_text(job_description)
        resume_text = clean_text(resume_text)
        try:
            screener = get_screener()
        except Exception as exc:
            st.error(f"Could not initialize the screener: {exc}")
            st.stop()

        with st.spinner("Indexing resume chunks, retrieving matches, and evaluating..."):
            try:
                result = screener.screen_resume(resume_text, job_description)
                st.session_state.results = result
            except Exception as exc:
                st.error(f"Screening failed: {exc}")
                st.stop()

        st.header("Screening results")
        show_single_result(result)

    elif st.session_state.results:
        st.header("Last screening results")
        show_single_result(st.session_state.results)

with batch_tab:
    st.subheader("Screen several resumes against one job description")
    st.caption(
        "Each resume is chunked and stored in ChromaDB, then retrieved independently "
        "against the same job description before Gemini evaluation."
    )

    batch_job_method = st.radio(
        "Job description input",
        ["Upload File", "Paste Text"],
        key="batch_job_method",
    )
    batch_job_description = ""
    if batch_job_method == "Upload File":
        batch_job_file = st.file_uploader(
            "Upload job description (PDF or TXT)",
            type=["pdf", "txt"],
            key="batch_job_file",
        )
        if batch_job_file:
            batch_job_description = load_text_from_upload(batch_job_file) or ""
            if batch_job_description:
                st.success("Job description loaded")
            else:
                st.error("Could not extract text from the job description file")
    else:
        batch_job_description = st.text_area(
            "Paste job description",
            height=180,
            key="batch_job_text",
        )

    batch_files = st.file_uploader(
        "Upload one or more resumes (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="batch_resume_files",
    )

    batch_button = st.button(
        "Screen batch",
        type="primary",
        disabled=(not batch_job_description or not batch_files),
    )

    if batch_button and batch_job_description and batch_files:
        resumes = []
        for uploaded in batch_files:
            text = load_text_from_upload(uploaded)
            resumes.append({"id": uploaded.name, "text": text or ""})

        try:
            screener = get_screener()
        except Exception as exc:
            st.error(f"Could not initialize the screener: {exc}")
            st.stop()

        with st.spinner(f"Screening {len(resumes)} resume(s)..."):
            try:
                batch_results = screener.batch_screen_resumes(resumes, batch_job_description)
                st.session_state.batch_results = batch_results
            except Exception as exc:
                st.error(f"Batch screening failed: {exc}")
                st.stop()

    batch_results = st.session_state.batch_results
    if batch_results:
        st.markdown("---")
        st.subheader("Batch ranking")
        rows = []
        for item in batch_results:
            rows.append(
                {
                    "Resume": item.get("resume_id"),
                    "Status": item.get("status"),
                    "Final score": item.get("final_score"),
                    "Retrieval": item.get("retrieval_score"),
                    "LLM": item.get("llm_score"),
                    "Recommendation": item.get("recommendation"),
                    "Error": item.get("error"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.download_button(
            label="Download batch results (JSON)",
            data=result_to_json(batch_results),
            file_name=f"batch_screening_results_{int(time.time())}.json",
            mime="application/json",
        )

        for item in batch_results:
            title = item.get("resume_id") or "resume"
            status = item.get("status")
            score = item.get("final_score")
            label = f"{title} · {status}"
            if score is not None:
                label += f" · {score:.1%}"
            with st.expander(label):
                show_single_result(item)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #999; padding: 1rem;'>
        <p>AI Resume Screener · LangChain · ChromaDB · Sentence Transformers · Google Gemini</p>
    </div>
    """,
    unsafe_allow_html=True,
)
