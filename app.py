"""Streamlit interface for the PDF-based RAG knowledge assistant."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_CHAT_MODEL,
    build_vector_store,
    extract_pdf_pages,
    generate_grounded_answer,
    retrieve_chunks,
    split_into_chunks,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PDF = BASE_DIR / "data" / "sample_troubleshooting_runbook.pdf"

st.set_page_config(page_title="ML Knowledge Assistant", page_icon="📚", layout="wide")
st.title("Machine Learning Knowledge Assistant")
st.caption("Ask questions about a PDF and receive answers grounded in retrieved passages.")

with st.sidebar:
    st.header("RAG settings")
    top_k = st.select_slider("Retrieved chunks (top-k)", options=[3, 5, 7], value=5)
    chat_model = st.text_input("Answer model", value=os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL))
    st.markdown("**Pipeline settings**")
    st.write(f"Chunk size: {CHUNK_SIZE} tokens")
    st.write(f"Chunk overlap: {CHUNK_OVERLAP} tokens")
    st.write("Embedding: `text-embedding-3-small`")
    st.write("Vector index: FAISS (cosine similarity)")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "OpenAI API key not found. Create a `.env` file from `.env.example` and set "
        "OPENAI_API_KEY before processing a PDF."
    )

uploaded_pdf = st.file_uploader("Upload a PDF knowledge source", type=["pdf"])

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    source_name = uploaded_pdf.name
    st.info(f"Selected document: {source_name}")
else:
    pdf_bytes = SAMPLE_PDF.read_bytes() if SAMPLE_PDF.exists() else b""
    source_name = SAMPLE_PDF.name
    if pdf_bytes:
        st.warning(
            "Using the supplied sample troubleshooting runbook. It is useful for testing the RAG "
            "pipeline, but it is not an ML textbook. Upload an ML/DL book PDF for the final "
            "Machine Learning Knowledge Assistant demonstration."
        )

if st.button("Process PDF and build index", type="primary", disabled=not api_key or not pdf_bytes):
    try:
        client = OpenAI(api_key=api_key)
        with st.spinner("Extracting pages, splitting text, creating embeddings and indexing chunks..."):
            pages, document_stats = extract_pdf_pages(pdf_bytes)
            chunks = split_into_chunks(pages)
            store = build_vector_store(client, chunks, source_name)

        st.session_state["rag_store"] = store
        st.session_state["document_stats"] = document_stats
        st.session_state["chunk_count"] = len(chunks)
        st.session_state["source_name"] = source_name
        st.success("Document processed and FAISS index created.")
    except Exception as exc:
        st.error(f"Could not process the PDF: {exc}")

if "rag_store" in st.session_state:
    stats = st.session_state["document_stats"]
    st.subheader("Document summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pages", stats["page_count"])
    col2.metric("Pages with text", stats["pages_with_text"])
    col3.metric("Text characters", f"{stats['total_characters']:,}")
    col4.metric("Chunks indexed", st.session_state["chunk_count"])

    for note in stats["quality_notes"]:
        st.warning(note)
    if not stats["quality_notes"]:
        st.success("All pages returned extractable text.")

    st.divider()
    st.subheader("Ask a question")
    question = st.text_input(
        "Question about the uploaded document",
        placeholder="For example: What are the recommended checks for a gateway timeout?",
    )
    ask = st.button("Search and generate answer", disabled=not bool(question.strip()) or not api_key)

    if ask:
        try:
            client = OpenAI(api_key=api_key)
            with st.spinner(f"Searching the top {top_k} chunks and generating an answer..."):
                retrieved = retrieve_chunks(client, st.session_state["rag_store"], question, top_k)
                answer = generate_grounded_answer(client, question, retrieved, chat_model)
            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Retrieved evidence")
            for index, item in enumerate(retrieved, start=1):
                title = (
                    f"{index}. Page {item['page']} | {item['chunk_id']} | "
                    f"Similarity: {item['similarity']:.3f}"
                )
                with st.expander(title):
                    st.caption(f"Source: {item['source']}")
                    st.write(item["text"])
        except Exception as exc:
            st.error(f"Unable to answer this question: {exc}")
else:
    st.info("Select a PDF and click **Process PDF and build index** to start asking questions.")

st.caption("Answers are for learning and document search. Always verify critical technical instructions against the source document.")
