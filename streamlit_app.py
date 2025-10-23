import streamlit as st
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
import tempfile
from sentence_transformers import CrossEncoder

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="RAG Q&A System", layout="wide")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
    )

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2")

def rerank_results(query, docs, reranker, k=3):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]

def process_pdfs(uploaded_files, embeddings) -> FAISS:
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = uploaded_file.name
        all_docs.extend(docs)
        os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    chunks = splitter.split_documents(all_docs)
    return FAISS.from_documents(chunks, embeddings)

def generate_answer(query: str, contexts: List[Document], history: str) -> str:
    context_text = ""
    for i, doc in enumerate(contexts, 1):
        context_text += f"\n[Source {i}: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}\n"
    prompt = f"""
You answer questions only using the context below and cite using [Source N].
If the answer is not present, say: I couldn't find this information in the provided documents.

Chat History:
{history}

Context:
{context_text}

Question: {query}

Answer:
"""
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content(prompt, generation_config={'temperature': 0.1})
    return response.text

def main():
    st.title("RAG Document Q&A System")

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'history' not in st.session_state:
        st.session_state.history = ""

    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing PDFs"):
                embeddings = load_embeddings()
                st.session_state.vectorstore = process_pdfs(uploaded_files, embeddings)
                st.success(f"Processed {len(uploaded_files)} files")

    st.header("Ask Questions")
    query = st.text_input("Enter your question")

    if st.button("Search & Answer") and query:
        if st.session_state.vectorstore is None:
            st.error("Upload and process documents first")
            return

        reranker = load_reranker()
        t1 = time.time()
        results = st.session_state.vectorstore.similarity_search(query, k=15)
        retrieval_time = (time.time() - t1) * 1000

        t2 = time.time()
        top_docs = rerank_results(query, results, reranker, k=3)
        rerank_time = (time.time() - t2) * 1000

        t3 = time.time()
        answer = generate_answer(query, top_docs, st.session_state.history)
        generation_time = (time.time() - t3) * 1000

        st.session_state.history += f"\nUser: {query}\nAssistant: {answer}\n"

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval", f"{retrieval_time:.0f} ms")
        col2.metric("Re-Rank", f"{rerank_time:.0f} ms")
        col3.metric("Generation", f"{generation_time:.0f} ms")

        st.subheader("Top Sources")
        for i, doc in enumerate(top_docs, 1):
            with st.expander(f"Source {i}: {doc.metadata.get('source', 'Unknown')}"):
                st.text(doc.page_content[:500] + "...")

        st.subheader("Conversation Memory")
        st.text(st.session_state.history)

if __name__ == "__main__":
    main()
