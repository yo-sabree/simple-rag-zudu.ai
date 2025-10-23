# 📄 RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system built with **Streamlit** that allows users to upload PDF documents and ask questions about their content using advanced AI techniques.

## ✨ Features

* **PDF Document Processing:** Upload multiple PDF files for analysis.
* **Intelligent Retrieval:** Uses **FAISS** vector store for efficient similarity search.
* **Re-ranking:** Employs **CrossEncoder** for improved result relevance.
* **Conversational Memory:** Maintains chat history for context-aware responses.
* **Source Citations:** Answers include references to specific source documents.
* **Performance Metrics:** Real-time tracking of retrieval, re-ranking, and generation times.

---

## 🏗️ Architecture

The system uses a two-stage retrieval process and the Gemini model for generation.

### Core Components

* **Document Processing Pipeline**
    * PDF loading via `PyPDFLoader`.
    * Text chunking with `RecursiveCharacterTextSplitter` (**1000 chars, 250 overlap**).
    * Vector embeddings using **Sentence Transformers** (`all-MiniLM-L6-v2`).
    * **FAISS** vector store for efficient retrieval.
* **Two-Stage Retrieval**
    * **Stage 1:** FAISS similarity search retrieves top **15** candidate documents.
    * **Stage 2:** **CrossEncoder** re-ranks results to select the most relevant **3** documents.
* **Answer Generation**
    * Uses Google's **Gemini 2.0 Flash** model.
    * Includes conversation history for context.
    * Cites sources in responses.
    * Low temperature (**0.1**) for factual accuracy.

---

## ⚙️ Installation

### Prerequisites

* **Python 3.8+**
* **pip** package manager

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rag-qa-system
    ```

2.  **Install dependencies:**
    ```bash
    pip install streamlit python-dotenv google-generativeai langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers pypdf
    ```

3.  **Create a `.env` file** in the project root:
    ```env
    GEMINI_API_KEY=your_gemini_api_key_here
    ```
    *Get your Gemini API key from Google AI Studio.*

---

## 🚀 Usage

1.  **Start the application:**
    ```bash
    streamlit run app.py
    ```

2.  **Upload Documents:**
    * Use the sidebar to upload one or more PDF files.
    * Click **"Process Documents"** to index them.

3.  **Ask Questions:**
    * Enter your question in the text input.
    * Click **"Search & Answer"** to get a response.
    * View cited sources and performance metrics below the answer.


### Key Functions

| Function | Description |
| :--- | :--- |
| `load_embeddings()` | Loads and caches the HuggingFace embedding model. |
| `load_reranker()` | Initializes and caches the CrossEncoder model for re-ranking. |
| `process_pdfs()` | Handles PDF upload, chunking, and FAISS index creation. |
| `rerank_results()` | Applies CrossEncoder to re-rank retrieved documents by relevance. |
| `generate_answer()` | Constructs prompt with context and generates the final answer using Gemini. |

---

## 📈 Performance Optimization

* **Caching:** Embedding and re-ranking models are cached with `@st.cache_resource`.
* **Batch Processing:** Embeddings are generated in batches of 16 for efficiency.
* **Efficient Search:** **FAISS** enables fast approximate nearest neighbor search.
* **Normalized Embeddings:** Improves similarity search accuracy.

### Adjustable Parameters

| Parameter | Value | Location/Purpose |
| :--- | :--- | :--- |
| **Chunk Size** | 1000 characters | Document chunking |
| **Chunk Overlap** | 250 characters | Document chunking |
| **Initial Retrieval** | 15 documents | Stage 1: FAISS search result count |
| **Final Results** | 3 documents | Stage 2: Documents passed to LLM after re-ranking |
| **Temperature** | 0.1 | Gemini model configuration for factual responses |
