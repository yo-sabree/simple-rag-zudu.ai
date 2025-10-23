RAG Document Q&A System
A Retrieval-Augmented Generation (RAG) system built with Streamlit that allows users to upload PDF documents and ask questions about their content using advanced AI techniques.
Features

PDF Document Processing: Upload multiple PDF files for analysis
Intelligent Retrieval: Uses FAISS vector store for efficient similarity search
Re-ranking: Employs CrossEncoder for improved result relevance
Conversational Memory: Maintains chat history for context-aware responses
Source Citations: Answers include references to specific source documents
Performance Metrics: Real-time tracking of retrieval, re-ranking, and generation times

Architecture
Core Components

Document Processing Pipeline

PDF loading via PyPDFLoader
Text chunking with RecursiveCharacterTextSplitter (1000 chars, 250 overlap)
Vector embeddings using Sentence Transformers (all-MiniLM-L6-v2)
FAISS vector store for efficient retrieval


Two-Stage Retrieval

Stage 1: FAISS similarity search retrieves top 15 candidate documents
Stage 2: CrossEncoder re-ranks results to select the most relevant 3 documents


Answer Generation

Uses Google's Gemini 2.0 Flash model
Includes conversation history for context
Cites sources in responses
Low temperature (0.1) for factual accuracy



Installation
Prerequisites

Python 3.8+
pip package manager

Setup

Clone the repository:

bashgit clone <repository-url>
cd rag-qa-system

Install dependencies:

bashpip install streamlit python-dotenv google-generativeai langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers pypdf

Create a .env file in the project root:

envGEMINI_API_KEY=your_gemini_api_key_here

Get your Gemini API key from Google AI Studio

Usage

Start the application:

bashstreamlit run app.py

Upload Documents:

Use the sidebar to upload one or more PDF files
Click "Process Documents" to index them


Ask Questions:

Enter your question in the text input
Click "Search & Answer" to get a response
View cited sources and performance metrics



Code Structure
├── main()                      # Streamlit app entry point
├── load_embeddings()          # Cached embedding model loader
├── load_reranker()            # Cached CrossEncoder loader
├── process_pdfs()             # PDF processing and vectorization
├── rerank_results()           # Re-ranking logic
└── generate_answer()          # LLM response generation
Key Functions

load_embeddings(): Loads and caches the HuggingFace embedding model for document vectorization
load_reranker(): Initializes the CrossEncoder model for result re-ranking
process_pdfs(): Handles PDF upload, chunking, and FAISS index creation
rerank_results(): Applies CrossEncoder to re-rank retrieved documents by relevance
generate_answer(): Constructs prompt with context and generates answer using Gemini

Session State
The application maintains state across interactions:

vectorstore: FAISS index containing document embeddings
history: Conversation history for contextual responses

Performance Optimization

Caching: Embedding and re-ranking models are cached with @st.cache_resource
Batch Processing: Embeddings generated in batches of 16
Efficient Search: FAISS enables fast approximate nearest neighbor search
Normalized Embeddings: Improves similarity search accuracy

Configuration
Adjustable Parameters

Chunk Size: 1000 characters (line 40)
Chunk Overlap: 250 characters (line 40)
Initial Retrieval: 15 documents (line 91)
Final Results: 3 documents after re-ranking (line 94)
Temperature: 0.1 for factual responses (line 60)
