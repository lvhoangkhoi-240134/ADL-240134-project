# Local RAG System for Academic Advising (AMS Major Handbook)

**Course:** Advanced Deep Learning (Final Project)  
**Student ID:** 240134  

## Overview
A fully local **Retrieval-Augmented Generation (RAG)** assistant designed to answer academic advising questions strictly grounded in the **Art and Media Studies Major Handbook** (`AMS_MajorHandbook.pdf`). The system runs completely offline on local hardware using **Ollama**, **LangChain**, **ChromaDB**, **Docker**, and **Streamlit**, ensuring zero API costs, data privacy, and verifiable PDF page citations.

## Key Features
- **PDF Ingestion & Vector Indexing:** Parses `AMS_MajorHandbook.pdf` into 115 semantic chunks (`chunk_size=1000`, `chunk_overlap=200`) and indexes embeddings locally using `nomic-embed-text` and **ChromaDB**.
- **Anti-Hallucination & Multi-Turn Memory:** Uses `qwen2.5:0.5b` (`temperature=0.2`) with strict system prompt constraints and sliding-window conversation memory.
- **Docker Containerization:** Packaged into a lightweight `python:3.11-slim` container (`ams-rag-chatbot`) communicating with the host Ollama service.
- **Interactive Web UI (Streamlit):** Features token-level streaming responses, dynamic `Top-K` retrieval slider, expandable PDF page citations, sample test prompts, and one-click session reset.

## Tech Stack
- **Language:** Python 3.11
- **Orchestration:** LangChain (`langchain-core`, `langchain-community`, `langchain-ollama`, `langchain-chroma`)
- **Local LLM & Embeddings:** Ollama (`qwen2.5:0.5b`, `nomic-embed-text`)
- **Vector Database:** ChromaDB
- **Containerization:** Docker
- **Frontend UI:** Streamlit

## Project Structure
```text
├── app.py                 # Streamlit web application (Task 4)
├── chatbot.py             # CLI conversational RAG chatbot (Task 2)
├── run_txt.py             # PDF ingestion & basic RAG script (Task 1)
├── Dockerfile             # Docker image definition (Task 3)
├── .dockerignore          # Excludes local caches from Docker build context
├── requirements.txt       # Python dependencies
└── data_txt/
    └── AMS_MajorHandbook.pdf
