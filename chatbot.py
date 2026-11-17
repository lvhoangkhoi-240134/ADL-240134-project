#!/usr/bin/env python3

"""
Task 2: Major Handbook RAG Chatbot
Usage:
  python chatbot.py                  # Start interactive chat mode
  python chatbot.py --rebuild        # Rebuild vector store from scratch and start chat
  python chatbot.py -q "Question"    # Ask a single question
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# RAG & CHATBOT SETTINGS
DATA_DIR = "data_txt"
VECTOR_STORE = DATA_DIR + "/vector_store"
COLLECTION = "major_handbook_rag"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:0.5b"
TEMPERATURE = 0.2  # Lower temperature for more factual, grounded handbook answers
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
TOP_K = 5
SCORE_THRESHOLD = 0.35

# STRICT GROUNDED PROMPT TEMPLATE WITH CONVERSATION HISTORY
PROMPT_TEMPLATE = """You are an academic advisor assistant for the Art and Media Studies (AMS) Major Handbook.
Answer the student's question accurately using ONLY the provided handbook excerpts below.

RECENT CONVERSATION:
{history}

HANDBOOK CONTEXT:
{context}
---
STUDENT QUESTION: {question}

INSTRUCTIONS:
1. Rely strictly on the HANDBOOK CONTEXT above. Do not invent courses, credits, or policies not mentioned in the context.
2. If the handbook context does not contain enough information to answer the question, explicitly say: "The Major Handbook excerpts provided do not contain enough information to answer this."
3. Keep the answer clear, structured, and concise (2-5 sentences or bullet points).

ANSWER:"""


def load_text_files(directory: str) -> List[Document]:
    documents = []
    for txt_file in Path(directory).glob("*.txt"):
        print(f"  Loading text: {txt_file.name}")
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append(Document(page_content=content, metadata={"source": txt_file.name}))
    return documents


def load_pdf_files(directory: str) -> List[Document]:
    documents = []
    for pdf_file in Path(directory).glob("*.pdf"):
        print(f"  Loading PDF: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        for page_doc in pages:
            # Normalize source filename and convert 0-based page index to 1-based page number
            page_num = page_doc.metadata.get("page")
            page_doc.metadata["source"] = pdf_file.name
            page_doc.metadata["page_display"] = str(int(page_num) + 1) if page_num is not None else "N/A"
        documents.extend(pages)
    return documents


def load_and_index(embeddings: OllamaEmbeddings, rebuild: bool = False) -> Chroma:
    store_path = Path(VECTOR_STORE)
    if rebuild and store_path.exists():
        print("Rebuilding vector store: removing existing index...")
        shutil.rmtree(store_path)

    if store_path.exists():
        print("Loading existing vector store...")
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=VECTOR_STORE,
        )

    print("No vector store found. Indexing Major Handbook documents...")
    txt_files = list(Path(DATA_DIR).glob("*.txt"))
    pdf_files = list(Path(DATA_DIR).glob("*.pdf"))

    if not txt_files and not pdf_files:
        print(f"Error: No .pdf or .txt files found in '{DATA_DIR}'.")
        sys.exit(1)

    documents = []
    if txt_files:
        documents.extend(load_text_files(DATA_DIR))
    if pdf_files:
        documents.extend(load_pdf_files(DATA_DIR))

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ).split_documents(documents)

    print(f"  Indexing {len(chunks)} chunks into ChromaDB...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=VECTOR_STORE,
    )


def format_history(chat_history: List[Tuple[str, str]], max_turns: int = 3) -> str:
    if not chat_history:
        return "None"
    recent = chat_history[-max_turns:]
    return "\n".join([f"Student: {q}\nAdvisor: {a}" for q, a in recent])


def ask(
    question: str,
    db: Chroma,
    llm: OllamaLLM,
    chat_history: List[Tuple[str, str]] | None = None,
) -> Dict[str, Any]:
    raw_results = db.similarity_search_with_score(question, k=TOP_K)
    sources = [
        {
            "source": str(doc.metadata.get("source", "Unknown")),
            "page": str(doc.metadata.get("page_display", "N/A")),
            "score": round(1 / (1 + score), 4),
            "content": doc.page_content,
        }
        for doc, score in raw_results
    ]

    avg_score = sum(float(s["score"]) for s in sources) / len(sources) if sources else 0
    used_fallback = not sources or avg_score < SCORE_THRESHOLD

    if used_fallback:
        context = "No sufficiently relevant sections were found in the Major Handbook."
    else:
        context = "\n\n---\n\n".join(
            f"[{s['source']} - Page {s['page']}]\n{s['content']}" for s in sources
        )

    history_str = format_history(chat_history or [])
    prompt = (
        PROMPT_TEMPLATE.replace("{history}", history_str)
        .replace("{context}", context)
        .replace("{question}", question)
    )
    answer = llm.invoke(prompt).strip()
    return {"answer": answer, "sources": sources, "used_fallback": used_fallback}


def print_response(result: Dict[str, Any]) -> None:
    print(f"\nAdvisor: {result['answer']}\n")
    if result["used_fallback"]:
        print("(Note: Low retrieval relevance — handbook context may not cover this question.)")
    else:
        print("Sources:")
        for s in result["sources"][:3]:
            print(f"  - {s['source']} (Page {s['page']}) | score={s['score']}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Major Handbook RAG Chatbot")
    parser.add_argument("--question", "-q", default=None, help="Ask a single question and exit")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the vector store")
    args = parser.parse_args()

    print(f"Loading Ollama embeddings ({EMBED_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    db = load_and_index(embeddings, rebuild=args.rebuild)

    print(f"Connecting to Ollama ({LLM_MODEL})...")
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)

    # Single-question mode
    if args.question:
        print(f"\nStudent: {args.question}")
        result = ask(args.question, db, llm)
        print_response(result)
        return

    # Interactive multi-turn chat mode
    print("\n" + "=" * 60)
    print("AMS Major Handbook Chatbot is ready!")
    print("Type your question below. Commands: 'clear' (reset memory), 'exit' (quit).")
    print("=" * 60)

    chat_history: List[Tuple[str, str]] = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("Conversation history cleared.")
            continue

        result = ask(user_input, db, llm, chat_history)
        print_response(result)
        chat_history.append((user_input, result["answer"]))


if __name__ == "__main__":
    main()