import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
PDF_PATH = os.path.join("data_txt", "AMS_MajorHandbook.pdf")
VECTOR_STORE_DIR = os.path.join("data_txt", "vector_store")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:0.5b"
TEMPERATURE = 0.2

st.set_page_config(
    page_title="AMS Major Handbook Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 0.75rem 1rem;
        border-radius: 0.4rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
        color: #1F2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "Hello! I am your **AMS Major Handbook** assistant. "
        "Ask me anything about degree credits, core courses, minor requirements, "
        "or graduation pathways in Art & Media Studies."
    ),
    "sources": [],
}


# ==========================================
# 2. INITIALIZE VECTOR STORE & MODEL (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Indexing AMS Major Handbook into ChromaDB...")
def init_rag_components():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings,
    )

    # Ensure PDF is actually indexed (fixes empty vector_store directory issue)
    if vectorstore._collection.count() == 0:
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"Cannot find PDF file at: {os.path.abspath(PDF_PATH)}")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)

    chunk_count = vectorstore._collection.count()

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=TEMPERATURE,
        num_ctx=2048,
        num_predict=200,
    )
    return vectorstore, llm, chunk_count


try:
    vectorstore, llm, chunk_count = init_rag_components()
    system_ready = True
except Exception as e:
    system_ready = False
    chunk_count = 0
    st.error(
        f"Unable to initialize RAG pipeline ({OLLAMA_URL}). "
        f"Please ensure Ollama is running and PDF exists. Details: {e}"
    )


# ==========================================
# 3. SIDEBAR CONTROLS & SAMPLE QUERIES
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("AMS RAG System")
    st.caption("CS311 Deep Learning — Final Project")

    st.divider()
    st.subheader("⚙️ Retrieval Settings (RAG)")
    top_k = st.slider(
        "Retrieved Chunks (Top-K)",
        min_value=2,
        max_value=6,
        value=3,
        help="Number of most relevant document chunks retrieved from ChromaDB as context.",
    )

    st.markdown(
        f"""
        - **Document:** `AMS_MajorHandbook.pdf`
        - **Indexed Chunks:** `{chunk_count} chunks`
        - **LLM:** `{LLM_MODEL}` (`temp={TEMPERATURE}`)
        - **Embedding:** `{EMBED_MODEL}`
        - **Vector DB:** `ChromaDB`
        """
    )

    st.divider()
    st.subheader("💡 Sample Test Questions")
    sample_questions = [
        "What are the requirements for completing a minor in Art and Media Studies?",
        "How many credits are required for the AMS major?",
        "What are the core courses in Art and Media Studies?",
        "Does the AMS handbook mention quantum computing engineering?",
    ]

    selected_sample = None
    for idx, q in enumerate(sample_questions):
        if st.button(q, key=f"sample_q_{idx}", use_container_width=True):
            selected_sample = q

    st.divider()
    if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.messages = [WELCOME_MESSAGE.copy()]
        st.rerun()


# ==========================================
# 4. MAIN CHAT INTERFACE
# ==========================================
st.markdown(
    '<div class="main-header">🎓 Art & Media Studies (AMS) Handbook Chatbot</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Academic advising assistant grounded strictly in the AMS Major Handbook with PDF page citations.</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [WELCOME_MESSAGE.copy()]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 PDF Citations ({len(msg['sources'])} relevant chunks) — Click to view"):
                for idx, src in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f'<div class="source-box"><b>Chunk {idx} — Page {src["page"]}:</b><br>{src["snippet"]}</div>',
                        unsafe_allow_html=True,
                    )


# ==========================================
# 5. RAG RETRIEVAL & STREAMING RESPONSE
# ==========================================
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful academic advisor assistant for the Art and Media Studies (AMS) Major Handbook. "
            "Answer the user's question concisely and STRICTLY based on the provided context below. "
            "If the answer is not found in the context, clearly state: "
            "'I cannot find that information in the AMS Major Handbook.' Do not guess or fabricate facts.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def stream_rag_response(user_question: str, k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(user_question)

    context_text = "\n\n---\n\n".join(
        [f"[Page {doc.metadata.get('page', 0) + 1}] {doc.page_content}" for doc in retrieved_docs]
    )

    history_messages = []
    for m in st.session_state.messages[:-1][-4:]:
        if m["role"] == "user":
            history_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history_messages.append(AIMessage(content=m["content"]))

    chain = PROMPT_TEMPLATE | llm

    def token_generator():
        for chunk in chain.stream(
            {
                "context": context_text,
                "chat_history": history_messages,
                "question": user_question,
            }
        ):
            if chunk.content:
                yield chunk.content

    sources_formatted = [
        {
            "page": doc.metadata.get("page", 0) + 1,
            "snippet": doc.page_content.strip().replace("\n", " ")[:320] + "...",
        }
        for doc in retrieved_docs
    ]
    return token_generator(), sources_formatted


user_input = st.chat_input("Ask a question about the AMS program...")
active_query = selected_sample if selected_sample else user_input

if active_query and system_ready:
    st.session_state.messages.append({"role": "user", "content": active_query, "sources": []})
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching handbook..."):
            stream_gen, sources = stream_rag_response(active_query, top_k)
        answer = st.write_stream(stream_gen)

        if sources:
            with st.expander(f"📚 PDF Citations ({len(sources)} relevant chunks) — Click to view"):
                for idx, src in enumerate(sources, start=1):
                    st.markdown(
                        f'<div class="source-box"><b>Chunk {idx} — Page {src["page"]}:</b><br>{src["snippet"]}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("No matching chunks found in ChromaDB.")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )