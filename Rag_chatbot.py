# import (python built-ins)
import os
import tempfile
import json
import streamlit as st
from dotenv import load_dotenv

## imports langchain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================
# Minimal Unicode Cleanup
# =========================

def clean_text(text):
    """
    Ensure string is safe for Groq API by removing problematic characters.
    """
    if not isinstance(text, str):
        return text
    return text.encode("utf-8", "ignore").decode("utf-8")

# =========================
# JSON Persistent Memory
# =========================

MEMORY_FILE = "chat_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

memory_store = load_memory()

def load_history_from_json(session_id):
    history = ChatMessageHistory()

    if session_id in memory_store:
        for msg in memory_store[session_id]:
            if msg["role"] == "user":
                history.add_user_message(msg["content"])
            else:
                history.add_ai_message(msg["content"])

    return history

def save_message_to_json(session_id, role, content):

    if session_id not in memory_store:
        memory_store[session_id] = []

    memory_store[session_id].append({
        "role": role,
        "content": content
    })

    save_memory(memory_store)

# setup : env + streamlit page

load_dotenv() 
st.set_page_config(page_title=" 📝 RAG Q&A ",layout="wide")
st.title("📝 RAG Q&A with Multiple PDFs + Chat History")

# Sidbar config: Groq API Key input

with st.sidebar:
    st.header("⚙️ Config")
    api_key_input = st.text_input("Groq API Key", type="password")
    st.caption("Upload PDFs -> Ask questions -> Get Answers")

# Accept key from input 
api_key = api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    st.warning(" Please enter your Groq API Key (or set GROQ_API_KEY in .env) ")
    st.stop()

# embeddings and llm initialization

embeddings = HuggingFaceEmbeddings(
hf_model = st.secrets["HF_MODEL"]
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

# upload PDFs (multiple)

uploaded_files = st.file_uploader(
    " 📚 Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more PDFs to begin")
    st.stop()

all_docs = []
tmp_paths = []

for pdf in uploaded_files:

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf.getvalue())
    tmp.close()
    tmp_paths.append(tmp.name)    

    loader = PyPDFLoader(tmp.name)
    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = pdf.name

    all_docs.extend(docs)

st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs")

# Clean up temp files

for p in tmp_paths:
    try:
        os.unlink(p)
    except Exception:
        pass

# chunking (split text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=120
)

splits = text_splitter.split_documents(all_docs)

# ── Vectorstore 

INDEX_DIR = "chroma_index"

vectorstore = Chroma.from_documents(
    splits,
    embeddings,
    persist_directory=INDEX_DIR
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

st.sidebar.write(f"🔍 Indexed {len(splits)} chunks for retrieval")

# ── Helper: format docs for stuffing ─────────────────────────────

def _join_docs(docs, max_chars=7000):

    chunks = []
    total = 0

    for d in docs:
        piece = clean_text(d.page_content)  # <-- clean here

        if total + len(piece) > max_chars:
            break

        chunks.append(piece)
        total += len(piece)

    return "\n\n---\n\n".join(chunks)

# prompts

contextualize_q_prompt = ChatPromptTemplate.from_messages([

    ("system",
     "Rewrite the user's latest question into a standalone search query using the chat history for the context."
     "Return only the rewritten query, no extra text."),

     MessagesPlaceholder("chat_history"),
     ("human","{input}")

])

qa_prompt = ChatPromptTemplate.from_messages([

    ("system",
     "You are a STRICT RAG assistant. You must answer using ONLY the provided context.\n"
     "If the context does NOT contain the answer, reply exactly:\n"
     "'Out of scope - not found in provided documents.'\n"
     "Do NOT use outside knowledge.\n\n"
     "Context:\n{context}"),

     MessagesPlaceholder("chat_history"),
     ("human","{input}")

])

# session state for chat history

if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

def get_history(session_id: str):

    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = load_history_from_json(session_id)

    return st.session_state.chathistory[session_id]

# chat ui

session_id = st.text_input(" 🆔 Session ID ", value="default_session")

user_q = st.chat_input("💬 Ask a question...")

if user_q:
    user_q = clean_text(user_q)  # <-- clean user input

# ── Chat Execution ─────────────────────────────

if user_q:

    history = get_history(session_id)

    # Rewrite question with history

    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q
    )

    standalone_q = clean_text(llm.invoke(rewrite_msgs).content.strip())  # <-- clean rewritten query

    # Retrieve chunks

    docs = retriever.invoke(standalone_q)

    if not docs:

        answer = "Out of scope — not found in provided documents."

        st.chat_message("user").write(user_q)
        st.chat_message("assistant").write(answer)

        history.add_user_message(user_q)
        history.add_ai_message(answer)

        save_message_to_json(session_id, "user", user_q)
        save_message_to_json(session_id, "assistant", answer)

        st.stop()

    # Build context string

    context_str = clean_text(_join_docs(docs))  # <-- clean context

    # Ask final question

    qa_msgs = qa_prompt.format_messages(
        chat_history=history.messages,
        input=user_q,
        context=context_str
    )

    answer = llm.invoke(qa_msgs).content

    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(answer)

    history.add_user_message(user_q)
    history.add_ai_message(answer)

    save_message_to_json(session_id, "user", user_q)
    save_message_to_json(session_id, "assistant", answer)

    # Debug panels

    with st.expander("🧪 Debug: Rewritten Query & Retrieval"):

        st.write("**Rewritten (standalone) query:**")
        st.code(standalone_q or "(empty)", language="text")

        st.write(f"**Retrieved {len(docs)} chunk(s).**")

    with st.expander("📑 Retrieved Chunks"):

        for i, doc in enumerate(docs, 1):

            st.markdown(
                f"**{i}. {doc.metadata.get('source_file','Unknown')} (p{doc.metadata.get('page','?')})**"
            )

            st.write(
                doc.page_content[:500] +
                ("..." if len(doc.page_content) > 500 else "")
            )
