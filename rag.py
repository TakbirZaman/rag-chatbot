"""
RAG Chatbot — PDF/Document Q&A
No LangChain chains — direct Anthropic API calls only (more stable)
"""
import os, tempfile
import streamlit as st

os.environ["ANTHROPIC_API_KEY"] = st.secrets.get(
    "ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")
)

import anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Document Q&A — RAG Chatbot")
st.caption("Upload a PDF or text file, then ask questions about it.")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "history" not in st.session_state:
    st.session_state.history = []

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt", "md"])

    if uploaded and st.button("Index Document"):
        with st.spinner("Chunking + embedding..."):
            suffix = ".pdf" if uploaded.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded.getvalue())
                path = f.name

            loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)

            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            st.session_state.history = []
            st.success(f"✅ Indexed {len(chunks)} chunks from '{uploaded.name}'")

# ── chat UI ───────────────────────────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your document..."):
    if not st.session_state.retriever:
        st.warning("Please upload and index a document first.")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # retrieve relevant chunks
                docs = st.session_state.retriever.invoke(prompt)
                context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

                # build messages for Anthropic (last 6 turns for memory)
                recent = st.session_state.history[-6:]
                messages = [{"role": m["role"], "content": m["content"]} for m in recent]

                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    system=f"""You are a document assistant. Answer using ONLY the excerpts below.
If the answer isn't there, say so clearly.

DOCUMENT EXCERPTS:
{context}""",
                    messages=messages
                )
                answer = response.content[0].text

            st.write(answer)

            with st.expander("📑 Source chunks"):
                for i, doc in enumerate(docs):
                    st.caption(f"Chunk {i+1} — page {doc.metadata.get('page', '?')}")
                    st.text(doc.page_content[:300] + "...")

        st.session_state.history.append({"role": "assistant", "content": answer})
