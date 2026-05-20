"""
RAG Chatbot — PDF/Document Q&A
Uses ChromaDB default embeddings (no sentence-transformers needed)
"""
import os, tempfile
import streamlit as st

os.environ["ANTHROPIC_API_KEY"] = st.secrets.get(
    "ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")
)

import anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import chromadb
from chromadb.utils import embedding_functions

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Document Q&A — RAG Chatbot")
st.caption("Upload a PDF or text file, then ask questions about it.")

if "collection" not in st.session_state:
    st.session_state.collection = None
if "history" not in st.session_state:
    st.session_state.history = []

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt", "md"])

    if uploaded and st.button("Index Document"):
        with st.spinner("Chunking + indexing..."):
            suffix = ".pdf" if uploaded.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded.getvalue())
                path = f.name

            loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)

            # ChromaDB with default embeddings (no extra package)
            chroma_client = chromadb.Client()
            ef = embedding_functions.DefaultEmbeddingFunction()

            # delete old collection if exists
            try:
                chroma_client.delete_collection("rag_docs")
            except Exception:
                pass

            collection = chroma_client.create_collection("rag_docs", embedding_function=ef)
            collection.add(
                documents=[c.page_content for c in chunks],
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )

            st.session_state.collection = collection
            st.session_state.history = []
            st.success(f"✅ Indexed {len(chunks)} chunks from '{uploaded.name}'")

# ── chat UI ───────────────────────────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your document..."):
    if not st.session_state.collection:
        st.warning("Please upload and index a document first.")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # retrieve top-4 relevant chunks
                results = st.session_state.collection.query(
                    query_texts=[prompt], n_results=4
                )
                chunks_text = results["documents"][0]
                context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks_text))

                # last 6 turns for memory
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
                for i, chunk in enumerate(chunks_text):
                    st.caption(f"Chunk {i+1}")
                    st.text(chunk[:300] + "...")

        st.session_state.history.append({"role": "assistant", "content": answer})
