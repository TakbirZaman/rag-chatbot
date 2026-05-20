
import os, tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import streamlit as st
import os
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Document Q&A — RAG Chatbot")
st.caption("Upload a PDF or text file, then ask questions about it.")

# ── session state ─────────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "history" not in st.session_state:
    st.session_state.history = []

# ── sidebar: upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt", "md"])

    if uploaded and st.button("Index Document"):
        with st.spinner("Chunking + embedding..."):
            # save to temp file
            suffix = ".pdf" if uploaded.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded.getvalue())
                path = f.name

            # load
            loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path)
            docs = loader.load()

            # split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)

            # embed into ChromaDB (local, no API key needed for embeddings)
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # LLM
            llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

            # memory + chain
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                return_source_documents=True
            )
            st.session_state.history = []
            st.success(f"✅ Indexed {len(chunks)} chunks from '{uploaded.name}'")

# ── chat UI ────────────────────────────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your document..."):
    if not st.session_state.chain:
        st.warning("Please upload and index a document first.")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain({"question": prompt})
                answer = result["answer"]
                sources = result.get("source_documents", [])

            st.write(answer)

            if sources:
                with st.expander("📑 Sources"):
                    for i, doc in enumerate(sources[:3]):
                        st.caption(f"**Chunk {i+1}** (page {doc.metadata.get('page','?')})")
                        st.text(doc.page_content[:300] + "...")

        st.session_state.history.append({"role": "assistant", "content": answer})
