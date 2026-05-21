import os
import re
import math
import streamlit as st
from pypdf import PdfReader
import anthropic

# ── API Key ───────────────────────────────────────────────────────────────────
try:
    API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Text Helpers ──────────────────────────────────────────────────────────────
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")

def chunk_text(text, size=800, overlap=100):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_index(chunks):
    df = {}
    for chunk in chunks:
        for word in set(tokenize(chunk)):
            df[word] = df.get(word, 0) + 1
    N = len(chunks)
    idf = {w: math.log(N / (v + 1)) for w, v in df.items()}
    vectors = []
    for chunk in chunks:
        tokens = tokenize(chunk)
        if not tokens:
            vectors.append({})
            continue
        tf = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        vec = {w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()}
        vectors.append(vec)
    return vectors, idf

def cosine(a, b):
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v ** 2 for v in a.values()))
    nb = math.sqrt(sum(v ** 2 for v in b.values()))
    return dot / (na * nb + 1e-9)

def retrieve(query, chunks, vectors, idf, k=4):
    tokens = tokenize(query)
    if not tokens:
        return chunks[:k]
    tf = {}
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    qvec = {w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()}
    scores = sorted(enumerate(vectors), key=lambda x: cosine(qvec, x[1]), reverse=True)
    return [chunks[i] for i, _ in scores[:k]]

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 RAG Chatbot — Document Q&A")
st.caption("Upload a PDF or TXT file, then ask questions about it.")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectors" not in st.session_state:
    st.session_state.vectors = []
if "idf" not in st.session_state:
    st.session_state.idf = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])

    if uploaded:
        if st.button("Index Document", type="primary"):
            with st.spinner("Reading and indexing..."):
                text = extract_text(uploaded)
            if not text.strip():
                st.error("Could not extract any text from this file.")
            else:
                with st.spinner("Building search index..."):
                    chunks = chunk_text(text)
                    vectors, idf = build_index(chunks)
                st.session_state.chunks = chunks
                st.session_state.vectors = vectors
                st.session_state.idf = idf
                st.session_state.doc_name = uploaded.name
                st.session_state.history = []
                st.success(f"✅ Done! {len(chunks)} chunks indexed.")

    if st.session_state.doc_name:
        st.info(f"Active file: **{st.session_state.doc_name}**")
        if st.button("Clear"):
            st.session_state.chunks = []
            st.session_state.vectors = []
            st.session_state.idf = {}
            st.session_state.history = []
            st.session_state.doc_name = ""
            st.rerun()

# ── Chat ──────────────────────────────────────────────────────────────────────
if not st.session_state.doc_name:
    st.info("👈 Upload a document from the sidebar to get started.")
else:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about your document...")

    if question:
        if not API_KEY:
            st.error("ANTHROPIC_API_KEY is missing. Add it in Streamlit Secrets.")
            st.stop()

        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and thinking..."):
                top_chunks = retrieve(
                    question,
                    st.session_state.chunks,
                    st.session_state.vectors,
                    st.session_state.idf,
                )
                context = "\n\n".join(
                    f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(top_chunks)
                )
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.history
                ]
                client = anthropic.Anthropic(api_key=API_KEY)
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    system=(
                        "You are a helpful document assistant. "
                        "Answer the user's question using ONLY the document excerpts provided below. "
                        "If the answer is not in the excerpts, say: 'I could not find that information in the document.'\n\n"
                        f"DOCUMENT EXCERPTS:\n{context}"
                    ),
                    messages=messages,
                )
                answer = response.content[0].text

            st.write(answer)

            with st.expander("📑 View source excerpts"):
                for i, chunk in enumerate(top_chunks):
                    st.caption(f"Excerpt {i+1}")
                    st.text(chunk[:400] + ("..." if len(chunk) > 400 else ""))

        st.session_state.history.append({"role": "assistant", "content": answer})
