import os
import re
import math
import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

# ── API Key ───────────────────────────────────────────────────────────────────
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")

def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return re.sub(r" +", " ", text).strip()

def chunk_text(text, size=600, overlap=80):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_index(chunks):
    df = {}
    for chunk in chunks:
        for word in set(tokenize(chunk)):
            df[word] = df.get(word, 0) + 1
    N = max(len(chunks), 1)
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
        vectors.append({w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()})
    return vectors, idf

def cosine(a, b):
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v**2 for v in a.values()))
    nb = math.sqrt(sum(v**2 for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def retrieve(query, chunks, vectors, idf, k=3):
    tokens = tokenize(query)
    if not tokens:
        return chunks[:k]
    tf = {}
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    qvec = {w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()}
    scores = sorted(enumerate(vectors), key=lambda x: cosine(qvec, x[1]), reverse=True)
    return [chunks[i] for i, _ in scores[:k]]

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 RAG Chatbot — Document Q&A")
st.caption("Upload a PDF or TXT file, then ask questions about it.")

for key in ["chunks", "vectors", "idf", "history", "doc_name"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "doc_name" else ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])
    if uploaded:
        if st.button("Index Document", type="primary"):
            with st.spinner("Reading and indexing..."):
                text = clean_text(extract_text(uploaded))
            if not text.strip():
                st.error("Could not extract any text from this file.")
            else:
                with st.spinner("Building search index..."):
                    chunks = [clean_text(c) for c in chunk_text(text)]
                    vectors, idf = build_index(chunks)
                st.session_state.chunks = chunks
                st.session_state.vectors = vectors
                st.session_state.idf = idf
                st.session_state.doc_name = uploaded.name
                st.session_state.history = []
                st.success(f"✅ Done! {len(chunks)} chunks indexed.")
    if st.session_state.doc_name:
        st.info(f"Active: **{st.session_state.doc_name}**")
        if st.button("Clear"):
            for k in ["chunks", "vectors", "idf", "history"]:
                st.session_state[k] = []
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
            st.error("GEMINI_API_KEY missing. Add it in Streamlit Secrets.")
            st.stop()

        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    top_chunks = retrieve(
                        question,
                        st.session_state.chunks,
                        st.session_state.vectors,
                        st.session_state.idf,
                    )
                    context = "\n\n".join(
                        f"[Excerpt {i+1}]\n{c[:800]}" for i, c in enumerate(top_chunks)
                    )
                    prompt = (
                        f"Answer this question using ONLY the document excerpts below.\n"
                        f"If the answer is not found, say: 'I could not find that in the document.'\n\n"
                        f"Question: {question[:500]}\n\n"
                        f"EXCERPTS:\n{context}"
                    )
                    genai.configure(api_key=API_KEY)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    answer = response.text

                except Exception as e:
                    answer = f"Error: {str(e)}"

            st.write(answer)
            with st.expander("📑 Source excerpts"):
                for i, chunk in enumerate(top_chunks):
                    st.caption(f"Excerpt {i+1}")
                    st.text(chunk[:300] + "...")

        st.session_state.history.append({"role": "assistant", "content": answer})
