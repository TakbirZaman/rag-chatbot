import os
import re
import math
import streamlit as st
from pypdf import PdfReader
import anthropic

try:
    API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        pages = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")

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
        vec = {w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()}
        vectors.append(vec)
    return vectors, idf

def cosine(a, b):
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v ** 2 for v in a.values()))
    nb = math.sqrt(sum(v ** 2 for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

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

def clean_text(text):
    # remove null bytes and non-printable characters that cause API errors
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

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
                raw = extract_text(uploaded)
                text = clean_text(raw)
            if not text.strip():
                st.error("Could not extract any text from this file.")
            else:
                with st.spinner("Building search index..."):
                    chunks = chunk_text(text)
                    chunks = [clean_text(c) for c in chunks]
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
            st.error("ANTHROPIC_API_KEY missing. Add it in Streamlit Secrets.")
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

                    # keep context under 3000 chars to avoid token issues
                    context_parts = []
                    total = 0
                    for i, c in enumerate(top_chunks):
                        part = f"[Excerpt {i+1}]\n{c[:800]}"
                        if total + len(part) > 3000:
                            break
                        context_parts.append(part)
                        total += len(part)
                    context = "\n\n".join(context_parts)

                    question_clean = clean_text(question)[:500]

                    prompt = (
                        f"Question: {question_clean}\n\n"
                        f"Answer using ONLY these excerpts from the document. "
                        f"If not found, say so.\n\n"
                        f"EXCERPTS:\n{context}"
                    )

                    client = anthropic.Anthropic(api_key=API_KEY)
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=800,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    answer = response.content[0].text

                except anthropic.BadRequestError as e:
                    answer = f"API error: {str(e)}\n\nTry asking a shorter or simpler question."
                except Exception as e:
                    answer = f"Error: {str(e)}"

            st.write(answer)
            with st.expander("📑 Source excerpts"):
                for i, chunk in enumerate(top_chunks):
                    st.caption(f"Excerpt {i+1}")
                    st.text(chunk[:300] + "...")

        st.session_state.history.append({"role": "assistant", "content": answer})
