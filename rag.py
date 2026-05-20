"""
RAG Chatbot — 100% pure Python, no heavy dependencies
Chunking: manual | Embeddings: TF-IDF cosine in pure Python | LLM: Anthropic
"""
import os, re, math, tempfile
import streamlit as st

os.environ["ANTHROPIC_API_KEY"] = st.secrets.get(
    "ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")
)

import anthropic
from pypdf import PdfReader

# ── pure-python RAG helpers ───────────────────────────────────────────────────

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")

def chunk_text(text, size=800, overlap=100):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

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
        tf = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        vec = {w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()}
        vectors.append(vec)
    return vectors, idf

def cosine(a, b):
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v**2 for v in a.values()))
    nb = math.sqrt(sum(v**2 for v in b.values()))
    return dot / (na * nb + 1e-9)

def retrieve(query, chunks, vectors, idf, k=4):
    tokens = tokenize(query)
    tf = {}
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    qvec = {w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()}
    scores = [(cosine(qvec, v), i) for i, v in enumerate(vectors)]
    scores.sort(reverse=True)
    return [chunks[i] for _, i in scores[:k]]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Document Q&A — RAG Chatbot")
st.caption("Upload a PDF or text file, then ask questions about it.")

for key in ["chunks", "vectors", "idf", "history", "doc_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "history" else []

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt", "md"])

    if uploaded and st.button("Index Document"):
        with st.spinner("Reading + indexing..."):
            text = extract_text(uploaded)
            if not text.strip():
                st.error("Could not extract text from this file.")
            else:
                chunks = chunk_text(text)
                vectors, idf = build_index(chunks)
                st.session_state.chunks = chunks
                st.session_state.vectors = vectors
                st.session_state.idf = idf
                st.session_state.doc_name = uploaded.name
                st.session_state.history = []
                st.success(f"✅ {len(chunks)} chunks indexed from '{uploaded.name}'")

    if st.session_state.doc_name:
        st.info(f"📄 Active: {st.session_state.doc_name}")

# ── chat UI ───────────────────────────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your document..."):
    if not st.session_state.chunks:
        st.warning("Please upload and index a document first.")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                top_chunks = retrieve(
                    prompt,
                    st.session_state.chunks,
                    st.session_state.vectors,
                    st.session_state.idf
                )
                context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(top_chunks))
                recent = st.session_state.history[-6:]
                messages = [{"role": m["role"], "content": m["content"]} for m in recent]

                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    system=f"""You are a document assistant. Answer using ONLY the excerpts below.
If the answer is not in the excerpts, say so clearly.

DOCUMENT EXCERPTS:
{context}""",
                    messages=messages
                )
                answer = response.content[0].text

            st.write(answer)

            with st.expander("📑 Source chunks used"):
                for i, chunk in enumerate(top_chunks):
                    st.caption(f"Chunk {i+1}")
                    st.text(chunk[:300] + "...")

        st.session_state.history.append({"role": "assistant", "content": answer})
