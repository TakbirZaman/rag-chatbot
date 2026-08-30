import os
import re
import math
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import anthropic

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

try:
    API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

MODEL = "claude-sonnet-5"
MAX_HISTORY_TURNS = 6

# ── Text Helpers ──────────────────────────────────────────────────────────────
def extract_text(uploaded_file) -> str:
    file_type = getattr(uploaded_file, "type", None)
    name = getattr(uploaded_file, "name", "")
    if file_type == "application/pdf" or name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    raw = uploaded_file.read()
    return raw.decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return re.sub(r" +", " ", text).strip()


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_index(chunks: list[str]):
    df: dict[str, int] = {}
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
        tf: dict[str, int] = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        vectors.append({w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()})
    return vectors, idf


def cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v**2 for v in a.values()))
    nb = math.sqrt(sum(v**2 for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def retrieve(
    query: str,
    chunks: list[str],
    vectors: list[dict],
    idf: dict,
    k: int = 4,
) -> list[str]:
    tokens = tokenize(query)
    if not tokens:
        return chunks[:k]
    tf: dict[str, int] = {}
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    qvec = {w: (c / len(tokens)) * idf.get(w, 0) for w, c in tf.items()}
    scores = sorted(
        enumerate(vectors), key=lambda x: cosine(qvec, x[1]), reverse=True
    )
    return [chunks[i] for i, _ in scores[:k]]


def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    if len(history) <= max_turns * 2:
        return history
    return history[-(max_turns * 2) :]


# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 RAG Chatbot — Document Q&A")
st.caption("Upload a PDF or TXT file, then ask questions about it.")

# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "chunks": [],
    "vectors": [],
    "idf": {},
    "history": [],
    "doc_name": "",
    "file_hash": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])

    if uploaded:
        file_hash = getattr(uploaded, "name", "") + str(getattr(uploaded, "size", 0))

        if st.button("Index Document", type="primary"):
            with st.spinner("Reading document..."):
                try:
                    text = extract_text(uploaded)
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
                    st.stop()

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
                st.session_state.file_hash = file_hash
                st.session_state.history = []
                st.success(f"✅ Indexed {len(chunks)} chunks from **{uploaded.name}**")
                st.rerun()

    if st.session_state.doc_name:
        st.info(f"📄 **{st.session_state.doc_name}**\n\n{len(st.session_state.chunks)} chunks loaded")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                for key in defaults:
                    st.session_state[key] = defaults[key]
                st.rerun()

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Upload a document\n"
        "2. Click *Index Document*\n"
        "3. Ask questions in the chat"
    )


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
            st.error(
                "🔑 **ANTHROPIC_API_KEY is missing.**\n\n"
                "Add it to `.streamlit/secrets.toml` (Streamlit Cloud) or a `.env` file (local)."
            )
            st.stop()

        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):
                top_chunks = retrieve(
                    question,
                    st.session_state.chunks,
                    st.session_state.vectors,
                    st.session_state.idf,
                )
                context = "\n\n---\n\n".join(
                    f"[Excerpt {i + 1}]\n{c}" for i, c in enumerate(top_chunks)
                )
                trimmed = trim_history(st.session_state.history)
                messages = [{"role": m["role"], "content": m["content"]} for m in trimmed]

                try:
                    client = anthropic.Anthropic(api_key=API_KEY)
                    response = client.messages.create(
                        model=MODEL,
                        max_tokens=1500,
                        system=(
                            "You are a helpful document assistant. "
                            "Answer the user's question using ONLY the document excerpts provided below. "
                            "If the answer is not in the excerpts, say so clearly.\n\n"
                            f"DOCUMENT EXCERPTS:\n{context}"
                        ),
                        messages=messages,
                    )
                    answer = response.content[0].text
                except anthropic.AuthenticationError:
                    answer = "🔑 Authentication failed. Please check your ANTHROPIC_API_KEY."
                except anthropic.RateLimitError:
                    answer = "⏳ Rate limited. Please wait a moment and try again."
                    st.toast("Rate limited — try again shortly.", icon="⏳")
                except anthropic.APIError as e:
                    answer = f"⚠️ API error: {e.message}"
                except Exception as e:
                    answer = f"Unexpected error: {e}"

            st.write(answer)

            with st.expander("📑 Source excerpts"):
                for i, chunk in enumerate(top_chunks):
                    st.caption(f"Excerpt {i + 1}")
                    st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))

        st.session_state.history.append({"role": "assistant", "content": answer})
