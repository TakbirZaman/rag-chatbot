# RAG Chatbot

A document Q&A chatbot using Retrieval-Augmented Generation (RAG) with Claude and Streamlit.

Upload a PDF or TXT file, and ask questions about its content.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

**Local development** — create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Streamlit Cloud** — add the key to `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

### 3. Run the app

```bash
streamlit run rag.py
```

## How it works

1. Upload a PDF or TXT document
2. The app splits it into chunks and builds a TF-IDF search index
3. When you ask a question, the most relevant chunks are retrieved
4. Claude answers using only those document excerpts as context
