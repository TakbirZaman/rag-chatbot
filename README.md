# RAG Chatbot — Document Q&A

A document Q&A chatbot powered by **Retrieval-Augmented Generation (RAG)** using Claude AI and Streamlit. Upload a PDF or TXT file, and ask natural language questions about its content.

---

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Tools & Technologies](#tools--technologies)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [User Manual](#user-manual)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Demo

```
Upload a PDF → Click "Index Document" → Ask questions → Get AI-powered answers
```

---

## Features

- PDF and plain text file support
- TF-IDF based document search (no external vector DB needed)
- Claude Sonnet 5 for intelligent answer generation
- Source excerpt viewer for transparency
- Chat history with token-safe trimming
- Dark themed UI
- Local `.env` and Streamlit Cloud `secrets.toml` support

---

## Tools & Technologies

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Core language |
| **Streamlit** | >= 1.62.0 | Web UI framework for the chatbot interface |
| **Anthropic SDK** | >= 1.2.0 | Python client for the Claude API |
| **Claude Sonnet 5** | Latest | Large Language Model for answering questions |
| **pypdf** | >= 5.1.0 | PDF text extraction |
| **python-dotenv** | >= 1.2.3 | Loading API keys from `.env` files locally |
| **TF-IDF** | Custom impl. | Document indexing and retrieval (built from scratch with `re`, `math`) |

### APIs

| API | Purpose |
|-----|---------|
| **Anthropic Messages API** | Powers the Claude AI responses |

---

## Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  User Upload │────>│  Text Extract │────>│  Chunking    │
│  (PDF / TXT) │     │  (pypdf)      │     │  (800 words) │
└──────────────┘     └───────────────┘     └──────┬───────┘
                                                  │
                                                  v
                                          ┌───────────────┐
                                          │  TF-IDF Index │
                                          │  Build        │
                                          └───────┬───────┘
                                                  │
                     User Question               │
                          │                      │
                          v                      v
                   ┌─────────────┐    ┌──────────────────┐
                   │ Query Vector│───>│ Cosine Similarity │
                   └─────────────┘    │ Top-K Retrieval   │
                                      └────────┬─────────┘
                                               │
                                               v
                                      ┌────────────────┐
                                      │  Claude API    │
                                      │  (Sonnet 5)    │
                                      │  + Context     │
                                      └────────┬───────┘
                                               │
                                               v
                                        ┌────────────┐
                                        │  Answer +  │
                                        │  Sources   │
                                        └────────────┘
```

---

## Prerequisites

- **Python 3.10** or higher
- An **Anthropic API key** ([Get one here](https://console.anthropic.com/))
- pip (Python package manager)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TakbirZaman/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### Option A: Local Development (`.env` file)

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Option B: Streamlit Secrets (Streamlit Cloud)

Add your key to `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

### Option C: Environment Variable

Set it in your terminal before running:

```bash
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"

# macOS/Linux
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

> **Note:** The app checks Streamlit Secrets first, then falls back to the `.env` file / environment variable.

---

## User Manual

### Step 1: Start the App

```bash
streamlit run rag.py
```

The app opens in your browser at `http://localhost:8501`.

### Step 2: Upload a Document

1. Look at the **left sidebar** labeled "Upload Document"
2. Click **"Choose a file"** and select a `.pdf`, `.txt`, or `.md` file
3. Click the **"Index Document"** button
4. Wait for the spinner to finish — you'll see a success message showing how many chunks were indexed

### Step 3: Ask Questions

1. Type your question in the **chat input** at the bottom of the page
2. Press **Enter** or click **Send**
3. Claude will search the document for relevant information and generate an answer
4. Click **"Source excerpts"** under the answer to see which parts of the document were used

### Step 4: Continue the Conversation

- You can ask **follow-up questions** — the app keeps the last 6 conversation turns for context
- Each answer is grounded in the document excerpts retrieved

### Step 5: Clear or Upload a New Document

- Click **"Clear"** in the sidebar to reset the index and start fresh
- Upload a different file and click "Index Document" to switch documents

### Tips for Best Results

| Do | Don't |
|----|-------|
| Ask specific questions | Ask vague questions like "tell me about stuff" |
| Ask about one topic at a time | Ask multiple unrelated questions in one message |
| Use the document's terminology | Use completely different words than what's in the doc |
| Ask factual questions the document can answer | Ask questions about topics not covered in the doc |

---

## Project Structure

```
rag-chatbot/
├── .env.example          # Template for the API key
├── .gitignore            # Protects secrets and caches
├── .streamlit/
│   ├── config.toml       # Streamlit dark theme + server config
│   └── secrets.toml      # API key (git-ignored)
├── rag.py                # Main application (all logic)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## How It Works

1. **Text Extraction** — The uploaded PDF is parsed with `pypdf` (or read as plain text)
2. **Chunking** — Text is split into ~800-word chunks with 100-word overlap for context continuity
3. **TF-IDF Indexing** — Each chunk is converted to a term frequency–inverse document frequency vector (built from scratch, no external library)
4. **Retrieval** — When you ask a question, your query is vectorized and the top 4 most similar chunks are found via cosine similarity
5. **Generation** — The retrieved chunks are sent to Claude Sonnet 5 as context, along with your question, and Claude generates a grounded answer
6. **History Trimming** — Only the last 6 conversation turns are sent to Claude to stay within token limits

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ANTHROPIC_API_KEY is missing` | Check your `.env` file or `secrets.toml` has the correct key |
| `Authentication failed` | Your API key is invalid — get a new one from [console.anthropic.com](https://console.anthropic.com/) |
| `Rate limited` | Wait a minute and try again |
| `Could not extract any text` | The file may be a scanned PDF (image-based) — OCR is not supported |
| App won't start | Run `pip install -r requirements.txt` to ensure all dependencies are installed |
| `ModuleNotFoundError` | Make sure your virtual environment is activated |

---

## License

MIT
