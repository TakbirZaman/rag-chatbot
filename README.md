# RAG Chatbot — PDF/Document Q&A

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
streamlit run rag.py
```

## How it works
1. Upload PDF/TXT → split into 1000-char chunks with 150-char overlap
2. Embed chunks with SentenceTransformer (local, free)
3. Store vectors in ChromaDB (in-memory)
4. On question: embed query → cosine similarity → top-4 chunks → Claude answers

## Architecture
```
PDF Upload
    ↓
PyPDFLoader → RecursiveTextSplitter (1000 tok, 150 overlap)
    ↓
SentenceTransformer embeddings → ChromaDB
    ↓
User Query → embed → similarity search → top-k chunks
    ↓
Claude (claude-sonnet-4-20250514) → Answer + Sources
```

## Customization
- Change chunk_size/overlap in `RecursiveCharacterTextSplitter`
- Swap `SentenceTransformerEmbeddings` for `OpenAIEmbeddings` if preferred
- Adjust `k=4` in retriever for more/fewer context chunks
- Swap Chroma for Pinecone/Weaviate for production persistence
