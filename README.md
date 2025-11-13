# RagFlow 🚀

A custom Retrieval-Augmented Generation (RAG) system that:
1. Chunks and vectorizes documents
2. Stores embeddings in ChromaDB vector database
3. Retrieves relevant chunks using cosine similarity
4. Generates responses using LM Studio

## Architecture

```
Document → Chunker → Embedder → Vector Store (ChromaDB)
                                              ↓
Query → Embedder → Retriever (Cosine Similarity) → Top K Chunks → LM Studio → Response
```

## Components

- **chunking**: Text splitting with overlap
- **embedding**: Sentence transformer embeddings
- **vectorstore**: ChromaDB integration
- **retriever**: Cosine similarity search
- **generator**: LM Studio API client
- **orchestrator**: End-to-end RAG pipeline

## Setup

1. **Install dependencies:**
```bash
cd RagFlow
uv sync
```

2. **Set up environment variables** (create `.env` file):
```bash
cp .env.example .env
```

Edit `.env` and configure:
- `LM_STUDIO_BASE_URL`: Your LM Studio API URL (default: http://localhost:1234)
- `CHROMA_DB_PATH`: Where to store the vector database
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)

3. **Start LM Studio:**
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Start LM Studio and load a model
   - Enable the local server (usually runs on port 1234)

## Usage

### Command Line Interface

**Index documents:**
```bash
uv run python -m src.main index document1.txt document2.txt
```

**Query the system:**
```bash
uv run python -m src.main query "What is the main topic?"
```

**View statistics:**
```bash
uv run python -m src.main stats
```

### Python API

```python
from src.orchestrator import RAGOrchestrator

# Initialize
rag = RAGOrchestrator(
    chunk_size=500,
    chunk_overlap=50,
    top_k=3,
)

# Index documents
rag.index_documents(file_paths=["doc1.txt", "doc2.txt"])

# Or index from text strings
rag.index_documents(texts=["Text content here..."])

# Query
result = rag.query("What is Python?")
print(result["answer"])

# Get statistics
stats = rag.get_stats()
print(f"Total chunks: {stats['total_chunks']}")
```

See `example.py` for a complete example.

## How It Works

1. **Chunking**: Documents are split into overlapping chunks (default: 500 chars, 50 overlap)
2. **Embedding**: Each chunk is vectorized using sentence transformers
3. **Storage**: Embeddings are stored in ChromaDB with metadata
4. **Retrieval**: Query is embedded and cosine similarity finds top-K chunks
5. **Generation**: Retrieved chunks are sent to LM Studio as context for answer generation

