# RagFlow Quick Start Guide

## Step-by-Step Setup

### 1. Install Dependencies

```bash
cd RagFlow
uv sync
```

This will install:
- `chromadb` - Vector database
- `sentence-transformers` - For embeddings
- `requests` - For LM Studio API calls
- `python-dotenv` - For environment variables

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` if needed (defaults work for local LM Studio).

### 3. Start LM Studio

1. Download [LM Studio](https://lmstudio.ai/)
2. Install and open LM Studio
3. Download a model (e.g., Llama 2, Mistral, etc.)
4. Load the model
5. Click "Start Server" (usually runs on `http://localhost:1234`)

### 4. Index Your Documents

```bash
# Create a sample document
echo "Python is a versatile programming language used for web development, data science, and AI." > sample.txt

# Index it
uv run python -m src.main index sample.txt
```

### 5. Query the System

```bash
uv run python -m src.main query "What is Python used for?"
```

## Python API Example

```python
from src.orchestrator import RAGOrchestrator

# Initialize
rag = RAGOrchestrator()

# Index documents
rag.index_documents(file_paths=["doc1.txt", "doc2.txt"])

# Query
result = rag.query("Your question here")
print(result["answer"])
```

## Troubleshooting

**LM Studio not responding:**
- Make sure LM Studio is running
- Check the server is enabled (Settings → Local Server)
- Verify the URL in `.env` matches LM Studio's port

**No chunks retrieved:**
- Make sure documents are indexed first
- Check `uv run python -m src.main stats` to see chunk count

**Import errors:**
- Run `uv sync` to ensure all dependencies are installed
- Make sure you're in the RagFlow directory

