"""Example usage of RagFlow."""

from src.orchestrator import RAGOrchestrator


def main() -> None:
    """Example RAG workflow."""
    # Initialize RAG system
    rag = RAGOrchestrator(
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
    )

    # Option 1: Index from files
    print("Indexing documents from files...")
    chunk_count = rag.index_documents(
        file_paths=["example_doc.txt"]  # Replace with your file paths
    )
    print(f"Indexed {chunk_count} chunks\n")

    # Option 2: Index from text strings
    print("Indexing documents from text...")
    texts = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "RAG systems combine retrieval and generation for better answers.",
    ]
    chunk_count = rag.index_documents(texts=texts)
    print(f"Indexed {chunk_count} chunks\n")

    # Query the system
    print("Querying the RAG system...")
    result = rag.query(
        query="What is Python?",
        temperature=0.7,
        max_tokens=200,
    )

    print(f"Answer: {result['answer']}\n")
    print(f"Retrieved {result['metadata']['retrieved_count']} context chunks")

    # Show statistics
    stats = rag.get_stats()
    print(f"\nTotal chunks in database: {stats['total_chunks']}")


if __name__ == "__main__":
    main()

