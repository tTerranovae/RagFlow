"""Main entry point for RagFlow CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.orchestrator import RAGOrchestrator


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RagFlow - Custom RAG system with LM Studio"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument(
        "files",
        nargs="+",
        help="Files to index",
    )
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size (default: 500)",
    )
    index_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap (default: 50)",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument(
        "query",
        help="Query text",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)",
    )
    query_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    query_parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens to generate (default: 500)",
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize orchestrator
    rag = RAGOrchestrator(
        chunk_size=args.chunk_size if hasattr(args, "chunk_size") else None,
        chunk_overlap=args.chunk_overlap if hasattr(args, "chunk_overlap") else None,
        top_k=args.top_k if hasattr(args, "top_k") else None,
    )

    # Execute command
    if args.command == "index":
        # Validate files exist
        for file_path in args.files:
            if not Path(file_path).exists():
                print(f"Error: File not found: {file_path}", file=sys.stderr)
                sys.exit(1)

        print(f"Indexing {len(args.files)} file(s)...")
        count = rag.index_documents(file_paths=args.files)
        print(f"✓ Indexed {count} chunks")

    elif args.command == "query":
        # Check if LM Studio is accessible
        if not rag.generator.check_health():
            print(
                "Warning: LM Studio may not be running. Make sure it's started on",
                rag.generator.base_url,
                file=sys.stderr,
            )

        print(f"Querying: {args.query}\n")
        result = rag.query(
            query=args.query,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        print("Answer:")
        print(result["answer"])
        print(f"\nRetrieved {result['metadata']['retrieved_count']} context chunks")

    elif args.command == "stats":
        stats = rag.get_stats()
        print("RAG System Statistics:")
        print(f"  Total chunks indexed: {stats['total_chunks']}")
        print(f"  Embedding model: {stats['embedding_model']}")
        print(f"  Chunk size: {stats['chunk_size']}")
        print(f"  Chunk overlap: {stats['chunk_overlap']}")
        print(f"  Top-K retrieval: {stats['top_k']}")


if __name__ == "__main__":
    main()

