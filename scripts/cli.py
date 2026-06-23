#!/usr/bin/env python3
"""
CLI tool for ingesting documents and querying EnterpriseRAG locally.
Usage:
    python scripts/cli.py ingest --file path/to/doc.pdf
    python scripts/cli.py ingest --url https://example.com
    python scripts/cli.py ingest --dir ./documents/
    python scripts/cli.py query "What is our refund policy?"
    python scripts/cli.py stats
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.orchestrator import EnterpriseRAG


def cmd_ingest(args, rag: EnterpriseRAG):
    if args.file:
        result = rag.ingest_file(args.file)
        print(f"✓ Ingested {args.file}")
        print(f"  Documents: {result['documents']}")
        print(f"  Chunks:    {result['chunks']}")
        print(f"  Time:      {result.get('duration', '?')}s")

    elif args.url:
        result = rag.ingest_url(args.url)
        print(f"✓ Ingested {args.url}")
        print(f"  Chunks: {result['chunks']}")

    elif args.dir:
        result = rag.ingest_directory(args.dir)
        print(f"✓ Ingested directory: {args.dir}")
        print(f"  Documents: {result['documents']}")
        print(f"  Chunks:    {result['chunks']}")
        print(f"  Time:      {result.get('duration', '?')}s")

    else:
        print("Specify --file, --url, or --dir")


def cmd_query(args, rag: EnterpriseRAG):
    print(f"\n📝 Query: {args.query}\n{'─'*60}")
    result = rag.query_sync(args.query, use_hyde=not args.no_hyde)

    print(f"\n💬 Answer:\n{result['answer']}")

    print(f"\n📚 Sources ({len(result['sources'])}):")
    for src in result["sources"]:
        print(f"  [{src['index']}] {src['source_name']} (score: {src['relevance_score']:.3f})")
        print(f"      {src['preview'][:80]}...")

    print(f"\n⏱  Latencies:")
    for stage, t in result["latencies"].items():
        print(f"  {stage:15} {t:.3f}s")

    grounded = "✓ Grounded" if result["is_grounded"] else "⚠ Hallucination risk"
    print(f"\n{grounded}")


def cmd_stats(args, rag: EnterpriseRAG):
    stats = rag.get_stats()
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="EnterpriseRAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    ingest_p = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_p.add_argument("--file", type=str, help="Path to file")
    ingest_p.add_argument("--url", type=str, help="URL to ingest")
    ingest_p.add_argument("--dir", type=str, help="Directory to ingest")

    query_p = subparsers.add_parser("query", help="Query the knowledge base")
    query_p.add_argument("query", type=str, help="Query string")
    query_p.add_argument("--no-hyde", action="store_true", help="Disable HyDE expansion")

    subparsers.add_parser("stats", help="Show vector store stats")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    rag = EnterpriseRAG()

    if args.command == "ingest":
        cmd_ingest(args, rag)
    elif args.command == "query":
        cmd_query(args, rag)
    elif args.command == "stats":
        cmd_stats(args, rag)


if __name__ == "__main__":
    main()
