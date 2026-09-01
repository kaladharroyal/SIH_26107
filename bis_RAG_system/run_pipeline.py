"""
Interactive CLI Runner for BIS AI Compliance RAG Pipeline.
Usage:
    python run_pipeline.py "what BIS standard should I use for a TMT bar?"
    python run_pipeline.py --interactive
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import BISRAGPipeline


def run_interactive():
    print("\n" + "=" * 70)
    print("      🏛️  BUREAU OF INDIAN STANDARDS (BIS) AI COMPLIANCE ENGINE")
    print("=" * 70)
    print("Type your query below (e.g. standard requirements, schemes, labs, complaints).")
    print("Type 'exit' or 'quit' to stop.\n")

    pipeline = BISRAGPipeline()

    while True:
        try:
            query = input("\n📝 Enter BIS Query > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\nExiting BIS AI Assistant. Goodbye!\n")
                break

            print("\n⏳ Processing query through BIS RAG Pipeline...")
            res = pipeline.query(query)

            print("\n" + "-" * 70)
            print(f"🎯 INTENT:     {res.get('intent')}")
            print(f"🚀 FLOW USED:  {res.get('flow_used')}")
            print(f"📊 CONFIDENCE: {res.get('confidence_score', 0.0):.4f}")
            print(f"⚙️  STATUS:     {res.get('status').upper()}")
            print("-" * 70)
            print("\n" + res.get("response", "") + "\n")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\n\nSession terminated by user.")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")


def main():
    parser = argparse.ArgumentParser(description="BIS AI Compliance RAG Pipeline Runner")
    parser.add_argument("query", nargs="?", default=None, help="Single query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive CLI session")

    args = parser.parse_args()

    if args.interactive or not args.query:
        run_interactive()
    else:
        pipeline = BISRAGPipeline()
        res = pipeline.query(args.query)
        print("\n" + "=" * 70)
        print(f"🎯 INTENT:     {res.get('intent')}")
        print(f"🚀 FLOW USED:  {res.get('flow_used')}")
        print(f"📊 CONFIDENCE: {res.get('confidence_score', 0.0):.4f}")
        print(f"⚙️  STATUS:     {res.get('status').upper()}")
        print("=" * 70 + "\n")
        print(res.get("response", "") + "\n")


if __name__ == "__main__":
    main()
