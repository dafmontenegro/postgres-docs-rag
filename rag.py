"""
rag.py

CLI query interface for the PostgreSQL documentation RAG system.

Usage:
    python3 rag.py "What index type should I use for equality comparisons?"
    python3 rag.py "What is pg_dump?" --top-k 5
    python3 rag.py "..." --no-generate   # retrieval only, no API call

Generation requires an API key: GEMINI_API_KEY (used first, free tier) or
ANTHROPIC_API_KEY (fallback). Without either (or with --no-generate), the
system returns the top retrieved chunks with full source citations — which
already satisfies the "cite the exact source" requirement and is useful for
evaluation purposes.

Retrieval confidence: if the best cosine distance is above DISTANCE_THRESHOLD
the system considers the question out of scope and says so, rather than
generating a potentially hallucinated answer.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))   # searches cwd and all parent dirs
except ImportError:
    pass   # python-dotenv optional; export GEMINI_API_KEY / ANTHROPIC_API_KEY manually instead

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pg_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4

# Cosine distance (1 - similarity). If the closest chunk is farther than
# this threshold we report "not found" instead of guessing.
#
# Calibrated on eval_set.json (22 questions): the 12 answerable ones land at
# 0.204-0.431, the 10 out-of-corpus ones at 0.514-0.955. 0.47 is the midpoint
# of that gap. An earlier value of 0.58 sat inside the negatives' range and
# accepted Q15 (pg_upgrade, 0.520) and Q16 (pgbench, 0.514) as answerable.
# This is a heuristic on a small author-written set, not a measured optimum.
DISTANCE_THRESHOLD = 0.47


def get_collection() -> chromadb.Collection:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Query ChromaDB and return the top_k most relevant chunks with metadata.
    Returns a dict with keys: chunks (list), distances (list), in_scope (bool).
    """
    collection = get_collection()
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    in_scope = bool(docs) and distances[0] < DISTANCE_THRESHOLD

    chunks = [
        {
            "content": doc,
            "chapter": meta["chapter"],
            "chapter_number": meta["chapter_number"],
            "section_number": meta["section_number"],
            "title": meta["title"],
            "source_url": meta["source_url"],
            "distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, distances)
    ]

    return {"chunks": chunks, "distances": distances, "in_scope": in_scope}


def format_citation(chunk: dict) -> str:
    return (
        f"Chapter {chunk['chapter_number']} ({chunk['chapter']}), "
        f"Section {chunk['section_number']}: {chunk['title']} — "
        f"{chunk['source_url']}"
    )


def _build_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_message) shared across providers."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        citation = format_citation(chunk)
        context_parts.append(f"[Source {i}: {citation}]\n\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a PostgreSQL documentation assistant. "
        "Answer the user's question using ONLY the provided documentation excerpts. "
        "If the excerpts do not contain enough information to answer, say so clearly. "
        "Always end your answer with a 'Sources:' section listing the exact sections used."
    )
    user_message = f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}"
    return system_prompt, user_message


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Generate a grounded answer from retrieved chunks.
    Uses GEMINI_API_KEY (gemini-flash-lite-latest) if set, otherwise falls back to
    ANTHROPIC_API_KEY (claude-haiku-4-5-20251001).
    """
    system_prompt, user_message = _build_prompt(question, chunks)

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
        except ImportError:
            return "[google-genai not installed — run: pip install google-genai]"

        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=f"{system_prompt}\n\n{user_message}",
        )
        return response.text

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
        except ImportError:
            return "[anthropic not installed — run: pip install anthropic]"

        client = anthropic.Anthropic(api_key=anthropic_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    return "[No API key found — set GEMINI_API_KEY or ANTHROPIC_API_KEY]"


def answer(question: str, top_k: int = DEFAULT_TOP_K, no_generate: bool = False) -> str:
    """
    Full RAG pipeline: retrieve → (optionally) generate → return answer with citations.
    """
    retrieval = retrieve(question, top_k=top_k)

    if not retrieval["chunks"]:
        return "No relevant sections found in the indexed documentation."

    if not retrieval["in_scope"]:
        best_dist = retrieval["distances"][0]
        return (
            f"I could not find relevant information in the indexed chapters "
            f"(Chapters 11, 12, 24, 25) to answer this question.\n"
            f"(Closest match distance: {best_dist:.3f}, threshold: {DISTANCE_THRESHOLD})\n\n"
            f"This question may be outside the scope of the indexed documentation."
        )

    chunks = retrieval["chunks"]

    if no_generate or not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")):
        # Retrieval-only mode: format the most relevant chunk with all citations
        lines = [
            f"Most relevant section: {format_citation(chunks[0])}",
            "",
            chunks[0]["content"][:1200] + ("…" if len(chunks[0]["content"]) > 1200 else ""),
            "",
            "Sources:",
        ]
        for chunk in chunks:
            lines.append(f"  - {format_citation(chunk)}")
        return "\n".join(lines)

    generated = generate_answer(question, chunks)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the PostgreSQL documentation RAG system."
    )
    parser.add_argument("question", help="The question to answer")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Skip generation, return retrieved chunks only"
    )
    args = parser.parse_args()

    print(f"\nQuestion: {args.question}\n")
    print("=" * 60)
    result = answer(args.question, top_k=args.top_k, no_generate=args.no_generate)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
