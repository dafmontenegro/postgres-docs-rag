"""
ingest.py

Reads data/raw_docs.json produced by scraper.py, splits it into chunks
(one chunk per section/subsection), and indexes them into a local
ChromaDB collection.

Chunking strategy: section boundaries as defined by the PostgreSQL TOC.
Each section or subsection maps to exactly one chunk, preserving the
authorial structure — code blocks and tables are never cut in half.

Run:
    python3 ingest.py

The ChromaDB collection is persisted to chroma_db/ and reused by rag.py.
Re-running this script drops and recreates the collection.
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

RAW_DOCS = Path("data/raw_docs.json")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pg_docs"

# sentence-transformers model — fast on CPU/M1, no API key required
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def flatten_sections(chapter_title: str, chapter: dict) -> list[dict]:
    """
    Recursively walk sections and subsections, yielding flat records
    with all metadata attached. Empty sections are skipped.
    """
    records = []

    def _walk(sections: dict, depth: int) -> None:
        for sec_num, sec in sections.items():
            content = sec.get("content_markdown", "").strip()
            if content:
                records.append({
                    "id": f"{chapter['chapter_number']}.{sec_num}",
                    "content": content,
                    "chapter": chapter_title,
                    "chapter_number": chapter["chapter_number"],
                    "section_number": sec_num,
                    "title": sec["title"],
                    "source_url": sec["source_url"],
                    "depth": depth,
                })
            _walk(sec.get("subsections", {}), depth + 1)

    _walk(chapter["sections"], depth=1)
    return records


def main() -> None:
    raw = json.loads(RAW_DOCS.read_text("utf-8"))

    all_records: list[dict] = []
    for chapter_title, chapter in raw.items():
        all_records.extend(flatten_sections(chapter_title, chapter))

    print(f"Chunks to index: {len(all_records)}")

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop collection if it already exists so re-runs start clean
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Dropped existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [r["id"] for r in all_records]
    documents = [r["content"] for r in all_records]
    metadatas = [
        {
            "chapter": r["chapter"],
            "chapter_number": r["chapter_number"],
            "section_number": r["section_number"],
            "title": r["title"],
            "source_url": r["source_url"],
            "depth": r["depth"],
        }
        for r in all_records
    ]

    # ChromaDB recommends batching for large collections
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        print(f"  indexed {min(i + batch_size, len(ids))}/{len(ids)} chunks …")

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")
    print(f"Persisted to {CHROMA_PATH}/")


if __name__ == "__main__":
    main()
