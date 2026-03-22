"""
ingest.py — Add a PDF paper to the ChromaDB vector database

Usage:
    python3 ingest.py "Papers/my_paper.pdf"

What it does:
  1. Extracts text from each page of the PDF (preserving page numbers)
  2. Splits each page into overlapping chunks so context isn't lost at boundaries
  3. Embeds each chunk using a local sentence-transformer model
  4. Stores chunks + embeddings + metadata (paper name, page number) in ChromaDB

The database is persisted to the db/ folder. You can add more papers over time
by running this script again — it skips papers that are already in the database.
"""

import os
import sys
import hashlib

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DB_PATH      = os.path.join(os.path.dirname(__file__), "db")
CHUNK_SIZE   = 600   # characters per chunk
CHUNK_OVERLAP = 100  # overlap between consecutive chunks
EMBED_MODEL  = "all-MiniLM-L6-v2"  # fast, free, runs locally


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """
    Split a page's text into overlapping chunks.

    Overlap ensures that a sentence split across a chunk boundary
    still appears fully in at least one chunk.
    """
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_paper(pdf_path: str):
    paper_name = os.path.basename(pdf_path)

    # Load embedding model (downloads once, cached locally after first run)
    print(f"Loading embedding model: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    # Connect to (or create) the persistent ChromaDB database
    db = chromadb.PersistentClient(path=DB_PATH)
    collection = db.get_or_create_collection(
        name="papers",
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )

    # Skip if this paper is already in the database
    existing = collection.get(where={"paper": paper_name})
    if existing["ids"]:
        print(f"'{paper_name}' already in database ({len(existing['ids'])} chunks). Skipping.")
        return

    print(f"\nProcessing: {paper_name}")

    chunks, metadatas, ids = [], [], []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages: {total_pages}\n")

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            page_chunks = chunk_text(text)

            for chunk_idx, chunk in enumerate(page_chunks):
                # Deterministic ID based on paper + page + position
                chunk_id = hashlib.md5(
                    f"{paper_name}-p{page_num}-c{chunk_idx}".encode()
                ).hexdigest()

                chunks.append(chunk)
                metadatas.append({
                    "paper":        paper_name,
                    "page":         page_num,
                    "total_pages":  total_pages,
                    "chunk_index":  chunk_idx,
                })
                ids.append(chunk_id)

            print(f"  Processed page {page_num}/{total_pages} — {len(page_chunks)} chunks", end="\r")

    print(f"\n\nEmbedding {len(chunks)} chunks (this may take a moment)...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    # Insert into ChromaDB in batches to avoid memory issues with large papers
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=chunks[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    print(f"\nDone. Added {len(chunks)} chunks from '{paper_name}'.")
    print(f"Database now contains {collection.count()} total chunks across all papers.")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ingest.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    ingest_paper(path)
