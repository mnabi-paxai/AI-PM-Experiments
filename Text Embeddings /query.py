"""
query.py — Ask questions across all ingested papers

Usage:
    python3 query.py

What it does:
  1. Embeds your question using the same model used during ingestion
  2. Searches ChromaDB for the 5 most semantically similar chunks
  3. Sends those chunks + your question to Claude (via AWS Bedrock)
  4. Returns a cited answer with paper name and page number for every claim

Run this anytime after ingesting at least one paper.
Type 'exit' to quit the interactive loop.
"""

import os

import chromadb
from anthropic import AnthropicBedrock
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DB_PATH       = os.path.join(os.path.dirname(__file__), "db")
EMBED_MODEL   = "all-MiniLM-L6-v2"
CLAUDE_MODEL  = "us.anthropic.claude-sonnet-4-20250514-v1:0"
TOP_K         = 5  # number of chunks to retrieve per query


# ── Query ─────────────────────────────────────────────────────────────────────

def ask(question: str, model: SentenceTransformer, collection) -> None:
    # 1. Embed the question using the same model used at ingestion time
    question_embedding = model.encode([question]).tolist()

    # 2. Find the TOP_K most similar chunks in the database
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # 3. Format retrieved chunks with source citations for the prompt
    context_blocks = []
    print("\nRetrieved chunks:")
    print("─" * 50)
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        paper = meta["paper"].replace(".pdf", "")
        page  = meta["page"]
        score = round(1 - dist, 3)   # cosine distance → similarity score
        print(f"  [{i}] {paper} — Page {page}  (similarity: {score})")
        context_blocks.append(f"[Source: {paper}, Page {page}]\n{doc}")

    context = "\n\n---\n\n".join(context_blocks)

    # 4. Send to Claude with instructions to cite sources
    prompt = f"""You are a research assistant helping analyze academic papers.

Answer the question below using ONLY the provided context excerpts.
For every claim or piece of information you use, cite the source inline \
in the format: (Paper Name, p. X).
If the context does not contain enough information to answer the question fully, \
say so explicitly rather than guessing.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    print("\nGenerating answer with Claude...\n")

    bedrock  = AnthropicBedrock()
    response = bedrock.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.content[0].text

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load model and DB once, reuse across all questions in the session
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    db = chromadb.PersistentClient(path=DB_PATH)
    collection = db.get_or_create_collection(
        name="papers",
        metadata={"hnsw:space": "cosine"},
    )

    count = collection.count()
    if count == 0:
        print("Database is empty. Run: python3 ingest.py <path_to_pdf>")
        exit(1)

    print(f"Database ready — {count} chunks across all papers.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        ask(question, embed_model, collection)
        print()
