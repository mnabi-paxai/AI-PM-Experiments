# RAG — Research Paper Q&A System

A persistent, growing knowledge base for academic papers. Ask natural language questions and get cited answers backed by the actual text — with paper name and page number for every claim.

---

## The Problem This Solves

When you read an academic paper, you have to hold the whole thing in your head. When your collection grows to 10, 20, 50 papers, finding a specific idea becomes a manual search problem.

**Keyword search fails here.** If you search for "nondeterminism" you might miss a paragraph that talks about "reproducibility issues caused by floating point operations" — same idea, different words.

**RAG (Retrieval-Augmented Generation) solves this** by searching for *meaning*, not words. You ask a question in plain English. The system finds the most semantically relevant passages across your entire library. An LLM then synthesizes those passages into a direct answer — with citations.

---

## What Are Text Embeddings?

An embedding is a way of converting text into a list of numbers — called a **vector** — that captures the *meaning* of that text.

```
"The model produces nondeterministic outputs"
        ↓  embedding model
[ 0.24, -0.71, 0.03, 0.55, ... ]   ← 384 numbers
```

The critical property: **texts with similar meanings produce vectors that are close together** in the 384-dimensional space. Texts with unrelated meanings produce vectors that are far apart.

```
"nondeterminism in inference"   → [ 0.24, -0.71, 0.03, ... ]
"reproducibility problems"      → [ 0.26, -0.68, 0.05, ... ]  ← close (similar meaning)
"chocolate chip cookies"        → [-0.81,  0.44, 0.92, ... ]  ← far (unrelated)
```

This is why you can search with a question like *"why do LLMs give different answers?"* and retrieve a chunk that talks about *"floating point nondeterminism"* — the embeddings are close, even though the words don't match.

---

## Embedding Architecture

### How an Embedding Model Works

Embedding models are transformer neural networks trained specifically to produce these meaning-preserving vectors. The architecture looks like this:

```
Input text
    │
    ▼
Tokenizer
(splits text into subword tokens)
    │
    ▼
Token Embeddings
(each token → initial vector, learned during training)
    │
    ▼
Transformer Layers  ×N
(self-attention: each token "looks at" all other tokens
 and updates its meaning based on context)
    │
    ▼
Pooling Layer
(collapses all token vectors into one single vector
 that represents the whole input — the "sentence embedding")
    │
    ▼
Output Vector  [ 0.24, -0.71, 0.03, ... ]
(384 or 768 or 1536 numbers depending on the model)
```

The key innovation over older models (like Word2Vec) is the **transformer's attention mechanism** — the meaning of a word is shaped by all the words around it, not just the word itself. "Bank" in "river bank" and "Bank of America" will have different vectors because the surrounding context changes the meaning.

### What "Similarity" Means

Once you have two vectors, you measure how similar they are using **cosine similarity**:

```
cosine similarity = cos(angle between two vectors)

= 1.0  →  identical meaning
= 0.8  →  very similar
= 0.5  →  loosely related
= 0.0  →  unrelated
```

This project uses cosine similarity. When you query the database, every chunk in the database gets a similarity score against your question. The top 5 are returned.

---

## Embedding Models — Landscape

Different models make different tradeoffs between speed, cost, quality, and dimensionality.

### Open Source / Local Models (no API key, run on your machine)

| Model | Dimensions | Best For | Speed |
|---|---|---|---|
| **all-MiniLM-L6-v2** ← used here | 384 | General semantic search, fast | Very fast |
| all-mpnet-base-v2 | 768 | Higher quality general search | Medium |
| multi-qa-MiniLM-L6-cos-v1 | 384 | Question-answer retrieval specifically | Very fast |
| bge-large-en-v1.5 | 1024 | State-of-the-art open source | Slow |
| e5-large-v2 | 1024 | High accuracy retrieval | Slow |

**Why we chose `all-MiniLM-L6-v2`:**
- Fast enough to embed 92 chunks in ~1 second on a laptop
- 384 dimensions = small storage footprint
- Good enough for single-domain retrieval (academic papers in English)
- No API key, no cost, no internet required after first download

### Commercial API Models

| Model | Provider | Dimensions | Notes |
|---|---|---|---|
| text-embedding-3-small | OpenAI | 1536 | Best price/performance |
| text-embedding-3-large | OpenAI | 3072 | Highest quality OpenAI offers |
| embed-english-v3.0 | Cohere | 1024 | Strong for retrieval tasks |
| amazon-titan-embed-text-v2 | AWS Bedrock | 1024 | Available in your current AWS setup |
| voyage-3 | Voyage AI | 1024 | Specialized for long documents |

**When to upgrade from local to API models:**
- Your queries feel imprecise or miss obvious matches
- You're working with multilingual documents
- You need the highest possible retrieval accuracy for production use

---

## Vector Databases — What They Are

A vector database is a database designed specifically to store and search vectors efficiently.

Regular databases search by exact match or range (`WHERE name = 'John'`). Vector databases search by **nearest neighbor** — "find the 5 vectors most similar to this query vector."

At 92 chunks this could be done with a simple list. At 1 million chunks, you need an index structure (like HNSW — Hierarchical Navigable Small World graph) that can find the nearest neighbors without checking every single vector.

### Vector Databases — Landscape

| Database | Type | Best For |
|---|---|---|
| **ChromaDB** ← used here | File-based, embedded | Local dev, growing collections, no infrastructure |
| FAISS | In-memory library | Research, pure speed, no persistence built-in |
| Pinecone | Fully managed cloud | Production, no ops burden, serverless |
| Weaviate | Self-hosted or cloud | Multi-modal (text + images), rich filtering |
| Qdrant | Self-hosted or cloud | High performance, strong filtering, open source |
| pgvector | PostgreSQL extension | Teams already on Postgres, SQL familiarity |
| Milvus | Self-hosted | Billion-scale, enterprise |

**Why we chose ChromaDB:**
- Zero infrastructure — it's just a folder (`db/`) on disk
- Persistent by default — data survives restarts
- Supports metadata filtering (filter by paper name, page range, etc.)
- Trivial to add documents over time
- Perfect for a growing personal paper library

---

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INGESTION  (run once per paper)                            │
│                                                             │
│  PDF File                                                   │
│     │                                                       │
│     ▼                                                       │
│  pdfplumber                                                 │
│  (extracts text, preserves page numbers)                    │
│     │                                                       │
│     ▼                                                       │
│  Chunker                                                    │
│  (splits each page into 600-char overlapping windows)       │
│  "The model produces..." │ "...nondeterministic outputs"    │
│  chunk 1, page 3         │ chunk 2, page 3  (100 overlap)  │
│     │                                                       │
│     ▼                                                       │
│  all-MiniLM-L6-v2                                          │
│  (converts each chunk → 384-dim vector)                     │
│     │                                                       │
│     ▼                                                       │
│  ChromaDB  (db/)                                            │
│  stores: vector + chunk text + {paper, page, chunk_index}   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  QUERY  (run anytime)                                       │
│                                                             │
│  Your question: "What causes nondeterminism?"               │
│     │                                                       │
│     ▼                                                       │
│  all-MiniLM-L6-v2                                          │
│  (same model — embeds the question into a vector)           │
│     │                                                       │
│     ▼                                                       │
│  ChromaDB                                                   │
│  (cosine similarity search → returns top 5 chunks)          │
│     │                                                       │
│     │  [chunk, paper="Defeating...", page=2, score=0.78]    │
│     │  [chunk, paper="Defeating...", page=9, score=0.74]    │
│     │  ...                                                  │
│     ▼                                                       │
│  Claude Sonnet 4 (AWS Bedrock)                              │
│  prompt: "Answer using ONLY this context. Cite sources."    │
│  + the 5 retrieved chunks                                   │
│  + your question                                            │
│     │                                                       │
│     ▼                                                       │
│  Cited Answer                                               │
│  "Nondeterminism arises from... (Defeating Nondeterminism,  │
│   p. 2). The authors also note... (p. 9)."                  │
└─────────────────────────────────────────────────────────────┘
```

---

## The Chunking Strategy

Why not embed a whole page or a whole paper at once?

Embedding a long text produces **one vector that averages all the meanings** in that text. A 3-page methods section averaged into one vector becomes a blurry representation — good at nothing specific.

Smaller chunks produce sharper vectors that represent specific ideas, and those ideas match more precisely to specific questions.

```
Too large (whole page):
"Introduction... problem statement... related work...
 methodology... floating point ops... CUDA kernels..."
→ one blurry vector that matches nothing precisely

Just right (600 chars):
"Nondeterminism arises because floating point addition
 is not associative. When parallel threads accumulate
 values in different orders, results differ."
→ one sharp vector that matches "floating point nondeterminism" very well

Too small (one sentence):
"Results differ."
→ too little context to be meaningful
```

**Overlap (100 chars)** ensures an idea that spans a chunk boundary isn't lost — it appears in full in at least one of the two adjacent chunks.

---

## Why Similarity Scores Matter

Every retrieved chunk has a similarity score from 0 to 1:

| Score | Meaning | What to expect |
|---|---|---|
| 0.7 – 1.0 | Strong match | Answer will be accurate and direct |
| 0.5 – 0.7 | Good match | Answer will be relevant but may miss nuance |
| 0.3 – 0.5 | Weak match | Claude may say "not enough context" |
| < 0.3 | No match | Question likely doesn't match anything in the DB |

**Low scores tell you something useful:**
- The question is too abstract ("what is the main message?")
- The concept exists in the paper but is named differently
- The paper doesn't contain the answer — and the system correctly says so

---

## How to Use

### Add a new paper

```bash
source venv/bin/activate
python3 ingest.py "Papers/new_paper.pdf"
```

The database grows automatically. All future queries search across all ingested papers.

### Ask questions

```bash
source venv/bin/activate
python3 query.py
```

Type questions at the prompt. Type `exit` to quit.

### Tips for good questions

| Instead of | Ask |
|---|---|
| "What is the main message?" | "What problem does this paper solve?" |
| "Summarize the paper" | "What are the key findings?" |
| "Is this important?" | "What is the proposed solution to nondeterminism?" |
| "Tell me everything" | "What is floating point nondeterminism?" |

Specific, technical questions that could plausibly appear verbatim in the paper give the best similarity scores and the best answers.

---

## Project Structure

```
Text Embeddings/
├── Papers/                  ← drop new PDFs here
│   └── Defeating Nondeterminism in LLM Inference.pdf
├── db/                      ← ChromaDB lives here (auto-created, git-ignored)
├── ingest.py                ← add a paper to the database
├── query.py                 ← interactive Q&A across all papers
├── requirements.txt         ← Python dependencies
├── .env                     ← AWS credentials (git-ignored)
└── .gitignore
```

---

## Dependencies

| Package | Role |
|---|---|
| `pdfplumber` | Extracts text from PDFs while preserving page structure |
| `sentence-transformers` | Runs the local embedding model |
| `chromadb` | Persistent vector database |
| `anthropic[bedrock]` | Claude API via AWS Bedrock |
| `python-dotenv` | Loads AWS credentials from `.env` |
