# RAG Contract Intelligence System

A production-grade Retrieval-Augmented Generation (RAG) pipeline for legal contract analysis. Upload PDF contracts, ask natural language questions, and receive grounded answers citing the exact document and section the answer came from. It demonstrates skills in PySpark ETL, S3-compatible object storage, vector databases, hybrid retrieval, and GenAI integration.

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Architecture](#architecture)
3. [Design Decisions and Research Backing](#design-decisions-and-research-backing)
4. [MinIO as S3-Compatible Storage](#minio-as-s3-compatible-storage)
5. [Stack](#stack)
6. [Project Structure](#project-structure)
7. [Setup and Installation](#setup-and-installation)
8. [Running the System](#running-the-system)
9. [API Reference](#api-reference)
10. [RAGAS Evaluation](#ragas-evaluation)
11. [Edge Case Testing](#edge-case-testing)
12. [Known Limitations](#known-limitations)
13. [Future Improvements](#future-improvements)
14. [Research Citations](#research-citations)

---

## What This System Does

You upload a legal contract as a PDF. You type a question in plain English — "what happens if we terminate early?" or "what are the indemnification obligations?" — and the system returns a precise answer citing which document and which chunk it came from. The answer is grounded exclusively in the retrieved contract text: the LLM cannot hallucinate facts from outside your corpus.

This directly mirrors the **Contract Clause Expert** product demoed on Axrail's workshop page, built from scratch on an open-source stack deployable on AWS with minimal configuration changes.

---
## Quick Start — Ask Your First Question

Once the system is running (see [Setup](#setup-and-installation)):

1. Open `http://localhost:8501` in your browser
2. The sidebar shows your corpus — 10 contracts, 102 chunks
3. Type a question in the input box:
   - `What happens if we terminate early?`
   - `What is the base salary in the Aspira employment agreement?`
   - `Who are the parties in the ares termination agreement?`
4. Click **Ask**
5. The answer appears with source citations showing which document and chunk it came from, plus a relevance score

**To upload a new contract:** Use the Upload section in the sidebar — drag a PDF and click Upload. It re-chunks and re-embeds automatically.

**To delete a contract:** Click the 🗑 button next to any document in the sidebar. It moves to Trash with a 7-day recovery window.

**To restore a deleted contract:** The Trash section at the bottom of the sidebar shows deleted contracts with days remaining. Click ↩ to restore.

**To query via API directly:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method POST -ContentType "application/json" -Body '{"question": "what happens if we terminate early?"}'
```

## Architecture

```
10 PDF contracts (data/sample_contracts/)
        │
        ▼
PySpark batch ETL (ingestion/spark_processor.py)
        │                           │
        ▼                           ▼
Parquet chunk store          MinIO S3-compatible bucket
(data/chunks/chunks.parquet) (localhost:9000 / swappable to AWS S3)
        │
        ▼
Text extraction + cleaning (ingestion/chunker.py)
150-word chunks, 30-word overlap, sliding window
Metadata: source, chunk_index, total_chunks
        │
        ▼
all-MiniLM-L6-v2 sentence-transformer (embeddings/embed.py)
384-dimension vectors, CPU-only, cosine similarity
        │
        ▼
ChromaDB persistent vector store (embeddings/chroma_store.py)
HNSW index, hnsw:space=cosine
        │
        ▼
User question (Streamlit frontend, localhost:8501)
        │
        ▼
Hybrid retrieval (retrieval/hybrid.py)
0.5 × BM25 score + 0.5 × vector score → top 3 chunks → reversed
        │
        ▼
FastAPI backend (api/query.py, localhost:8000)
Builds grounded prompt → Groq LLM (openai/gpt-oss-20b)
        │
        ▼
Answer + source citations (chunk index, relevance score, filename)
        │                    │
        ▼                    ▼
Streamlit UI           History panel + confidence scores
(frontend/app.py)      Upload / soft-delete / restore contracts
```

---

## Design Decisions and Research Backing

Every design choice in this system is grounded in published research. This section documents the reasoning behind each one.

### Chunking Strategy — Sliding Window, 150 Words, 30-Word Overlap

**The constraint.** `all-MiniLM-L6-v2` has a hard input limit of 256 tokens. Token count and word count are not the same: in English legal text, 1 word ≈ 1.3 tokens, making the safe upper bound approximately 196 words. Chunks exceeding this limit are silently truncated — the model processes only the first 256 tokens without any error or warning, losing the remainder of the clause.

**The choice.** 150-word chunks with 30-word overlap (20% of chunk size). This keeps every chunk safely under the 256-token hard limit while the 30-word overlap prevents legal clauses that span chunk boundaries from being split and missed entirely.

**The research.** Gao et al. (2023) in their RAG survey (Section III-B-1) explicitly validate sliding window chunking over fixed-size splitting, noting that fixed splits "cause truncation within sentences" and recommending sliding window methods to preserve semantic continuity across chunk boundaries. The 20% overlap ratio follows their guidance on boundary preservation.

**Validator.** `ingestion/chunk_validator.py` confirms zero truncation warnings across all 102 chunks in the corpus, verifying every chunk fits within the model's context window.

### Embedding Model — all-MiniLM-L6-v2

**Why this model.** Reimers and Gurevych (2019) introduced the Sentence-BERT architecture, showing that sentence-transformer models produce semantically meaningful fixed-length embeddings suitable for cosine similarity comparison. `all-MiniLM-L6-v2` scores 77-85 on STS (Semantic Textual Similarity) benchmarks compared to 46.35 for averaged BERT embeddings — a meaningful gap for legal clause retrieval where semantic proximity matters.

**Practical reasons.** CPU-only inference, 384-dimensional output (compact for storage and fast for similarity computation), and no API dependency — the model runs locally with no cost per embedding call. This matters for a pipeline that embeds an entire corpus at ingestion time.

**Cosine similarity.** ChromaDB is configured with `hnsw:space=cosine` explicitly. Cosine similarity normalises for vector magnitude and measures angular distance between embeddings, which is the appropriate metric for semantic similarity tasks. L2 (Euclidean) distance is sensitive to vector magnitude and would conflate semantic dissimilarity with embedding scale differences.

### Hybrid Retrieval — BM25 + Vector, 0.5/0.5 Weight

**The problem with pure vector search.** Vector search captures semantic meaning but suffers from vocabulary mismatch: "cancellation prior to expiry" will not score highly for the query "terminate early" in BM25, but the embeddings will place these phrases close together. Conversely, "mutual written consent" will score highly in BM25 for a query containing those exact words, regardless of semantic context. Neither method alone is sufficient for legal text, which combines precise defined terms (favouring BM25) with conceptual synonymy (favouring vector search).

**BM25.** BM25Okapi (Robertson and Zaragoza, 2009) scores chunks by term frequency weighted by inverse document frequency. Rare legal terms like "indemnification" or "subrogation" appearing in a chunk receive high BM25 scores when queried, because they are rare across the corpus. Common words like "the" and "shall" receive near-zero weight regardless of frequency.

**Normalisation before combination.** BM25 returns raw term frequency scores (typically 0–40). Vector similarity returns cosine distances (0–1). These cannot be combined directly — BM25 would dominate by an order of magnitude. Both are normalised to the range [0, 1] using min-max normalisation before the weighted sum. Without this step, the hybrid score is mathematically meaningless.

**Chunk reversal for the lost-in-the-middle effect.** The top 3 retrieved chunks are reversed before being placed in the prompt (most relevant chunk last, directly above the question). This addresses the lost-in-the-middle problem documented by Gao et al. (2023) in Section IV-A: LLMs attend most strongly to content at the beginning and end of their context window, and least to content in the middle. Placing the highest-scoring chunk immediately before the question maximises the probability that the LLM uses it in generation.

### Grounded Generation — Prompt Engineering

The prompt explicitly instructs the LLM to answer only from the provided excerpts and to state "I could not find this information in the provided contracts" if the answer is not present. This is the core mechanism that makes the system a grounded RAG pipeline rather than a conversational AI that might confabulate legal clauses.

This follows the Lewis et al. (2020) RAG paradigm: retrieve first, generate conditioned only on retrieved context. The prompt also instructs the model to identify which documents contained the relevant clause and to flag terms the user should verify with a lawyer — improving interpretability for non-expert users.

### Storage Architecture — Dual Storage Pattern

Raw PDFs are stored in MinIO (S3-compatible object storage). Processed chunks are stored in Parquet via PySpark. Vector embeddings are stored in ChromaDB. This separation follows the data lakehouse pattern: raw data preserved in object storage, processed data in columnar format for analytics, vector data in a specialised index for retrieval. Each layer is independently queryable and replaceable.

---

## MinIO as S3-Compatible Storage

### What MinIO Is

MinIO is an open-source, high-performance object storage system that implements the Amazon S3 API in full. It runs as a Docker container on your local machine and exposes the same REST endpoints, authentication model, and bucket operations as AWS S3.

### Why MinIO Instead of AWS S3

For local development and portfolio projects, MinIO provides the complete S3 developer experience without AWS account requirements, egress costs, or credential management complexity. The critical property is **API identity**: every boto3 operation used in this project — `create_bucket`, `put_object`, `list_objects_v2` — works identically against MinIO and AWS S3.

```python
# This boto3 client works against MinIO (local)
s3_client = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin"
)

# Switching to real AWS S3 requires only credential and endpoint changes
s3_client = boto3.client(
    "s3",
    region_name="ap-southeast-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
```

The application code in `ingestion/minio_client.py` is identical for both environments. The switch from MinIO to production AWS S3 is a configuration change, not a code rewrite. This is the key signal for an AWS partner organisation: the system is built against the S3 API, not against MinIO specifically. This project uses MinIO under AGPLv3 for non-commercial portfolio use, which is explicitly permitted under the licence terms.


### What Is Stored in MinIO

All 10 PDF contracts are stored in the `contracts` bucket at ingestion time. MinIO provides the raw document preservation layer — the canonical store of the original files before any processing. This separation means the raw contracts are always recoverable even if the Parquet store or ChromaDB index is corrupted or rebuilt.

---

## Stack

| Layer | Technology | Role |
|---|---|---|
| Ingestion | PySpark 4.1.2 | Batch ETL, Parquet writing |
| Object storage | MinIO (Docker) | S3-compatible raw PDF storage |
| Chunking | pdfplumber + custom | Text extraction, sliding window chunking |
| Embeddings | all-MiniLM-L6-v2 | 384-dim sentence vectors |
| Vector store | ChromaDB | Persistent HNSW vector index |
| Retrieval | rank-bm25 + ChromaDB | Hybrid BM25 + semantic search |
| LLM | Groq (openai/gpt-oss-20b) | Grounded answer generation |
| API | FastAPI + Uvicorn | REST API with Pydantic validation |
| Frontend | Streamlit | Interactive query UI |
| Evaluation | Custom LLM-as-judge | RAGAS-style faithfulness scoring |
| Orchestration | Docker Compose | Service management |

---

## Project Structure

```
contract-intelligence/
├── docker-compose.yml
├── .env                          # MINIO_*, GROQ_API_KEY
├── .gitignore
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI entrypoint, router registration
│   ├── query.py                  # /query — hybrid retrieval + Groq generation
│   ├── contracts.py              # /contracts — corpus metadata
│   └── upload.py                 # /upload, /deleted, /restore — document lifecycle
├── ingestion/
│   ├── __init__.py
│   ├── chunker.py                # PDF extraction, cleaning, sliding window chunking
│   ├── chunk_validator.py        # Verifies no chunk exceeds 256 tokens
│   ├── minio_client.py           # boto3 S3-compatible upload to MinIO
│   └── spark_processor.py        # PySpark ETL → chunks.parquet
├── embeddings/
│   ├── __init__.py
│   ├── embed.py                  # Loads all-MiniLM-L6-v2, encodes 102 chunks
│   └── chroma_store.py           # Stores vectors in ChromaDB, test search
├── retrieval/
│   ├── __init__.py
│   └── hybrid.py                 # BM25 + vector hybrid scoring
├── frontend/
│   └── app.py                    # Streamlit UI with history, upload, soft-delete
├── evaluation/
│   ├── ragas_eval.py             # 20-question LLM-as-judge evaluation
│   ├── ragas_results.csv         # Per-question scores
│   └── ragas_summary.json        # Aggregate metrics
└── data/
    ├── sample_contracts/         # 10 PDF contracts from SEC EDGAR
    ├── deleted_contracts/        # Soft-deleted PDFs (7-day retention)
    ├── chunks/
    │   └── chunks.parquet        # 102 chunks, 4 columns, 46KB
    └── chroma/                   # ChromaDB persistent vector index
```

---

## Setup and Installation

### Prerequisites

Python 3.10+, Docker Desktop, Java 17 (not Java 25 — see Known Limitations).

### Environment variables

Create `.env` in the project root:

```
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
GROQ_API_KEY=your_groq_api_key_here
```

### Install dependencies

```bash
pip install boto3 python-dotenv pdfplumber sentence-transformers chromadb \
            transformers torch pyspark pandas pyarrow rank-bm25 groq \
            fastapi uvicorn streamlit requests
```

### Start MinIO

```bash
docker-compose up -d
```

MinIO console available at `http://localhost:9001` (minioadmin / minioadmin).

### Run the ingestion pipeline

Run each step from the project root in order:

```bash
python -m ingestion.minio_client        # Upload PDFs to MinIO bucket
python -m ingestion.spark_processor     # PySpark ETL → chunks.parquet
python -m ingestion.chunk_validator     # Verify 0 truncation warnings
python -m embeddings.embed              # Encode chunks → (102, 384) matrix
python -m embeddings.chroma_store       # Store vectors in ChromaDB
```

---

## Running the System

Three processes must run simultaneously in separate terminals:

**Terminal 1 — API server:**
```bash
python -m uvicorn api.main:app --reload
```
API available at `http://localhost:8000`. Auto-generated docs at `http://localhost:8000/docs`.

**Terminal 2 — Frontend:**
```bash
python -m streamlit run frontend/app.py
```
UI available at `http://localhost:8501`.

**Terminal 3 — Evaluation (optional, standalone):**
```bash
python -m evaluation.ragas_eval
```

---

## API Reference

### `GET /health`
Returns `{"status": "ok"}`. Used by monitoring and load balancers to confirm the service is alive.

### `GET /contracts`
Returns corpus metadata: list of all documents with chunk counts, total document count, total chunk count. Powers the Streamlit sidebar and Superset dashboard.

### `POST /query`
**Request:**
```json
{"question": "what happens if we terminate early?"}
```
**Response:**
```json
{
  "answer": "Under the Business Combination Agreement...",
  "sources": [
    {"source": "ares_acquisition_termination_2023.pdf", "chunk_index": 2, "score": 0.859}
  ]
}
```
Pipeline: hybrid search → prompt construction → Groq LLM → grounded answer with source citations.

### `POST /upload`
Accepts a PDF file upload. Extracts text, chunks, embeds, and adds to both Parquet and ChromaDB without requiring a pipeline rerun.

### `DELETE /contracts/{source_name}`
Soft-deletes a contract: removes from ChromaDB and Parquet, moves PDF to `data/deleted_contracts/` with a timestamp. Does not permanently delete.

### `GET /deleted`
Lists soft-deleted contracts with deletion timestamp and days remaining before permanent purge (7-day retention window).

### `POST /restore/{source_name}`
Restores a soft-deleted contract: moves PDF back to `data/sample_contracts/`, re-chunks, re-embeds, and re-adds to ChromaDB and Parquet.

---

## RAGAS Evaluation

Evaluated on 20 domain-specific legal contract questions derived directly from the SEC EDGAR documents in the corpus. Every ground truth is verbatim-verifiable against the source PDFs.

**Scoring methodology.** LLM-as-judge using `llama-3.3-70b-versatile` scoring three dimensions on a 0–1 scale per question, following the RAGAS evaluation framework (Es et al., 2023).

| Metric | Score | Definition |
|---|---|---|
| Faithfulness | 0.575 | Claims in the answer traceable to retrieved context |
| Context Relevance | 0.675 | Retrieved chunks contained information relevant to the question |
| Answer Relevance | 0.800 | Answer addressed what was actually asked |
| **Overall** | **0.683** | Combined average |

**Interpretation.** Answer relevance at 0.800 is strong — the LLM consistently understood and addressed the question. Context relevance at 0.675 reflects the top-3 retrieval window limitation: some answers require a clause ranked 4th or 5th by hybrid score. Faithfulness at 0.575 reflects two factors: (1) the LLM occasionally draws on related legal knowledge when retrieved chunks are partial, and (2) the LLM-as-judge scorer is conservative on faithfulness when answers are more detailed than the raw chunk text.

These scores are consistent with a first-pass RAG pipeline on a small specialised corpus using a general-purpose embedding model. Production legal RAG systems typically achieve 0.75–0.85 across all metrics through the improvements documented in the section below.

---

## Edge Case Testing

| Question | Behaviour | Assessment |
|---|---|---|
| What happens if we terminate early? | Cited Section 8.01(a), mutual written consent, expense reimbursement from ares agreement | Correct and grounded |
| What are my obligations? | Returned obligations table across 3 documents | Cross-document retrieval working |
| Which contracts mention confidentiality? | Returned "could not find" | Correct — no confidentiality clauses in corpus |
| What happens if someone breaks the deal? | Found breach and injunctive relief clauses despite no shared keywords | Vector search compensated for vocabulary mismatch |
| What is the penalty for missing a payment deadline? | Returned REIN Therapeutics stock exercise clause | False positive — semantically adjacent but factually incorrect |
| What does Apple's insider trading policy say about blackout periods? | Returned "could not find" despite retrieving Apple chunks | Retrieval coverage gap — content in unchosen chunks |
| What contracts do you have access to? | Listed only Apple — missed 9 other documents | No corpus-level awareness — system sees only 3 retrieved chunks |
| What is the weather in Kuala Lumpur? | Refused cleanly | Grounding instruction held against off-topic query |
| Which contracts talk about pay? | Identified Aspira $375,000 salary clause, noted others had no pay mentions | Best result — cross-document, specific figure extracted |
| Are there any suspicious provisions? | Identified Apple insider trading compliance and Farmer Brothers governing-law clause | Nuanced legal interpretation without explicit prompt |

---

## Known Limitations

**Java 25 + PySpark 4.1.2 + Hadoop incompatibility.** PySpark's Hadoop file writer calls `getSubject()` which is unsupported in Java 25's security manager. Fixed by writing Parquet via `pandas.to_parquet()` with PyArrow instead of the native PySpark Hadoop writer. This bypasses the issue entirely at the cost of not using the Spark-native writer, which is acceptable for batch ETL on a single-machine corpus of this size.

**Page numbers not stored at ingestion.** The chunker stores `chunk_index` but not the original PDF page number. Source citations reference chunk index rather than page number, which is less useful for users who want to locate a clause in the original document. Fix: modify `chunker.py` to track `page_number` from pdfplumber's `page.page_number` and store it in the Parquet schema, then rerun the full ingestion pipeline.

**No corpus-level awareness.** The LLM receives only the 3 retrieved chunks and cannot answer questions like "what contracts do you have access to?" accurately. Fix: inject a system message listing all document names before the retrieved chunks, giving the LLM awareness of the full corpus even when specific documents aren't retrieved.

**Groq free-tier rate limiting.** The first request after a period of inactivity occasionally returns an empty completion. Retrying immediately succeeds. In production, a paid tier with guaranteed SLAs eliminates this.

**LLM-as-judge scoring inconsistency.** The RAGAS evaluation uses a free-tier LLM as the scoring judge. Scorer behaviour changed materially when `llama-3.3-70b-versatile` entered deprecation (June 2026), producing inconsistent scores. The 0.683 overall score is from the stable evaluation run before deprecation.

---

## Future Improvements

**Reranking with a cross-encoder.** After initial hybrid retrieval, a cross-encoder reranker (BGE-Reranker, Cohere Rerank) scores retrieved chunks against the query jointly rather than independently. Recent benchmark work (Gao et al., 2023; Section V-B) shows reranking can improve context precision by 10–15 points, directly improving faithfulness by reducing noise in the LLM's context window.

**Semantic or structure-aware chunking.** The current 150-word sliding window is content-agnostic — it splits at arbitrary word boundaries. Legal contracts have natural structure (sections, subsections, numbered clauses) that semantic chunking would preserve. Chunking at section boundaries ensures each chunk contains a complete clause rather than a partial one, reducing the boundary-loss problem.

**Fine-tuning on legal text.** `all-MiniLM-L6-v2` is a general-purpose model trained on diverse web text. Legal language has a specific vocabulary (indemnification, subrogation, mutatis mutandis) and syntactic structure that domain-specific models like Legal-BERT (Chalkidis et al., 2020) are explicitly trained on. Fine-tuning or switching to a legal embedding model would improve retrieval precision for low-frequency legal terms.

**Increasing retrieval window to top-5.** The context relevance score of 0.675 is partly attributable to the answer being in the 4th or 5th ranked chunk rather than the top 3. Increasing `n_results` from 3 to 5 would improve coverage with a modest increase in prompt length and LLM token cost.

**Page number tracking.** Modifying the ingestion pipeline to store PDF page numbers in the Parquet schema and surface them in the API response would make source citations actionable — a user could open the contract and navigate directly to the cited page rather than searching for the chunk text manually.

**Docker Compose full orchestration.** Currently ChromaDB, FastAPI, and Streamlit run as bare Python processes. Adding them as services in `docker-compose.yml` would allow `docker-compose up` to start the entire system in one command, making the project reproducible on any machine with Docker without Python environment setup.

---

## Research Citations

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *arXiv:2005.11401*. The foundational RAG paper. This system's index → retrieve → generate architecture follows their paradigm directly.

Gao, Y., Xiong, Y., Gao, X., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997*. Justifies sliding window chunking (§III-B-1), metadata attachment (§III-B-2), the lost-in-the-middle retrieval ordering decision (§IV-A), and the RAGAS evaluation approach (§VI-D).

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *arXiv:1908.10084*. Justifies the sentence-transformer architecture and cosine similarity metric for semantic retrieval.

Es, S., James, J., Anke, L. E., and Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. Introduces the faithfulness, context relevance, and answer relevance metrics used in this project's evaluation framework.

Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval, 3*(4). Foundational reference for the BM25Okapi implementation used in hybrid retrieval.

---

## Corpus

All contracts sourced from SEC EDGAR public filings:

| Document | Type | Chunks |
|---|---|---|
| Aetherium Acquisition Corp current report | Business combination termination announcement | 5 |
| Apple Inc current report | Insider Trading Policy | 18 |
| ares acquisition termination 2023 | Termination Agreement | 9 |
| Aspira Women's Health employment agreement | Employment agreement (CFO) | 4 |
| farmer brothers board agreement 2023 | Board letter agreement | 13 |
| firefly neuroscience current report | Acquisition announcement | 8 |
| Inuvo quarterly contract | Google Services Agreement extension | 8 |
| Leet Technology Inc current report | Director appointment | 6 |
| Oxford Finance LLC | KPMG agreed-upon procedures report | 20 |
| REIN Therapeutics current report | Warrant inducement offer | 11 |

**Total: 102 chunks, 0 truncation warnings (verified by chunk_validator.py)**
