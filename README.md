<div align="center">

<img src="frontend/public/logo.png" alt="AI Research Copilot logo" width="96" />

# AI Research Copilot

### A private, citation-grounded RAG workspace for research papers

Upload a PDF, turn it into a searchable knowledge base, and ask questions answered from the paper—not from guesswork.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Dense%20Retrieval-7C3AED)

</div>

---

## Why this project?

Reading research papers is slow; finding the relevant evidence inside them is often slower. AI Research Copilot is a full-stack Retrieval-Augmented Generation (RAG) application that lets each authenticated user build a private research library and ask grounded questions about their uploaded papers.

The project explores the complete lifecycle of a modern AI application: authentication, document ingestion, text extraction, chunking, semantic search, LLM generation, citations, testing, and frontend integration.

## What it does

- Authenticates users with Supabase Auth.
- Uploads and validates PDF research papers.
- Prevents exact duplicate uploads per user with SHA-256 content hashes.
- Extracts paper metadata, including title, authors, abstract, and year where available.
- Parses PDF text page by page and creates overlapping chunks.
- Builds a private vector index for each user's papers.
- Retrieves semantically relevant chunks for a question.
- Generates grounded answers through Groq with page-level source citations.
- Enforces ownership checks across paper, chunk, vector, and chat operations.

## System architecture

```text
┌───────────────────────────┐
│ React + TypeScript client │
│  Supabase session token   │
└─────────────┬─────────────┘
              │ HTTPS / REST
              ▼
┌────────────────────────────────────────────────┐
│                  FastAPI backend                │
│                                                │
│  Routes → Services → Storage adapters           │
│                                                │
│  Auth dependency validates Supabase JWT/user    │
└───────┬────────────────┬─────────────────┬─────┘
        │                │                 │
        ▼                ▼                 ▼
  PDF file storage    SQLite           ChromaDB
  original files      metadata/chunks  embeddings/vectors
        │                                  │
        └───────────┬──────────────────────┘
                    ▼
        Sentence Transformer retrieval
                    │
                    ▼
          Groq LLM answer generation
                    │
                    ▼
          Answer with page citations
```

## RAG pipeline

```text
PDF upload
   │
   ├── Validate PDF + calculate SHA-256 hash
   ├── Save original file and user-owned paper record
   ├── Extract text and metadata
   ├── Clean text and create overlapping chunks
   ├── Create embeddings for chunks
   └── Store vectors in ChromaDB with paper_id + user_id metadata

User question
   │
   ├── Validate Supabase access token
   ├── Embed the question
   ├── Retrieve top-k chunks filtered by user_id
   ├── Add selected-paper metadata when relevant
   ├── Ask Groq to answer only from retrieved context
   └── Return answer, confidence signal, and cited pages
```

### RAG approach

This is a **single-stage dense RAG** pipeline:

- **Dense retrieval:** sentence embeddings find semantically related text, not just exact keyword matches.
- **Top-k context selection:** only the most relevant chunks are sent to the LLM.
- **Citation-grounded generation:** answers are paired with the original paper and page numbers.
- **Private retrieval:** every vector search is filtered to the authenticated user's library.

It intentionally does not yet include hybrid BM25 search, reranking, query rewriting, background workers, or conversational memory. Those are planned scalability and quality improvements, not hidden implementation claims.

## Key engineering decisions

| Decision | Why | Trade-off |
|---|---|---|
| FastAPI + Pydantic | Typed request/response models, automatic OpenAPI docs, and testable REST endpoints | Python backend is separate from the frontend build |
| Layered architecture | Routes handle HTTP, services hold business logic, storage adapters isolate persistence | More files than a single-script prototype |
| Supabase Auth verification | Avoids storing passwords and gives the backend a reliable user identity | Each authenticated request verifies a token remotely |
| SQLite for metadata | Portable, simple, and ideal for local MVP development | Not suited to multi-instance production deployment |
| ChromaDB for vectors | Makes semantic retrieval easy to understand and implement from first principles | Local persistence requires durable hosting in production |
| Overlapping text chunks | Preserves context at chunk boundaries | More chunks increase indexing time and storage |
| SHA-256 duplicate detection | Reliably rejects byte-for-byte duplicate PDFs for the same user | Does not recognise differently formatted copies of the same paper |
| Groq for generation | Fast hosted LLM inference without running a local LLM | Requires an external API and API-key management |

## Security and privacy design

The application treats paper ownership as a backend responsibility, not a frontend assumption.

```text
Supabase session → backend token validation → authenticated user ID
                                            ↓
                     owner_id passed to every paper, retrieval,
                     indexing, vector-store, and delete operation
```

This prevents a user from listing, reading, querying, indexing, or deleting another user's papers—even if they manually alter a frontend request.

## Technology stack

| Area | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite, Tailwind CSS, TanStack Query, React Markdown |
| Authentication | Supabase Auth |
| Backend | FastAPI, Uvicorn, Pydantic Settings |
| PDF processing | PyMuPDF |
| Retrieval | Sentence Transformers, ChromaDB |
| Generation | Groq API |
| Data | SQLite, local PDF storage, Chroma persistent store |
| Quality | Pytest, type hints, service-level tests, API tests |
| Local deployment | Docker and Docker Compose |

## Project structure

```text
ai-research-copilot/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes and dependencies
│   │   ├── core/         # configuration, auth, logging, exceptions
│   │   ├── models/       # domain and API schemas
│   │   ├── services/     # ingestion, chunking, retrieval, chat
│   │   └── storage/      # SQLite, ChromaDB, PDF file storage
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── api/          # authenticated API client
│       ├── features/     # auth, library, and chat UI
│       └── lib/          # Supabase client
├── docs/
├── docker-compose.yml
└── README.md
```

## Run locally

### Prerequisites

- Python 3.13+
- Node.js 20+
- A Supabase project
- A Groq API key

### 1. Configure the backend

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install packages and create the environment file:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set these values in `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
SUPABASE_URL=your_supabase_project_url
SUPABASE_PUBLISHABLE_KEY=your_supabase_publishable_key
CORS_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]
```

Start the API:

```bash
uvicorn app.main:app --reload
```

API documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Configure the frontend

```bash
cd ../frontend
npm install
```

Create `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_PUBLISHABLE_KEY=your_supabase_publishable_key
```

Start the frontend:

```bash
npm run dev
```

Open `http://localhost:5173`.

### Run with Docker

```bash
docker compose up --build
```

Docker Compose creates a named `research_data` volume so PDFs, SQLite data, and vector data survive normal local container restarts.

## Testing

Run the backend test suite:

```bash
cd backend
python -m pytest
```

The tests cover core models, PDF processing, chunking, metadata extraction, storage behavior, indexing, API validation, ownership boundaries, and duplicate-upload handling.

## Production roadmap

Version 1 prioritises a transparent, end-to-end RAG workflow and local developer experience. A production-scale Version 2 would include:

- Object storage for original PDFs.
- Postgres for relational data and `pgvector` or a managed vector database for embeddings.
- A background queue for long-running PDF processing and indexing.
- Hybrid retrieval and reranking for stronger relevance.
- DOI/title-based duplicate matching in addition to file hashing.
- Observability, rate limiting, audit logs, and automated deployment checks.

## What I learned

- Building RAG requires more than calling an LLM: retrieval quality, chunking, metadata, citations, and data ownership all matter.
- AI application architecture has the same foundations as any backend system: API design, authentication, persistence, testing, and clear separation of responsibilities.
- A good MVP makes deliberate trade-offs, documents them, and creates a clear path toward a more scalable architecture.

## License

No license has been selected yet. Add an open-source license before accepting external contributions or reuse.
