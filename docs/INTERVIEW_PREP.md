# Interview Preparation Guide: AI Research Copilot (+ OCR Document-AI)

> A complete reference for defending this project in a technical interview where the interviewer
> talks to you like a colleague: "why this?", "what if I want that?", "what breaks at scale?".
>
> **How to use this guide**
> 1. Read **Part 0** first. It lists what is *actually in the code* and what is only roadmap.
>    Interviewers often open your GitHub repo during the call, so never claim something the code doesn't do.
> 2. **Part 1** is a file-by-file, number-by-number tour of Project 1.
> 3. **Part 2** covers the concepts the interviewer will push on: RAG vs fine-tuning, LangChain, dense/sparse/hybrid,
>    rerankers, similarity search, vector DBs, hallucination, evaluation, multimodal input, scaling.
> 4. **Part 3** has 30+ "I want X, what changes in your architecture?" scenarios with model answers.
> 5. **Part 4** covers Project 2 (OCR-based Intelligent Document-AI). Its code is **not in this repository**,
>    so that part is a reference architecture you should adjust to match what you actually built.
> 6. **Part 5** is rapid-fire Q&A and interview strategy.

---

## Part 0: Ground truth (read this before anything else)

### 0.1 The pitches

**30-second version**
> "AI Research Copilot is a full-stack, multi-tenant RAG application. A user signs in with Supabase, uploads research
> PDFs, and the FastAPI backend parses them with PyMuPDF, extracts metadata, splits the text into page-aware overlapping
> chunks, embeds them with a Sentence-Transformers model (all-mpnet-base-v2) and stores the vectors in ChromaDB, tagged with the
> user's ID. When the user asks a question, I embed it, run a cosine-similarity search filtered to that user (and
> optionally one paper), and send the top-k chunks to an LLM on Groq with a strict 'answer only from context' prompt. The answer comes back
> with page-level citations. The frontend is React + TypeScript with TanStack Query."

**2-minute version.** Add these points:
- **Layered architecture:** routes → services → storage adapters, wired by FastAPI dependency injection. Every service is unit-testable with fakes.
  There are 63 pytest tests across models, services, storage and API.
- **Security is enforced by the backend:** `owner_id` is passed into *every* SQL query, *every* Chroma `where` filter, and every delete.
  A user can't read another user's paper even if they edit the frontend request.
- **Idempotency and data integrity:** SHA-256 content hashing blocks duplicate uploads per user (unique index on `(user_id, content_hash)`).
  Re-indexing deletes old vectors before upserting, and chunks have `UNIQUE(paper_id, chunk_index)` plus `ON DELETE CASCADE`.
- **Grounding:** temperature 0.2, a system prompt that forbids outside knowledge, sources labelled `[Source n | page=p]`,
  and citations built by the *application* from retrieval results, not by the LLM. The model can't invent a page number.
- **Honest trade-offs:** V1 is single-stage dense retrieval with no hybrid search, no reranker and no background queue. I documented a
  V2 roadmap: Postgres/pgvector or Qdrant, object storage, workers, hybrid + reranking, evaluation.

### 0.2 What is implemented vs not (be precise)

| Topic | In the code? | Where / note |
|---|---|---|
| PDF parsing (PyMuPDF) | ✅ | `backend/app/services/pdf_parser.py` |
| Metadata extraction (title/authors/abstract/year/DOI/keywords) | ✅ heuristic | `services/metadata_extractor.py`: font-size and layout heuristics plus regex |
| Text cleaning (NFKC, whitespace) | ✅ | `services/text_cleaner.py` |
| Chunking: page-aware, word-based, overlapping | ✅ | `services/text_chunker.py`: **200 words, 40 overlap** |
| Embeddings: `all-mpnet-base-v2`, normalized | ✅ | `services/embedding_service.py`, `api/dependencies.py` |
| Vector DB: ChromaDB, HNSW, cosine | ✅ | `storage/chroma.py` |
| Multi-tenant filtering (`user_id`, `paper_id`) | ✅ | Chroma `where` filters plus SQL `WHERE user_id = ?` |
| LLM generation via Groq (`openai/gpt-oss-20b`) | ✅ | `services/llm_service.py` |
| Citations (paper title + page) | ✅ | `services/chat_service.py::_build_citations` |
| "Confidence" score | ✅ but naive | `max(cosine similarity of retrieved chunks)`. **Not calibrated.** |
| Supabase auth | ✅ | `core/auth.py`: remote token validation via `/auth/v1/user` |
| Duplicate detection (SHA-256) | ✅ | `services/paper_service.py` |
| Docker / docker-compose / Vercel | ✅ | `backend/Dockerfile`, `docker-compose.yml`, `frontend/vercel.json` |
| **LangChain / LlamaIndex** | ❌ **Not used** | Built from first principles on purpose (see §2.2) |
| **Hybrid retrieval (BM25 + dense)** | ❌ roadmap | Note: `docs/architecture.md` describes "Dense + BM25 → Fusion → Reranking" as the *target* design |
| **Cross-encoder reranker** | ❌ roadmap | |
| **RAGAS / automated eval** | ❌ roadmap | Have a plan ready (§2.10) |
| Conversation memory | ❌ | `Query.conversation_id` exists in the schema but is unused |
| Streaming responses | ❌ | |
| Background jobs | ❌ | upload → process → index are 3 synchronous HTTP calls orchestrated by the frontend |
| OCR for scanned PDFs / images | ❌ | A scanned PDF yields empty pages, so 0 chunks (see §2.11) |

> ⚠️ **If your resume says "hybrid retrieval", "reranker", "LangChain" or "RAGAS" for this project**, do one of two things:
> (a) implement them before the interview (§3.1 and §2.10 give exact designs; each is a small change in this codebase), or
> (b) say clearly: *"V1 ships dense-only; hybrid + reranking was my designed V2 and I evaluated it in my research work."*
> Interviewers forgive a smaller scope. They do not forgive claims that collapse after one follow-up question.

> ⚠️ **Docs drift you might be asked about:** `.env.example` and `core/config.py` define `CHUNK_SIZE=800` / `CHUNK_OVERLAP=200`,
> but `TextChunker()` is constructed with its defaults (200 words / 40 overlap) in `api/dependencies.py`, so those settings are
> currently **unused**. If asked, say: *"Good catch. The config values are legacy; the effective values are 200/40 words. The fix
> is to inject `settings.chunk_size` into `TextChunker` in the dependency provider."*

---

## Part 1: Project 1 deep dive: AI Research Copilot

### 1.1 Tech stack, with the reason for each choice

| Layer | Technology | Version (pinned) | Why this, and not the alternative |
|---|---|---|---|
| Language | Python | 3.13 | ML ecosystem (sentence-transformers, torch); type hints and dataclasses |
| Web framework | FastAPI + Uvicorn | fastapi 0.139, uvicorn 0.50 | Async, Pydantic validation, auto OpenAPI docs at `/docs`, `Depends()` DI. Flask lacks built-in validation and DI; Django is too heavy for an API-only service |
| Validation / settings | Pydantic v2, pydantic-settings | 2.13 | Typed models are the API contract; settings come from `.env` |
| PDF parsing | PyMuPDF (`pymupdf`) | 1.28 | Fast (MuPDF is written in C) and gives layout info (`get_text("dict")`: spans, font sizes, bboxes), which the metadata extractor needs. pdfplumber is slower; pypdf has weaker layout support |
| Embeddings | sentence-transformers, `all-mpnet-base-v2` | ST 6.0, torch 2.13 | Strong general-purpose bi-encoder, **768-dim**, max **384 tokens**, runs on CPU, free, private (no data sent out) |
| Vector DB | ChromaDB (PersistentClient) | 1.5 | Embedded (no separate server), persistent on disk, metadata filtering, HNSW. Ideal for an MVP |
| Relational store | SQLite (`sqlite3` stdlib) | n/a | Zero ops, file-based, ACID, foreign keys. Single-node limitation is accepted for V1 |
| LLM | Groq API, model `openai/gpt-oss-20b` | groq 1.6 | Very low latency (Groq LPU hardware), OpenAI-compatible chat API, cheap, open-weight model (can be self-hosted later) |
| Auth | Supabase Auth | supabase-js 2.x | Managed auth (email, OAuth) so no password storage on my side; backend gets a verified user UUID |
| Frontend | React 19, TypeScript, Vite 8, Tailwind 4 | | Type safety shared with API types (`src/types/api.ts`) |
| Server state | TanStack Query 5 | | Caching, `invalidateQueries` after mutations, loading and error states |
| Markdown rendering | react-markdown + remark-gfm | | The LLM is told to answer in Markdown (headings, bullets, tables) |
| Testing | pytest 9 | | 63 tests: models, services, storage, API |
| Quality | ruff, black, isort, mypy (pydantic plugin) | | `pyproject.toml` |
| Deploy | Docker (`python:3.13-slim`), docker-compose named volume, Vercel for the SPA | | Persistent volume for SQLite, Chroma and PDFs |

### 1.2 High-level architecture

```text
 Browser (React + TS, Vite)                     Supabase Auth
 ┌─────────────────────────┐   sign-in     ┌───────────────────┐
 │ AuthPage / PaperLibrary │──────────────►│  issues JWT access │
 │ ChatPanel (TanStack Q.) │◄──────────────│  token (session)   │
 └───────────┬─────────────┘               └─────────▲─────────┘
             │ Authorization: Bearer <JWT>            │ GET /auth/v1/user
             ▼                                        │ (token validation)
 ┌───────────────────────────── FastAPI (/api/v1) ────┴──────────────────┐
 │ Routes:  /papers  /papers/{id}/process  /papers/{id}/index            │
 │          /search  /chat  /health  /auth                               │
 │ Depends(get_current_user) → AuthenticatedUser(id, email)              │
 │                                                                       │
 │ Services: PaperService, DocumentProcessor(PDFParser, TextCleaner,     │
 │           TextChunker, MetadataExtractor), IndexingService,           │
 │           EmbeddingService, RetrievalService, LLMService, ChatService │
 │                                                                       │
 │ Storage adapters: FileStorage (PDF bytes)                             │
 │                   SQLitePaperRepository (papers, chunks)              │
 │                   ChromaVectorStore (vectors + metadata)              │
 └──────┬───────────────────────┬───────────────────────┬────────────────┘
        ▼                       ▼                       ▼
  data/papers/{uuid}.pdf   data/database/papers.db   data/vector_store/ (HNSW, cosine)
                                                        │
                    SentenceTransformer(all-mpnet-base-v2) [in-process, cached]
                                                        │
                                       Groq chat.completions (gpt-oss-20b)
```

**Design patterns to name in the interview**
- **Layered (n-tier) architecture / separation of concerns.** Routes only do HTTP mapping (status codes, exceptions to `HTTPException`); services
  hold business rules; storage adapters hold persistence.
- **Dependency injection.** `api/dependencies.py` builds the object graph. Expensive singletons (embedding model, Chroma client, Groq client)
  are cached with `@lru_cache`, so the 420 MB model loads once per process.
- **Repository pattern.** `SQLitePaperRepository` hides SQL. Swapping to Postgres changes one class.
- **Ports and adapters.** `DocumentEmbeddingModel` is a `typing.Protocol`, so tests inject a fake model and production can swap
  SentenceTransformer for OpenAI, Cohere or TEI without touching `EmbeddingService`.
- **Immutable value objects.** `@dataclass(frozen=True)` for `ExtractedPage`, `ParsedDocument`, `EmbeddedChunk`, `VectorSearchResult`.
- **Composition over inheritance.** `DocumentProcessor` is composed of parser, cleaner, chunker and metadata extractor.

### 1.3 End-to-end request flows

#### Flow A: Upload (`POST /api/v1/papers`, multipart)
1. `get_current_user` validates the bearer token with Supabase (`GET {SUPABASE_URL}/auth/v1/user`, 5 s timeout). Invalid → 401; Supabase down → 503.
2. Route checks the filename, reads bytes, calls `PaperService.upload_paper`.
3. Extension check (`.pdf`) → `sha256(content)` → `get_by_content_hash_and_owner`. If one exists → `DuplicatePaperError` → **409 Conflict**.
4. New `paper_id = uuid4()`. The file is saved as `{paper_id}.pdf`, never under the user's filename, which prevents path traversal and collisions.
5. A row is inserted in `papers` with `status = uploaded`, `user_id`, `content_hash`. Response: **201 Created**.

#### Flow B: Process (`POST /papers/{id}/process`)
1. Ownership check (`get_by_id_and_owner`) → 404 if not yours. A non-owner gets 404, not 403, which avoids leaking that the resource exists.
2. Status → `processing`.
3. `DocumentProcessor.process`:
   - `PDFParser.extract`: `page.get_text("text", sort=True)` per page. `sort=True` orders text blocks top-left to bottom-right, which helps with two-column papers.
   - `TextCleaner.clean` per page: Unicode **NFKC** normalization (ligatures like "ﬁ" become "fi", full-width characters become ASCII), CRLF → LF,
     collapse spaces and tabs, trim spaces around newlines, collapse 3+ newlines to 2 (paragraph boundaries are kept).
   - `TextChunker.chunk_document`: per page, split on whitespace, sliding window of **200 words**, stride **160** (overlap 40 = 20%).
     Each chunk keeps `page_number`, a global `chunk_index` and `token_count` (actually a *word* count).
   - `MetadataExtractor.extract`: embedded PDF metadata first, then first-page layout heuristics.
4. `save_processing_result` stores chunks + metadata and sets status `processed`. On `PDFProcessingError` → status `failed` → **422**.

#### Flow C: Index (`POST /papers/{id}/index`)
1. Ownership check, plus the paper must be `processed` → otherwise **409**.
2. `IndexingService.index_paper`: load chunks from SQLite (owner-scoped) → `encode_document(texts, normalize_embeddings=True)` in one batch
   → `delete_by_paper_id` (idempotent re-index) → `collection.upsert(ids=chunk UUIDs, embeddings, documents, metadatas={user_id, paper_id, page_number, chunk_index})`
   → `mark_chunks_embedded`.

> Why three endpoints instead of one? Each step can fail on its own and be retried on its own, each has a clear status transition
> (`uploaded → processing → processed/failed`), and they map one-to-one onto future background jobs. The frontend's `uploadPipeline`
> mutation (`PaperLibrary.tsx`) chains them and shows "Uploading… / Processing… / Creating embeddings…".
> **Weakness:** the work happens inside the HTTP request. A 300-page PDF on CPU can take a long time and hit proxy timeouts (fix: §3, scenario "large PDFs").

#### Flow D: Chat (`POST /api/v1/chat`, body `Query{question, top_k=5 (1..20), paper_id?, conversation_id?}`)
1. Auth → `ChatService.answer_question(query, owner_id)`.
2. If `paper_id` is given, load that paper *only if owned*, to add trusted **metadata context** (title, authors, year, DOI, abstract).
   This handles questions like "who are the authors?", which chunk retrieval often misses because the author line sits on page 1.
3. `RetrievalService.search`: `embed_query` (`encode_query`, normalized) → `ChromaVectorStore.search(top_k, where={"user_id": X} or {"$and":[user, paper]})`.
4. No results and no selected paper → a fixed fallback message with `confidence = 0.0`, and **no LLM call**. That saves cost and prevents a hallucinated answer.
5. Build the context: `[Source i | page=p]\n<chunk text>` blocks.
6. System prompt (in `chat_service.py`): *answer only from supplied metadata and context; if it's not enough, say so; no outside knowledge; Markdown;
   cite as [Source n]; never reveal internal IDs; don't add a Sources section (the app renders sources).*
7. `LLMService.generate` → Groq `chat.completions.create(model, temperature=0.2, max_tokens=1024)`. Exceptions and empty output → `LLMError`.
8. Citations are built **from retrieval results** (paper title from SQLite, owner-scoped, plus page number), so the citations are always real.
9. Response: `Answer{answer_text, citations[], retrieved_chunks[{chunk_id, score}], confidence = max(score), generation_time}`.

#### Flow E: Search (`POST /api/v1/search`)
Same retrieval as chat without the LLM. Returns raw chunks with scores. Useful for debugging retrieval separately from generation.

### 1.4 Key numbers (memorize these)

| Parameter | Value | Reasoning |
|---|---|---|
| Embedding model | `sentence-transformers/all-mpnet-base-v2` | MPNet base, trained contrastively on 1B+ sentence pairs |
| Embedding dimension | **768** | float32 → 3,072 bytes per vector |
| Model max input | **384 word-piece tokens** | Longer text is **silently truncated** |
| Chunk size | **200 words** ≈ 260–300 tokens | Fits under 384 tokens even with technical vocabulary. Big enough to hold one idea, small enough to stay precise |
| Overlap | **40 words (20%)** | A sentence cut at a boundary still appears whole in a neighbouring chunk |
| Chunk boundary | **Never crosses pages** | Every chunk maps to exactly one page, so citations are exact. Cost: a paragraph that spans pages is split |
| Similarity | **cosine** (`hnsw:space = cosine`), vectors L2-normalized | Chroma returns distance = 1 − cos, so score = `clamp(1 − distance, 0, 1)` |
| ANN index | HNSW (Chroma default) | O(log N)-ish search, high recall |
| top_k | default **5**, allowed 1–20 | 5 × ~280 tokens ≈ 1.4k context tokens: cheap and focused |
| LLM | `openai/gpt-oss-20b` on Groq | Open-weight MoE (~21B total, ~3.6B active params): fast and cheap |
| Temperature | **0.2** | Near-deterministic, factual |
| max_tokens | **1024** | Caps cost and latency |
| Upload limit constant | 100 MB (`MAX_UPLOAD_SIZE_MB`) | Defined but not enforced in the route (an honest gap) |
| Auth timeout | 5 s | |

**Back-of-envelope storage.** A 12-page paper gives about 6,000 words, so about 38 chunks (stride 160). 1,000 papers ≈ 38k vectors × 3 KB ≈ **115 MB raw**,
plus HNSW graph overhead (roughly M×2×4 bytes per vector for links), plus the text. That's trivial for Chroma. **1M papers ≈ 38M vectors ≈ 115 GB raw floats**,
which needs a distributed vector DB, quantization, or both (see §2.8 and §3).

### 1.5 Component-by-component details and likely questions

#### PDFParser (`pdf_parser.py`)
- `pymupdf.open` as a context manager, so file handles are always closed.
- Catches `OSError`, `ValueError`, `pymupdf.FileDataError` and re-raises the domain error `PDFProcessingError` (exception chaining with `from error`).
- **Q: Two-column papers?** `sort=True` sorts blocks by vertical then horizontal position, which works for most layouts. For hard layouts use
  `get_text("blocks")` with column detection, or layout-aware parsers: GROBID (made for scientific papers, outputs TEI XML with sections and references),
  Marker, Nougat (academic PDF → Markdown including LaTeX math), Docling, Unstructured.
- **Q: Tables and equations?** Plain text extraction flattens tables and mangles math. Upgrades: `page.find_tables()` in PyMuPDF and storing tables as
  Markdown chunks; Nougat or Marker for LaTeX; figure captions as separate chunks.
- **Q: Headers, footers, references?** Not removed in V1, so a repeating header like "arXiv:xxxx" can pollute chunks. Fixes: detect lines repeated across pages,
  and drop or down-weight the References section (or index it separately for citation questions).

#### TextCleaner
- **NFKC vs NFC:** NFKC applies *compatibility* decomposition: ligatures (ﬁ → fi), superscripts, full-width forms. Good for search; it loses some typographic
  detail, which doesn't matter for retrieval.
- It keeps paragraph breaks because they're semantic boundaries a smarter chunker could use.
- Not yet handled: de-hyphenation across line breaks ("informa-\ntion"). Easy add: `re.sub(r"(\w)-\n(\w)", r"\1\2", text)`.

#### TextChunker
- Validates `max_words > 0` and `0 ≤ overlap < max_words`, which guarantees the window always advances (no infinite loop).
- Loop: `end = min(start + max, n)`; append; break when `end == n`; else `start = end - overlap`.
- **Q: Why words, not tokens?** It's simple, needs no tokenizer dependency, and 200 words stays safely under the 384-token limit. A better version counts
  tokens with the model's own tokenizer (`model.tokenizer`) so chunks are never truncated.
- **Q: Why not semantic chunking?** See §2.4 for a comparison. For papers, the best next step is *structure-aware* chunking: split on section headings
  (Abstract, Introduction, Methods…), then recursively by paragraph, then sentence, and prefix each chunk with `Title > Section` breadcrumbs.

#### MetadataExtractor (the most "engineering-heavy" piece)
- Reads `document.metadata` (embedded title and author) **and** the first page as `get_text("dict")` to get each line's bbox and **max font size**.
- **Title heuristic:** take the lines above the "Abstract" line, skip `arXiv:` lines, find the largest font, and take consecutive lines within 0.25 pt of it.
  If every line has the same font size, fall back to "the first line that looks like an author line".
- **Authors:** lines after the title. Skip emails and URLs, cut at affiliation keywords (university, institute, lab, Google Research…), strip footnote
  markers (*, †, ‡, digits), split on `,` `;` `and`. Keep candidates that look like 2–4 capitalized words. Deduplicate case-insensitively, keeping order.
- **Embedded vs visual:** embedded metadata is used only if valid (title ≤ 250 chars, no "@", no "abstract"; every author passes the name check), because
  embedded PDF metadata is often junk like "Microsoft Word - draft3.docx".
- **Year:** prefer the arXiv stamp year, then `creationDate`, then the first 4-digit year on page 1. That order avoids picking a year from a citation.
- **DOI:** regex `10.\d{4,9}/[-._;()/:A-Z0-9]+`, with trailing punctuation stripped.
- **Abstract:** regex between "Abstract" and the next "Introduction" / "Keywords" / "Index Terms".
- **Q: Why not an LLM for metadata?** Cost, latency and hallucination risk for something deterministic heuristics already do well. Better upgrades:
  GROBID (CRF / deep models trained on papers), or a lookup by DOI/title in Crossref, Semantic Scholar or arXiv APIs for authoritative metadata.
  An LLM with JSON-schema output could be a *fallback* when heuristics return nothing.

#### EmbeddingService
- `embed_chunks` batches all texts in one `encode_document` call (SentenceTransformers batches internally, default batch_size 32) with `normalize_embeddings=True`,
  checks the count matches, and returns tuples of floats.
- `embed_query` uses `encode_query`. Sentence-Transformers v5+ added separate query and document methods because **asymmetric models**
  (E5: "query: " / "passage: ", BGE query instruction, Nomic "search_query:") need different prefixes. For MPNet they are identical, but the code is
  ready for an asymmetric model.
- **Why normalize?** With unit vectors, cosine = dot product, and L2 distance² = 2 − 2·cos, so all three metrics give the *same ranking*.
- The `Protocol`-typed model makes tests fast: a fake model returns fixed vectors.

#### ChromaVectorStore
- `PersistentClient(path)`: embedded, persists HNSW and SQLite metadata on disk.
- **One collection** `paper_chunks` with `{"hnsw:space": "cosine"}` for all users; tenancy is enforced with metadata `where` filters (see §2.8 for alternatives).
- `upsert` validates a one-to-one match between chunk IDs and embedding IDs (it raises on mismatch).
- `delete_by_paper_id` uses `$and` on paper_id **and** user_id, so one user can never delete another user's vectors.
- `search`: `query(query_embeddings, n_results=top_k, where, include=[documents, metadatas, distances])`.
- **Q: Is metadata filtering pre- or post-filtering?** Chroma applies the `where` filter and searches within the matching set, so you still get up to
  k results from that user. With a naive *post*-filter (search the global top-k, then filter) a small tenant could get zero results. See §2.7.

#### SQLitePaperRepository
- Tables: `papers` (id, user_id, content_hash, title, authors (JSON), abstract, year, doi, journal, keywords, stored_filename UNIQUE, total_pages, total_chunks,
  uploaded_at, status) and `chunks` (id, paper_id FK `ON DELETE CASCADE`, chunk_index, page_number, chunk_text, token_count, embedding_created; `UNIQUE(paper_id, chunk_index)`).
- Indexes: **unique partial index** `(user_id, content_hash) WHERE content_hash IS NOT NULL` (DB-level duplicate guarantee, not just app-level), plus `idx_papers_user_id`.
- **Schema migration on startup:** `PRAGMA table_info` and `ALTER TABLE ADD COLUMN` for `user_id` and `content_hash`. Legacy rows from before auth have NULL
  user_id and are intentionally hidden.
- `PRAGMA foreign_keys = ON` on every connection (SQLite has it off by default).
- **Q: Why keep chunk text in SQLite *and* in Chroma?** SQLite is the source of truth (re-embed with a new model without re-parsing PDFs, show chunks, backfill).
  Chroma is a derived index. That separation is what makes an embedding-model migration easy.

#### LLMService
- Thin wrapper: wraps any exception as `LLMError`, rejects empty answers.
- **Missing for production:** retries with exponential backoff (tenacity is already in requirements), timeout, fallback model, token usage logging, streaming.

#### ChatService and prompt design
- Context is labelled `[Source i | page=p]` so the model can cite `[Source i]` and the UI can map that to the citation list.
- Metadata context is called "trusted" because it comes from our extractor, not the user.
- **Confidence = max cosine similarity.** Be honest: it measures *retrieval similarity*, not answer correctness. MPNet cosine scores aren't calibrated
  (unrelated text often scores 0.2–0.4). The frontend hides it (commit "hidden confidence score"). Better options: reranker probability, an NLI-based
  faithfulness score, or an LLM-judge groundedness score.

#### Auth (`core/auth.py`)
- `HTTPBearer(auto_error=False)`, then an explicit 401 with a clear message.
- Validation by **calling Supabase** `/auth/v1/user` with the token and the publishable key. Pro: it respects revocation and logout right away and needs no key handling.
  Con: one network round-trip (tens of ms) on *every* request, plus a hard dependency on Supabase availability (503 if it's down).
- **Production alternative:** verify the JWT **locally** with Supabase's JWKS public keys (asymmetric ES256/RS256 signing keys; older projects use an HS256 shared secret).
  Check `exp`, `aud = "authenticated"` and `iss`, cache the JWKS, and accept up to `exp` (short-lived tokens, about 1 h) as the revocation window.
  `pyjwt` or `python-jose` does this.
- **Why 404 instead of 403 for other users' papers?** It avoids leaking which resource IDs exist (enumeration protection).

#### Frontend
- `api/client.ts` pulls the session from `supabase.auth.getSession()`, attaches `Authorization: Bearer`, and normalizes errors into `ApiError(message, status)` using FastAPI's `detail`.
- TanStack Query: `useQuery(["papers"])` for the library; `useMutation` for upload pipeline, delete and ask; `invalidateQueries` refreshes the list.
- Client-side PDF check (MIME or extension) is only for UX. The server validates again.
- Chat answers are rendered with ReactMarkdown + GFM (tables). Citations are shown from structured data, not parsed from text.
- Vercel `rewrites` everything to `index.html` (SPA routing).

### 1.6 Security and privacy

- **Tenant isolation at three layers:** SQL `WHERE user_id = ?`, Chroma `where user_id`, and the ownership check before every action. Tested in `tests/storage/test_sqlite.py::test_owner_cannot_access_another_users_paper`, `tests/storage/test_chroma.py::test_search_returns_only_vectors_owned_by_current_user` and `tests/services/test_paper_service.py` (same PDF allowed for different users, rejected for the same user). Gap: there's no API-level cross-user test yet; add one that checks user B gets 404 on user A's paper.
- **File safety:** stored under a server-generated UUID name; the user's filename is never used as a path.
- **Secrets:** `.env` via pydantic-settings; the Groq key never reaches the browser; only the Supabase *publishable* key is public (by design).
- **CORS:** explicit allowed origins list.
- **Gaps to admit and fix:** no upload size limit enforced (read the whole file into memory → DoS risk); no MIME or magic-byte check (`%PDF-`);
  no rate limiting (LLM cost abuse); **prompt injection via document text** (a PDF saying "ignore previous instructions…"). Mitigations: delimit context,
  instruct the model to treat context as data, strip suspicious patterns, apply output filters; the blast radius is limited because the model has no tools.
  Also: no audit log; delete isn't atomic across Chroma, the file and SQLite (a crash midway leaves orphans; fix with a reconciler job or an outbox pattern).

### 1.7 Testing strategy

- 63 pytest tests: `models/` (Pydantic validation such as page_number ≥ 1, score in [0, 1], top_k 1..20), `services/` (parser, cleaner, chunker, metadata,
  document processor, embedding with a fake model, indexing, paper service), `storage/` (SQLite, Chroma with a temp directory, file storage),
  `api/` (FastAPI `TestClient` with dependency overrides: upload 201, non-PDF 400, list/get/delete, missing → 404, process creates chunks, duplicate upload → 409). Ownership boundaries are tested at the storage and service levels.
- **Why it works:** DI plus Protocols mean no network or model downloads in unit tests.
- **Missing:** RAG quality evaluation (§2.10), load tests, end-to-end browser tests.

### 1.8 Deployment

- Backend: `python:3.13-slim`, `PYTHONUNBUFFERED`, `PIP_NO_CACHE_DIR`, `uvicorn --host 0.0.0.0 --port ${PORT:-8000}` (PaaS-friendly `$PORT`).
- docker-compose: named volume `research_data:/data` with `DATA_DIR=/data`, so PDFs, SQLite and Chroma survive restarts.
- Frontend on Vercel with `VITE_API_BASE_URL` pointing to the API.
- **Image size concern:** torch + sentence-transformers make the image several GB, and the model downloads from the Hugging Face Hub on first use (cold start).
  Fixes: CPU-only torch wheels, pre-download the model at build time (`RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('...')"`),
  or move embeddings to a separate service (Hugging Face TEI) or an API.
- **Statefulness:** SQLite and embedded Chroma mean **one replica only**. Horizontal scaling needs Postgres plus a server vector DB (§3).

### 1.9 Weaknesses to raise yourself, each with a fix

Raising these before the interviewer does shows seniority.
1. Synchronous heavy work in request handlers. **Fix:** task queue (Celery/RQ/Arq + Redis, or cloud queues), with status polling or SSE/websockets.
2. Dense-only retrieval misses exact terms (acronyms, equation names, dataset IDs). **Fix:** hybrid BM25 + RRF (§2.6).
3. No reranker, so top-5 precision is limited. **Fix:** retrieve 30–50, then cross-encoder rerank to 5 (§2.7).
4. No conversation memory. **Fix:** store turns, condense follow-up questions into standalone queries before retrieval.
5. Uncalibrated confidence. **Fix:** reranker score, faithfulness check, or abstain threshold.
6. Silent truncation risk at 384 tokens. **Fix:** token-based chunking.
7. Config drift (chunk_size 800 unused). **Fix:** inject settings.
8. No evaluation harness. **Fix:** golden set + RAGAS in CI (§2.10).
9. Scanned PDFs produce 0 chunks with no error. **Fix:** detect empty text, then OCR fallback (§2.11).
10. Per-request auth round-trip to Supabase. **Fix:** local JWKS verification.
11. `SQLitePaperRepository` is constructed per request and runs schema init each time. **Fix:** a cached singleton or proper migrations (Alembic once on Postgres).

---

## Part 2: Concepts the interviewer will probe

### 2.1 Why RAG? Why not retraining, fine-tuning or just long context?

**RAG (Retrieval-Augmented Generation)** = retrieve relevant external text at query time, then put it in the prompt so the LLM answers *from evidence*.
(Lewis et al., 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks".)

| Criterion | RAG | Fine-tuning (full / LoRA / QLoRA) | Pre-training / retraining | Long-context stuffing |
|---|---|---|---|---|
| New document available | **Instantly** (index it) | Needs a training run | Not realistic | Instantly, but bounded by the window |
| Per-user private data | **Metadata filter per user** | One model per user? Not feasible | No | Yes, but per request |
| Citations / traceability | **Yes**: chunk + page | No: knowledge is mixed into the weights | No | Partial |
| Hallucination control | Strong (grounded, can abstain) | Weak: can confidently mix facts | Weak | Medium ("lost in the middle") |
| Forgetting / deletion (GDPR) | **Delete vectors** | Need to retrain; machine unlearning is hard | Same | n/a |
| Cost | Cheap embeddings + small context | GPU training + data labelling | Millions of dollars | Cost scales with tokens on *every* query |
| What it teaches | **Knowledge** | **Behaviour / style / format / domain language** | Everything | Knowledge |

**Key sentence:** *"Fine-tuning teaches a model **how** to respond; RAG gives it **what** to respond with. My problem is knowledge that is private,
per-user, changes daily and needs citations. That's exactly RAG's job. I'd consider fine-tuning only on top of RAG, for example to make
a small model follow my citation format reliably, or to fine-tune the *embedding model* on domain pairs to improve retrieval."*

- **When is fine-tuning right?** A fixed output schema, domain jargon, tone, tool-calling behaviour, latency/cost reduction by distilling into a
  smaller model, or a task skill (classification). **RAFT** (Retrieval-Augmented Fine-Tuning) trains the model to use retrieved context and ignore
  distractor documents, so the two approaches combine.
- **Why not long context alone?** 1M-token windows exist, but: (1) cost and latency scale with input tokens on every question; (2) "lost in the middle":
  recall drops for facts buried mid-context; (3) a user's library can exceed any window; (4) no natural citations; (5) no per-user index.
  Long context *helps* RAG: you can afford a larger top-k or whole sections. **Prompt caching** plus long context can beat RAG for a *single* small document set
  that gets queried many times.
- **Fine-tuning gotchas:** catastrophic forgetting; training on facts gives weak recall and more hallucination of near-facts; data must be curated;
  every update needs a new training run.

### 2.2 LangChain: what it is, and why I didn't use it

**What LangChain is:** an orchestration framework for LLM apps. Core abstractions:
- **Document loaders** (`PyMuPDFLoader`, `PyPDFLoader`, `UnstructuredPDFLoader`), producing `Document(page_content, metadata)`.
- **Text splitters**: `RecursiveCharacterTextSplitter` (tries `["\n\n", "\n", " ", ""]` in order, with `chunk_size` / `chunk_overlap`), `TokenTextSplitter`,
  `MarkdownHeaderTextSplitter`, `SemanticChunker` (experimental).
- **Embeddings** wrappers (`HuggingFaceEmbeddings`, `OpenAIEmbeddings`).
- **VectorStores** (`Chroma`, `FAISS`, `Qdrant`, `PGVector`) and **Retrievers** (`.as_retriever(search_type="similarity" | "mmr", search_kwargs={"k":5, "filter":...})`),
  `BM25Retriever`, `EnsembleRetriever` (hybrid via weighted RRF), `ContextualCompressionRetriever` + `CrossEncoderReranker`,
  `MultiQueryRetriever`, `ParentDocumentRetriever`, `SelfQueryRetriever`.
- **Prompt templates**, **chat models**, **output parsers** / `with_structured_output`.
- **LCEL** (LangChain Expression Language): `retriever | format_docs`, `{"context": ..., "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`.
- **LangGraph**: a stateful graph or agent runtime (nodes, edges, cycles, checkpointing). Used for agentic RAG, CRAG and self-RAG loops.
- **LangSmith**: tracing and evaluation platform.

**Why I built it from first principles:**
1. **Learning and explainability.** I can explain every step (chunk maths, filter syntax, score conversion) because I wrote it.
2. **Control over the critical parts.** Page-aware chunking, owner-scoped filters on every call, application-built citations, and a custom metadata extractor
   don't map neatly onto generic abstractions.
3. **Fewer dependencies and less churn.** LangChain's API has changed a lot between versions; the pipeline here is about five small services.
4. **Testability.** Plain classes with Protocol-typed dependencies are trivial to fake.

**But I know how it maps.** If asked "rewrite it in LangChain":
```python
loader = PyMuPDFLoader(path)                                # PDFParser
docs = loader.load()                                        # one Document per page, metadata["page"]
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)   # TextChunker
chunks = splitter.split_documents(docs)
for c in chunks: c.metadata |= {"user_id": uid, "paper_id": pid}
vs = Chroma(collection_name="paper_chunks", embedding_function=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"), persist_directory=...)
vs.add_documents(chunks, ids=[...])
dense = vs.as_retriever(search_kwargs={"k": 20, "filter": {"user_id": uid}})
sparse = BM25Retriever.from_documents(user_chunks); sparse.k = 20
hybrid = EnsembleRetriever(retrievers=[sparse, dense], weights=[0.4, 0.6])     # weighted RRF
reranked = ContextualCompressionRetriever(
    base_compressor=CrossEncoderReranker(model=HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base"), top_n=5),
    base_retriever=hybrid)
chain = ({"context": reranked | format_docs, "question": RunnablePassthrough()}
         | prompt | ChatGroq(model="openai/gpt-oss-20b", temperature=0.2) | StrOutputParser())
```
**When I *would* use it:** fast prototyping, many integrations, agents with tools (LangGraph), or when the team already standardizes on it.
**LlamaIndex** is more retrieval/index-centric (node parsers, `VectorStoreIndex`, query engines, built-in eval). Haystack is pipeline-centric and production-oriented.
DSPy *programs and optimizes* prompts.

### 2.3 Ingestion pipeline: general best practice vs this project

`Load → Parse/extract → Clean → (OCR if needed) → Structure (sections, tables) → Chunk → Enrich metadata → Embed → Index → Verify`

Production additions:
- **Idempotency** (content hash, as in this project), **versioning** (re-ingest when the parser or model changes, keep `embedding_model` and `parser_version` per chunk),
  **async workers**, **dead-letter queue** for failed files, **progress status**.
- **Enrichment:** section titles, contextual headers ("Paper X > 3. Method"), **Anthropic-style contextual retrieval** (an LLM writes a 50–100 token
  description of the chunk in the context of the whole document, prepended before embedding *and* BM25 indexing; Anthropic reported far fewer
  retrieval failures, especially when combined with reranking).
- **Deduplication** (exact hash; near-duplicate via MinHash/SimHash or by DOI).
- **PII redaction** if needed.

### 2.4 Chunking strategies (expect "why 200 words?")

| Strategy | How | Pros | Cons |
|---|---|---|---|
| Fixed-size (words/tokens) + overlap (**this project**) | Sliding window | Simple, predictable, fast | Cuts mid-sentence or mid-idea |
| Recursive character | Split by `\n\n` → `\n` → sentence → word until under size | Respects structure | Variable sizes |
| Sentence-based | NLTK/spaCy sentences grouped up to N tokens | Clean boundaries | Long sentences in papers |
| Structure/heading-aware | Split by sections (Markdown or PDF heading detection) | Semantically coherent, great for papers | Needs good parsing |
| Semantic chunking | Embed sentences; split where adjacent similarity drops below a threshold | Topic-coherent | Costly; results vary |
| Parent–child (small-to-big) | Index small chunks (precision), return parent section (context) | Best of both | More storage and complexity |
| Sentence window | Retrieve one sentence, expand ±k sentences | High precision | |
| Late chunking | Embed the whole doc with a long-context model, then pool token embeddings per chunk | Chunks "know" document context | Needs long-context embedder |
| Propositional | LLM rewrites text into atomic facts | Very precise | Expensive |

**Size trade-off:** small chunks give precise embeddings and better ranking but lose context; large chunks keep context but blur the embedding
("averaging" several topics) and waste LLM tokens. Typical sweet spot: 200–500 tokens, 10–20% overlap. **Choose it with evaluation**: run Recall@k
on a golden set for 128/256/512 tokens.

**Why page-bounded?** Exact page citations. The trade-off is splitting cross-page paragraphs. Alternative: chunk across pages and store a `page_start`–`page_end` range.

### 2.5 Embeddings: how they work

- A **bi-encoder** (sentence-transformer) maps text to a fixed-size vector. MPNet is a transformer; its token outputs are **mean-pooled** and then normalized.
- **Training:** contrastive learning (MultipleNegativesRankingLoss / InfoNCE). Pull (query, positive) together, push in-batch negatives apart.
  all-mpnet-base-v2 was trained on 1B+ pairs (Reddit, S2ORC citation pairs, StackExchange, MS MARCO, NQ…).
- **Why semantic search works:** "How does the model reduce overfitting?" matches "we apply dropout and weight decay" without shared words.
- **Choosing a model:** see the **MTEB** leaderboard (Massive Text Embedding Benchmark), retrieval tab. Consider dimension, max tokens, language, licence and latency.
  Options: `BAAI/bge-base-en-v1.5` / `bge-m3` (multilingual, dense + sparse + multi-vector in one model, 8k tokens), `intfloat/e5-*` (needs "query:" / "passage:" prefixes),
  `gte`, `nomic-embed-text` (8k tokens, Matryoshka), `jina-embeddings-v3`, hosted OpenAI `text-embedding-3-small/large`, Cohere embed v3, Voyage.
  For science text: SPECTER2 / SciNCL (citation-trained) suit *paper-level* similarity.
- **Matryoshka embeddings:** vectors trained so the first 256/128 dims still work. Truncate to save storage.
- **Quantization of vectors:** float32 → int8 (4× smaller) or binary (32×) with a rescoring step.
- **Changing the embedding model = re-embed everything.** Vectors from different models are not comparable. That's why chunk text lives in SQLite.

### 2.6 Dense vs sparse retrieval, and hybrid

**Sparse (lexical):** vectors of vocabulary size, mostly zeros. Stored in an **inverted index** (term → posting list).
- **TF-IDF:** term frequency × log(N/df).
- **BM25** (the standard):
  `score(D,Q) = Σ_{q∈Q} IDF(q) · f(q,D)·(k1+1) / ( f(q,D) + k1·(1 − b + b·|D|/avgdl) )`
  - `k1` (≈1.2–2.0) saturates term frequency, so the 10th occurrence adds little.
  - `b` (≈0.75) normalizes for document length.
  - `IDF(q) = ln( (N − n(q) + 0.5)/(n(q) + 0.5) + 1 )`: rare terms weigh more.
- **Learned sparse:** **SPLADE** (a BERT MLM head predicts weights over the vocabulary, including *expansion* terms), BGE-M3 sparse output.
  These keep inverted-index efficiency and add some semantics.
- Strengths: exact matches (names, acronyms like "LoRA", "BLEU-4", gene IDs, error codes, equation numbers), no training needed, interpretable, strong
  out-of-domain (BEIR showed BM25 is hard to beat zero-shot).
- Weakness: vocabulary mismatch (synonyms, paraphrase).

**Dense (semantic):** low-dimensional (384–4096) real-valued vectors from neural encoders; ANN index.
- Strengths: synonyms, paraphrases, cross-lingual, natural-language questions.
- Weaknesses: exact identifiers, rare terms, numbers, out-of-domain text; needs a GPU or CPU to embed; less interpretable.

**Hybrid = run both and fuse the results.**
- **Reciprocal Rank Fusion (RRF):** `RRF(d) = Σ_r 1 / (k + rank_r(d))`, with `k = 60` by convention (Cormack et al., 2009).
  It uses *ranks only*, so no score normalization is needed (BM25 scores are unbounded, cosine is in [-1, 1]). Robust, parameter-light.
- **Weighted score fusion (convex combination):** `α·norm(dense) + (1−α)·norm(bm25)` with min-max or z-score normalization. Tunable but sensitive.
- **Why it helps for papers:** questions mix concepts ("how do they regularize?") with exact terms ("what is the BLEU on WMT14 En-De?").
  Typical gain is several points of Recall@k.

**How I'd add hybrid to *this* code (concrete):**
1. Add a `SparseIndex` adapter. Simplest: `rank_bm25.BM25Okapi` built per user from SQLite chunks (fine for small libraries). Better: **SQLite FTS5**
   (built-in BM25 via `bm25()`), already in our stack: `CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_text, content='chunks', ...)`
   and a query filtered by user's paper IDs. Production: Qdrant sparse vectors, Elasticsearch/OpenSearch, or Postgres `tsvector`.
2. `RetrievalService.search`: dense top-30 + sparse top-30 → RRF → top-N to the reranker or LLM.
3. Evaluate the gain on the golden set before shipping.

### 2.7 Rerankers: cross-encoder vs bi-encoder (and ColBERT)

- **Bi-encoder:** encode query and document **separately** → compare vectors. Documents are pre-computed, so it's fast at scale, but query and
  document tokens never "see" each other, which loses fine-grained interaction.
- **Cross-encoder:** feed `[CLS] query [SEP] document [SEP]` **together** through a transformer → one relevance score (logit or sigmoid). Full
  token-to-token attention gives much higher accuracy, but nothing can be pre-computed: it costs one forward pass per (query, doc) pair, O(n) per query. That's too slow for
  millions of documents, but perfect for re-scoring a **shortlist**.
- **Two-stage retrieve-then-rerank:** retrieve 50–100 with a cheap method (hybrid) for **recall**, then rerank to 5 with a cross-encoder for **precision**.
- Models: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast), `BAAI/bge-reranker-v2-m3` (multilingual, strong), `mixedbread-ai/mxbai-rerank`, `jina-reranker`,
  hosted Cohere Rerank, Voyage rerank. **LLM rerankers** (RankGPT, listwise) are strongest and slowest.
- Latency rule of thumb: MiniLM-L6 on CPU ≈ a few ms per pair; 50 pairs batched ≈ 50–200 ms on CPU, much faster on a GPU.
- **ColBERT (late interaction):** stores a vector *per token*; score = Σ over query tokens of the max similarity with doc tokens (MaxSim). Close to cross-encoder quality
  with pre-computable docs, at a high storage cost. Implementations: ColBERTv2 / PLAID, RAGatouille; Qdrant and Vespa support multi-vector search.
- **Bonus:** the reranker score is a much better "confidence" signal than raw cosine, so I'd use it for abstention (if the top reranker score is below a threshold,
  answer "not found in your papers").
- **MMR (Maximal Marginal Relevance):** `argmax λ·sim(d,q) − (1−λ)·max sim(d, selected)`. Reduces redundant near-duplicate chunks (overlap makes neighbours similar).

### 2.8 Similarity search: metrics and algorithms

**Metrics**
- **Cosine similarity** = (a·b)/(‖a‖‖b‖): the angle; ignores magnitude.
- **Dot product (inner product):** magnitude matters; equals cosine for normalized vectors. Used by models trained with dot product.
- **Euclidean (L2):** for unit vectors, `‖a−b‖² = 2 − 2cos`, so it gives the same ranking.
- Others: Manhattan (L1), Hamming (binary vectors), Jaccard (sets/MinHash).
- **Rule:** use the metric the embedding model was trained with. For normalized sentence-transformers, cosine = dot.

**Exact vs approximate nearest neighbour (ANN)**
- **Exact kNN (brute force / flat):** compare against all N vectors, O(N·d). 100% recall. Fine up to about 100k–1M vectors on a GPU (FAISS `IndexFlatIP`).
- **ANN** trades a little recall for big speed-ups. Measure **recall@k vs QPS vs memory**.

| Algorithm | Idea | Key params | Notes |
|---|---|---|---|
| **HNSW** (Hierarchical Navigable Small World) | Multi-layer proximity graph; greedy search from a sparse top layer down to the dense bottom layer | `M` (links per node, 16–64), `efConstruction` (100–400), `efSearch` (≥k; higher means better recall, slower) | Best recall/latency for in-memory data; memory-heavy; deletes are tricky. Used by Chroma, Qdrant, Weaviate, pgvector, Milvus |
| **IVF** (Inverted File) | k-means into `nlist` clusters; search only the `nprobe` nearest clusters | `nlist` (≈√N to 4√N), `nprobe` | Needs training; good for large data; recall depends on nprobe |
| **PQ** (Product Quantization) | Split the vector into m sub-vectors, quantize each to 256 centroids (1 byte) | `m`, `nbits` | 768-dim float32 (3,072 B) becomes e.g. 96 B: **32× compression**; distances via lookup tables (ADC) |
| **IVF-PQ / IVF-HNSW-PQ** | Combinations | | Billion-scale (FAISS) |
| **ScaNN** (Google) | Anisotropic quantization | | Very fast for MIPS |
| **DiskANN / Vamana** | Graph on SSD + compressed vectors in RAM | | Billion-scale on one machine; Milvus, Azure |
| **LSH** | Random hyperplane hashing | | Older; lower recall |
| Scalar quantization (SQ8) / binary quantization | float → int8 / 1 bit, plus rescoring | | Qdrant, Weaviate, FAISS |

**Filtering + ANN (important for multi-tenancy):**
- **Post-filtering:** ANN top-k, then drop non-matching results. You may end up with fewer than k (bad for small tenants).
- **Pre-filtering:** compute the allowed IDs, then search only those (brute force if the set is small, filtered graph traversal if large). Qdrant's
  payload index plus "filterable HNSW" handles this well, and Chroma applies `where` before the vector search.
- **Partitioning:** a collection, namespace or partition per tenant. Strong isolation; overhead with many tenants.

### 2.9 Vector databases: FAISS vs Chroma vs Qdrant (and others)

| | **FAISS** | **ChromaDB** | **Qdrant** | Milvus | Weaviate | pgvector | Pinecone | Elasticsearch / OpenSearch |
|---|---|---|---|---|---|---|---|---|
| Type | **Library** (C++/Python, Meta) | Embedded or client-server DB | Server DB (Rust) | Distributed DB | Server DB (Go) | Postgres extension | Managed SaaS | Search engine + kNN |
| Persistence | Manual (`write_index`) | Built-in | Built-in (WAL, snapshots) | Built-in (object storage) | Built-in | Postgres | Managed | Built-in |
| Metadata filtering | ❌ (keep it yourself) | ✅ `where` | ✅✅ rich payload filters + indexes | ✅ | ✅ | ✅ full SQL | ✅ | ✅✅ |
| Index types | Flat, IVF, PQ, HNSW, GPU | HNSW | HNSW (+ quantization), sparse vectors, multi-vector | HNSW, IVF, DiskANN, GPU | HNSW | HNSW, IVFFlat | Proprietary | HNSW |
| Hybrid (BM25 + dense) | ❌ | Limited (newer versions add sparse/full-text search) | ✅ sparse vectors + RRF fusion via Query API | ✅ | ✅ built-in BM25 + fusion | ✅ with tsvector (manual fusion) | ✅ sparse-dense | ✅✅ native |
| Scale | Billions (single node, GPU) | Small–medium | Large; sharding + replication | Billions; cloud-native | Large | Medium (tens of millions with tuning) | Large | Large |
| Ops effort | Code-level | **Zero** | Low–medium (Docker / cloud) | High | Medium | Low if you already run Postgres | Zero | Medium–high |
| Best for | Research, benchmarking, offline batch, GPU speed | **Prototypes, MVPs, local apps** | **Production RAG with filtering and multi-tenancy** | Very large enterprise | Hybrid / modular | One DB for relational + vectors | No-ops teams | Existing search stack |

**"Why ChromaDB for this project?"**
> "This is a single-node MVP whose goal was to make the whole RAG lifecycle transparent. Chroma is embedded, so there's no extra server to deploy;
> it persists to disk next to SQLite in one Docker volume; it supports metadata `where` filters, which I needed for per-user isolation; it uses HNSW with
> cosine; and the Python API is minimal. At a few thousand papers per user it's more than fast enough. Its limits are single-node scaling, weaker
> hybrid support, and tenant isolation by filter only. I wrapped it behind `ChromaVectorStore`, so moving to Qdrant or pgvector is one adapter class."

**"Why FAISS / Qdrant for your summer research?"** (adjust to what you actually did)
> "In research I needed to **benchmark retrieval at scale and compare index types**: flat (exact) vs IVF vs HNSW vs PQ. I measured recall@k against
> exact search, latency and memory. FAISS is a library, not a database, so it gives that low-level control, GPU acceleration and reproducibility
> with no server overhead. Metadata didn't matter much there. I used **Qdrant** where I needed a production-like setup: rich payload filtering, hybrid
> dense + sparse vectors with fusion, quantization, snapshots, and a REST/gRPC server that several processes could share. So: FAISS for experimentation and
> raw speed, Qdrant for serving with filters, Chroma for a self-contained app."

**"When pgvector?"** When you already run Postgres: one transactional store for papers, chunks *and* vectors (no consistency problem between two stores),
SQL joins plus row-level security for tenancy, HNSW index. Upper limit around tens of millions of vectors per instance.

### 2.10 Evaluation: RAGAS, metrics and benchmarks

**Evaluate retrieval and generation separately.** If the answer is wrong, you need to know whether retrieval missed or the generator failed.

**Retrieval metrics** (need relevance labels: question → relevant chunk IDs or pages):
- **Hit rate / Recall@k:** fraction of questions where (or fraction of relevant items) the relevant chunk is in the top-k.
- **Precision@k:** relevant items in the top-k divided by k.
- **MRR** (Mean Reciprocal Rank): mean of 1/rank of the first relevant result.
- **nDCG@k:** graded relevance with a log discount for position, normalized by the ideal ranking.
- **MAP.**
- Systems metrics: latency p50/p95, cost per query.

**Generation metrics:**
- **Faithfulness / groundedness:** is every claim supported by the retrieved context? (This is the hallucination metric.)
- **Answer relevance:** does it address the question?
- **Correctness:** vs reference answer (exact match, F1, semantic similarity, LLM-judge).
- Citation precision / recall: do cited sources actually support the sentence?

**RAGAS** (open-source, "Retrieval-Augmented Generation Assessment"). Mostly LLM-as-judge, reference-free where possible:
- **Faithfulness** = (# claims in the answer supported by the context) / (total # claims). The LLM splits the answer into atomic statements, then verifies each against the context.
- **Answer relevancy:** the LLM generates n questions *from the answer*; score = mean cosine similarity between them and the original question (penalizes incomplete or off-topic answers).
- **Context precision:** are the relevant chunks ranked high? (Precision@k averaged over the positions of relevant chunks.)
- **Context recall:** fraction of ground-truth-answer claims attributable to the retrieved context (needs a reference answer).
- Also: context entities recall, noise sensitivity, answer correctness, answer similarity.
- RAGAS can **generate a synthetic test set** from your documents (questions of different types: simple, reasoning, multi-context).

**Other frameworks:** TruLens ("RAG triad": context relevance, groundedness, answer relevance), DeepEval (pytest-style LLM unit tests), ARES, LangSmith and Langfuse
eval datasets, Arize Phoenix, promptfoo.

**Benchmarks (know names and what they test):**
- Retrieval: **BEIR** (18 heterogeneous datasets, zero-shot), **MTEB** (embedding models across tasks), **MS MARCO** (passage ranking), **TREC-COVID**, **SciFact** / **SCIDOCS** (scientific!).
- QA / RAG: **Natural Questions**, **TriviaQA**, **HotpotQA** (multi-hop), **KILT**, **RGB** (noise robustness, negative rejection, info integration, counterfactual robustness),
  **CRAG** (Meta's comprehensive RAG benchmark), **MultiHop-RAG**, **FRAMES**, **LongBench**.
- Scientific QA: **QASPER** (questions over NLP papers; a good fit for this project).
- Hallucination: **TruthfulQA**, **HaluEval**, **FActScore** (atomic-fact precision).
- Documents/OCR: DocVQA, InfographicVQA, ChartQA, FUNSD, CORD, SROIE (Part 4).

**How I would evaluate *this* project (concrete plan):**
1. Build a **golden set**: 50–150 questions over 10–20 papers, each with the reference answer + relevant page(s). Mix factual, numeric, method,
   author/metadata, multi-chunk, and **unanswerable** questions (to test abstention). Bootstrap with RAGAS synthetic generation, then hand-verify.
2. **Retrieval eval** (no LLM needed): Recall@5, MRR, nDCG@10, using page-level matches. Run a grid over chunk size {128, 256, 512 tokens}, overlap,
   embedding models (mpnet vs bge vs e5), dense vs hybrid, with and without reranker.
3. **Generation eval:** RAGAS faithfulness, answer relevancy, context precision and recall, with a strong judge LLM. Plus an **abstention rate** on unanswerable questions.
4. Calibrate the judge with ~30 human labels (check agreement).
5. Put it in **CI** as a regression gate (e.g., fail if faithfulness drops more than 3 points) and track it in a dashboard.
6. **Online:** thumbs up/down, citation click-through, "no answer" rate, latency and cost per query; sample production traces for review.

### 2.11 Hallucination: causes, prevention, detection

**Types:** *intrinsic* (contradicts the source) vs *extrinsic* (adds unsupported info); *factual* vs *faithfulness* hallucination. In RAG we mostly care about
**faithfulness to retrieved context**.

**Causes in a RAG system:**
1. Retrieval miss: the right chunk isn't in the top-k, so the model "fills the gap" from its own knowledge.
2. Noisy or contradictory context; distractor chunks.
3. Chunk boundary cut the key sentence.
4. The prompt allows outside knowledge; high temperature.
5. Context too long ("lost in the middle").
6. Numeric and table data mangled by parsing.

**Mitigations in this project:**
- System prompt: *answer only from context; say so if insufficient; no outside knowledge.*
- Temperature 0.2.
- Labelled sources plus inline `[Source n]` citations.
- **Citations generated by code from retrieval results**, so page numbers and titles can't be hallucinated.
- No-results short-circuit: no LLM call, fixed "couldn't find" response.
- Trusted metadata context for author/title questions (instead of hoping retrieval finds page 1).
- Overlapping chunks.

**Additional techniques I'd add:**
- Better retrieval (hybrid + reranker): most RAG hallucinations are really retrieval failures.
- **Abstention threshold** on the reranker score.
- **Post-generation verification:** split the answer into claims → check each against the cited chunk with an **NLI model** (entailment), e.g. a DeBERTa MNLI cross-encoder,
  or an LLM judge. Drop or flag unsupported sentences. **Citation verification:** each `[Source n]` must entail its sentence.
- **Self-consistency:** sample several answers; disagreement signals uncertainty.
- **Structured output:** JSON `{answer, supporting_quotes[], cannot_answer: bool}`. Requiring verbatim quotes, and checking that they appear in the chunk with a
  string match, is a cheap and strong guard.
- **Corrective RAG (CRAG) / Self-RAG:** grade retrieved docs; if they're bad, rewrite the query or fall back.
- Guardrails libraries (NeMo Guardrails, Guardrails AI).
- Measure it with RAGAS faithfulness plus a hallucination rate on the golden set.

### 2.12 Multimodal input: "what if the user uploads a photo instead of a PDF?"

**Today:** the backend rejects non-`.pdf` uploads (400). A *scanned* PDF passes but `get_text` returns empty strings, so there are 0 chunks and retrieval finds nothing (a silent failure).

**Design:**
1. **Accept** `image/png`, `image/jpeg`, `image/heic`, `image/tiff` (and multi-page TIFF). Validate by **magic bytes**, not extension.
2. **Route by content type** in a `DocumentLoader` strategy registry: PDF-with-text-layer → PyMuPDF; image or scanned PDF → OCR path;
   DOCX → python-docx; HTML → readability parser.
   - **Detect scanned PDFs:** characters per page below a threshold (e.g. < 50 chars) **or** the page is mostly one image → rasterize with
     `page.get_pixmap(dpi=300)` → OCR.
3. **Image preprocessing** (phone photos): EXIF orientation fix, perspective correction / document edge detection (OpenCV contours + warpPerspective), deskew,
   denoise, adaptive thresholding, contrast (CLAHE), upscale to ~300 DPI equivalent.
4. **OCR engine:** Tesseract (`pytesseract`; PyMuPDF also has `page.get_textpage_ocr()` using Tesseract), PaddleOCR (strong, multilingual), docTR, EasyOCR;
   or cloud (Google Document AI / Vision, AWS Textract, Azure Document Intelligence). Keep the **per-word confidence and bboxes**.
5. **Or a VLM** (Gemini, GPT-4o-class, Claude, Qwen-VL) to transcribe to Markdown, which handles tables, handwriting and layout better. Use OCR + VLM together for
   reliability (see Part 4).
6. **Figures, charts and diagrams:** caption or describe with a VLM and index the description as text ("figure chunk" with page and figure number), or use **multimodal
   embeddings** (CLIP / SigLIP, Jina-CLIP, Cohere embed v3 image) or **ColPali** (embeds page *images* directly with late interaction; no OCR needed)
   to retrieve page images, then send the image to a VLM for the answer.
7. Store `source_type = "ocr"` and the OCR confidence per chunk. Warn the user ("low OCR quality on page 3"), and down-weight low-confidence chunks.
8. Same downstream pipeline: chunk → embed → index. Citations still point to page and image.

### 2.13 Query-side improvements (likely follow-ups)

- **Conversational query rewriting:** "what about its limitations?" becomes "What are the limitations of <paper/method from previous turn>?". Needs chat history (store it in SQLite with `conversation_id`).
- **Multi-query:** the LLM generates 3–5 paraphrases; retrieve for each; union + RRF.
- **HyDE:** the LLM writes a *hypothetical answer*; embed that instead of the question (answer-to-answer similarity is closer).
- **Query decomposition** for multi-hop or comparison questions ("compare paper A and B results"): sub-questions per paper → retrieve → synthesize.
- **Step-back prompting:** ask a more general question first.
- **Routing:** metadata questions → SQL/metadata; "summarize the paper" → map-reduce over all chunks (top-k retrieval is the wrong tool for summarization);
  specific facts → RAG.
- **Self-query / metadata filter extraction:** "papers from 2023 about diffusion" → filter `year=2023` + semantic "diffusion".
- **Agentic RAG** (LangGraph): the agent decides to search, re-search, use tools or stop.
- **GraphRAG** (Microsoft): an entity/relation graph plus community summaries for global "what are the themes across my library?" questions.

---

## Part 3: Production scaling and "what would you change if…" scenarios

### 3.1 Production architecture (V2)

```text
                   CDN (Vercel/CloudFront) ── React SPA
                               │
                        API Gateway / LB  (TLS, WAF, rate limit per user)
                               │
             ┌─────────────────┴──────────────────┐
             ▼                                    ▼
   FastAPI API pods (stateless, N replicas,     SSE / WebSocket for streaming answers
   autoscaled on CPU/RPS), JWT verified          and ingestion progress
   locally via cached JWKS
     │        │            │                   │
     │        │            │ enqueue job       │ cache
     │        │            ▼                   ▼
     │        │      Queue (Redis/SQS/RabbitMQ) Redis: semantic cache, rate limits,
     │        │            │                    session state
     │        │            ▼
     │        │   Ingestion workers (Celery/Arq/Temporal), autoscaled on queue depth
     │        │     parse → OCR if needed → chunk → enrich → embed (batch) → upsert
     │        │            │
     ▼        ▼            ▼
  Object storage   Postgres (papers, chunks,   Vector DB: Qdrant cluster or pgvector
  (S3/GCS) PDFs    users, conversations,       (sharded/replicated, payload index on
  pre-signed URL   eval results; RLS)          tenant_id, HNSW + scalar quantization,
  uploads                                       sparse vectors for hybrid)
                               │
           Model serving: embedding service (HF TEI / GPU, batched),
           reranker service (GPU), LLM via gateway (Groq/OpenAI/Anthropic/vLLM)
           with retries, timeouts, fallbacks, token budgets
                               │
   Observability: OpenTelemetry traces, Prometheus/Grafana, structured logs,
   LLM tracing (Langfuse/LangSmith/Phoenix), eval dashboards, cost tracking, alerts
```

**Changes, component by component**
- **Uploads:** pre-signed S3 URLs (the browser uploads directly, keeping large bodies off the API) → event → queue → worker. Return `202 Accepted` + job ID; the frontend polls `GET /papers/{id}` or subscribes via SSE.
- **SQLite → Postgres** (concurrent writers, replicas, backups, Alembic migrations, row-level security on `user_id`).
- **Chroma embedded → Qdrant (or pgvector, or Chroma server/cloud).** Stateless API pods need a shared network vector DB.
- **Embedding at scale:** batch on GPU (TEI). Throughput from hundreds to thousands of chunks/s; use CPU only for queries. Or use a hosted embedding API.
- **LLM gateway:** retries with backoff + jitter (tenacity), circuit breaker, fallback provider, per-user token quotas, **streaming** (SSE) for better perceived latency.
- **Caching:** exact-match cache of question+paper → answer; **semantic cache** (embed the query, hit if cos > 0.95 and same tenant/paper scope); embedding cache by chunk hash.
- **Consistency:** an outbox pattern or reconciliation job between Postgres and the vector DB (deletes, re-index).
- **Security:** rate limiting, upload size and MIME checks, malware scan (ClamAV), prompt-injection filters, audit logs, encryption at rest, data retention and deletion.
- **CI/CD:** lint, type check, tests, **RAG eval gate**, Docker build, staged deploy (canary/blue-green).

**Latency budget per chat (target p95 < 3 s to first token):** auth (local JWT) ~1 ms · query embed ~10–30 ms (CPU) · hybrid search ~10–50 ms · rerank 50 docs ~50–150 ms (GPU) ·
LLM time-to-first-token ~100–500 ms (Groq is very fast) · streaming the rest.

**Scale maths:** 10M chunks × 768 dims × 4 B ≈ 30 GB of vectors (+ HNSW). With int8 scalar quantization that's ~7.5 GB in RAM, with originals on disk for rescoring.
Shard by tenant or hash across nodes; replicate for QPS.

### 3.2 Scenario bank: "I want X. What do you change?"

Answer each one with the same structure: **What changes → where in my code → trade-offs → how I'd verify it.**

1. **"Answers miss exact terms like 'BLEU' or 'Table 3'."** → Add BM25 (SQLite FTS5 now, Qdrant sparse later) + RRF in `RetrievalService`. Verify Recall@5 on keyword-heavy questions.
2. **"Top results are relevant but in the wrong order / the LLM gets noise."** → Retrieve 30–50, cross-encoder rerank to 5 (`bge-reranker-v2-m3`). Measure context precision and nDCG. About +100 ms.
3. **"I want 10,000 concurrent users."** → Stateless API replicas, Postgres, Qdrant cluster, Redis, workers via a queue, local JWT verification, LLM gateway with quotas, autoscaling, caching. Load test with k6/Locust.
4. **"Indexing a 500-page book times out."** → Async background job + progress events; batch embedding on GPU; stream pages (don't load everything into memory); a status machine (`queued/processing/indexed/failed`) with retries and a DLQ.
5. **"Follow-up questions like 'what about its limitations?' fail."** → Conversation memory (turn table) + a query-rewriting step that condenses history into a standalone question; include the last N turns in the prompt within a token budget.
6. **"Users want to chat across ALL their papers and compare them."** → Already supported (no `paper_id` filter). Improve with per-paper diversity (MMR, or top-k per paper), query decomposition per paper, and paper titles in the source labels.
7. **"Summarize the whole paper."** → Top-k retrieval is wrong for this. Use map-reduce or refine summarization over all chunks or sections; precompute and cache a summary at ingestion time; or a long-context LLM with the full text.
8. **"Switch the embedding model to a better one."** → Chunk text is in SQLite, so re-embed into a **new collection** (blue-green), dual-run eval, switch an alias, delete the old one. Store the `embedding_model` version per vector. Never mix vector spaces.
9. **"Switch from Groq to a self-hosted model (data must not leave our VPC)."** → `LLMService` is already an adapter: implement it against **vLLM** (OpenAI-compatible server) or Ollama/TGI with Llama/Qwen/gpt-oss-20b (open weights!). Size GPUs (a 20B MoE fits on one 16–24 GB GPU with quantization). Run the eval suite for parity. Embeddings are already local.
10. **"Groq is down or rate-limited."** → Retries with exponential backoff for 429/5xx, a timeout, a circuit breaker, a fallback provider/model, and a graceful error to the user; queue non-urgent requests.
11. **"Users upload Word docs, HTML, slides, or photos."** → Loader registry by MIME type (§2.12); OCR/VLM path for images; unify into `ExtractedPage`-like units (page/slide/section) so chunking, citations and the rest stay the same.
12. **"Multilingual papers."** → Multilingual embedder (`bge-m3`, `multilingual-e5`), language detection metadata, multilingual reranker, an LLM prompt that answers in the user's language.
13. **"Tables and numbers are answered wrong."** → Table extraction (PyMuPDF `find_tables`, Camelot, Table Transformer, Docling); store tables as Markdown/HTML chunks plus a row-level text form; for numeric questions, a structured-query tool over extracted tables.
14. **"Questions about figures."** → VLM captions per figure (index them), or ColPali page-image retrieval + VLM answer.
15. **"Reduce cost by 10×."** → Smaller/cheaper LLM for simple questions (a **router**), semantic cache, fewer and better chunks (reranker → top-3), prompt caching, shorter outputs, batch embeddings, int8 vectors.
16. **"Reduce latency."** → Streaming, parallel dense + sparse search, a smaller reranker or GPU, cached query embeddings, Groq-class inference, a precomputed metadata route.
17. **"How do you know an answer is right? Add trust."** → Inline citations with highlighted source spans (store char offsets), NLI faithfulness check, abstention, a feedback button, evaluation dashboards.
18. **"Enterprise tenants with strict isolation."** → A collection or namespace per tenant (or a Qdrant shard key per tenant), Postgres schemas or RLS, separate encryption keys, audit logs. Trade-off: many small collections add overhead, so use a hybrid approach (big tenants get dedicated collections, small ones share with a filter).
19. **"Team sharing: share a paper with collaborators."** → An ACL table (`paper_permissions(paper_id, user_id, role)`). The vector filter becomes `paper_id IN allowed_papers(user)`; or store `allowed_group_ids` in the payload and filter by group membership. Keep auth checks in the service layer.
20. **"Delete my account (GDPR)."** → Cascade: vectors (filter by user_id), objects, rows, caches, logs/traces containing content; async job + confirmation. RAG makes this possible (fine-tuned weights couldn't "forget").
21. **"The same paper uploaded as a different file (arXiv v1 vs v2)."** → Beyond SHA-256: DOI/arXiv ID match, normalized title match, MinHash near-duplicate detection on text; ask the user to version or replace.
22. **"Prompt injection inside an uploaded PDF."** → Treat context as untrusted data: delimiters, instructions to ignore commands in context, a classifier or regex scan at ingestion, no tools or side effects available to the model, output filtering (no secrets in the prompt anyway).
23. **"Need real-time ingestion from arXiv daily."** → A scheduled crawler (arXiv API/RSS) → queue → workers; dedupe by arXiv ID; incremental upserts; HNSW handles inserts online.
24. **"Answers should cite the exact sentence, not just the page."** → Store char offsets per chunk; ask the LLM for quote spans (structured output); fuzzy-match them in the chunk; highlight in a PDF viewer (pdf.js) using PyMuPDF `search_for` to get bboxes.
25. **"Agentic features: find related papers, fetch citations."** → A LangGraph/tool-calling agent with tools: `search_library`, `semantic_scholar_lookup`, `arxiv_search`, `get_metadata`; guard with step limits and tool permissions; evaluate trajectories.
26. **"Millions of papers (a global corpus, not per user)."** → Distributed vector DB (Milvus/Qdrant cluster/Vespa), IVF-PQ or HNSW + quantization, sharding, a metadata-first narrowing (year/field), a hybrid search engine (OpenSearch), paper-level retrieval first (SPECTER2), then chunk-level retrieval.
27. **"We need to measure if a change helped."** → Offline golden-set eval (Recall@k, MRR, RAGAS) + online A/B (feedback rate, abstention, latency) with tracing to compare variants.
28. **"Run fully offline on a laptop."** → Already close: local embeddings + Chroma + SQLite; swap Groq for Ollama (Llama 3.x 8B / Qwen 2.5 7B, 4-bit quantized GGUF); Supabase → local auth or none.
29. **"Mobile app: photo of a printed page, ask a question."** → Image upload endpoint → preprocess → OCR/VLM → temporary or persistent doc → same RAG; or skip indexing for a single page and send the image directly to a VLM.
30. **"The confidence score is misleading."** → Replace it with calibrated signals: reranker sigmoid score, faithfulness score, an "insufficient context" flag from the LLM; calibrate thresholds on the golden set (reliability diagram).
31. **"Explain how you'd A/B test chunk size in production."** → Two collections (chunking v1 and v2), route x% of traffic, log retrieval + feedback; but do offline eval first, because chunking changes require re-indexing.
32. **"Why not just use OpenAI Assistants / a managed RAG service?"** → Faster to start, but: data control, vendor lock-in, cost at scale, no control over chunking, filters or citations, and harder evaluation. My architecture keeps each piece swappable.

---

## Part 4: Project 2: OCR-based Intelligent Document-AI system

> ⚠️ This project's code is **not in this repository**, so this section is a **reference architecture** built around the typical design
> (OCR + Gemini). Replace the components below with what you actually used: engine names, fields, document types, numbers.
> Questions from the other candidate's interview: *How OCR works · Why not only a VLM · Why Gemini · How to replace hosted APIs.*

### 4.1 Typical architecture

```text
 Upload (PDF / JPG / PNG / TIFF, phone photos, scans)
   │  validate (magic bytes, size, page count), hash/dedupe, store in object storage
   ▼
 Pre-processing (OpenCV / Pillow)
   EXIF rotate → document edge detect + perspective warp → deskew (Hough / minAreaRect)
   → denoise (median / bilateral) → grayscale → binarize (Otsu / adaptive Sauvola)
   → contrast (CLAHE) → resize to ~300 DPI
   ▼
 Document classification (invoice / receipt / ID / form / contract…)
   rules, a small CNN/LayoutLM classifier, or the VLM
   ▼
 OCR: text detection + recognition → words + bounding boxes + confidence
   (Tesseract / PaddleOCR / docTR / EasyOCR / cloud OCR)
   ▼
 Layout analysis: blocks, reading order, tables, key-value pairs
   ▼
 Understanding / extraction with an LLM/VLM (Gemini)
   input = page image + OCR text (with coordinates)  →  JSON with a strict schema
   (response_schema / Pydantic), fields + confidence + source bbox
   ▼
 Validation & post-processing
   Pydantic types, regex (dates, GSTIN/PAN/IBAN), checksums, totals = sum(line items),
   cross-field rules, OCR-vs-LLM agreement check, confidence thresholds
   ▼
 Human-in-the-loop review UI for low-confidence fields  →  corrections → feedback dataset
   ▼
 Output: DB / API / export (JSON, CSV), optional search/RAG over documents
```

### 4.2 How OCR works (explain it from first principles)

1. **Image acquisition and pre-processing.** OCR accuracy depends heavily on image quality. Steps: grayscale, **binarization** (global Otsu picks the
   threshold that minimizes intra-class variance; **adaptive** thresholding/Sauvola handles uneven lighting), **deskew**, **denoise**, **dewarp** (curved
   pages), DPI normalization (~300 DPI; characters ~20–30 px tall).
2. **Text detection** (where is text?). Classical: connected components and MSER. Deep learning: **EAST** (rotated boxes), **CRAFT** (character region + affinity
   maps), **DBNet** (differentiable binarization; used in PaddleOCR), all outputting word or line polygons.
3. **Text recognition** (what does it say?). Crop each line and feed it to:
   - **CRNN + CTC:** CNN feature extractor → BiLSTM sequence model → **CTC loss/decoding**. CTC (Connectionist Temporal Classification) lets the
     model output a character or a *blank* per time step without character-level alignment; decoding collapses repeats and removes blanks
     ("hh-e-ll-ll-oo" → "hello"). Greedy or beam-search decoding, optionally with a language model.
   - **Attention / transformer encoder–decoder:** **TrOCR** (ViT encoder + text decoder), PARSeq, SVTR (Paddle).
   - **Tesseract 4/5:** an LSTM line recognizer, plus its own layout analysis (page segmentation modes `--psm`), with language packs.
4. **Post-processing:** dictionary or LM correction, regex normalization, and confidence per word/character.
5. **Layout and structure:** reading order (multi-column), tables (Table Transformer / TATR, PaddleOCR PP-Structure), key-value pairs, checkboxes.
   **Layout-aware models:** **LayoutLM / LayoutLMv3** (text + 2D position + image patches → token classification for KIE), **LiLT**,
   **Donut** (OCR-free: image → JSON directly, Swin encoder + BART decoder), **Pix2Struct**, **Nougat** (academic PDFs → Markdown/LaTeX).

**Metrics:** **CER** (character error rate) = (substitutions + deletions + insertions) / reference characters (Levenshtein); **WER** for words;
field-level **exact match / F1** for extraction; **ANLS** (DocVQA); **TEDS** for table structure.

**Engines compared**

| Engine | Strengths | Weaknesses |
|---|---|---|
| Tesseract | Free, CPU, 100+ languages, mature | Weak on photos, complex layouts, handwriting |
| PaddleOCR (PP-OCR, PP-Structure) | High accuracy, fast, tables and layout, multilingual | Paddle framework |
| EasyOCR | Simple API, PyTorch | Slower, less accurate on documents |
| docTR (Mindee) | Modern detection + recognition, TF/PyTorch | |
| Cloud: Google Document AI / Vision, AWS Textract, Azure Document Intelligence | Best-in-class, forms/tables/handwriting, managed | Cost, data leaves your infra, lock-in |
| VLM-based OCR: GOT-OCR2.0, olmOCR, Qwen-VL, Mistral OCR, Gemini | Layout and table aware, Markdown output | GPU/cost, hallucination risk |

### 4.3 "Why not only a VLM?" (very likely question)

A VLM (Gemini, GPT-4o-class, Claude, Qwen-VL) *can* read documents end-to-end, but on its own it has these problems:
1. **Hallucination of characters and numbers.** It may "autocorrect" an invoice number, swap digits, or invent a plausible value for a blurry field.
   It is fluent, so errors look correct. OCR errors are *visible* (garbled text) and come with **confidence scores**.
2. **No reliable bounding boxes or per-character confidence.** These are needed for highlighting, audit trails, human review and compliance.
3. **Determinism and reproducibility.** The same input can give different outputs; that's hard to regression-test.
4. **Cost and latency at volume.** OCR on CPU costs fractions of a cent per page; VLM calls per page add up at millions of pages.
5. **Long documents.** 100-page PDFs mean image tokens per page and context limits; OCR text is compact.
6. **Privacy and compliance.** Hosted VLMs send documents out (PII, KYC, medical).
7. **Validation:** with OCR text you can **cross-check** the VLM's extraction (is the extracted value actually present in the OCR text?).

**So I use a hybrid:** OCR gives text + boxes + confidence (ground truth for *what characters exist*); the LLM/VLM does *understanding*: classification,
key-value mapping, normalization, reasoning over layout. Then **cross-validation**: a value the LLM extracts must fuzzy-match OCR tokens; disagreement
or low confidence → human review. For clean digital PDFs, skip OCR and use the embedded text layer.

The counter-argument (show balance): modern VLMs beat classical OCR on messy layouts, handwriting and tables, and remove pipeline complexity.
For low volume or very heterogeneous documents, a VLM-first design with OCR as the verifier is reasonable.

### 4.4 "Why Gemini?"

- **Native multimodality:** trained on images and text together; strong document and chart understanding; accepts **PDFs directly** (handles pages as images + text).
- **Long context** (1M tokens for many models): multi-page documents in one call.
- **Structured output:** `response_mime_type="application/json"` + `response_schema` (from a Pydantic model), so it returns valid JSON matching the schema. That's essential for extraction.
- **Price/performance:** Flash / Flash-Lite tiers are cheap and fast for high-volume extraction; Pro for hard documents (route by difficulty).
- Free tier and an easy SDK for prototyping; enterprise path via Vertex AI (data governance, regional endpoints, VPC-SC).
- Good multilingual OCR ability (including Indic scripts, if relevant).
- **Trade-offs to acknowledge:** vendor lock-in, data leaves your infra (mitigate with Vertex AI terms / regions), rate limits, model version changes (pin versions, re-run evals).
- **Alternatives considered:** GPT-4o-class / GPT-4.1 / newer OpenAI models, Claude (strong document reasoning), Mistral OCR, Azure Document Intelligence (template/custom models), open models.

### 4.5 "How would you replace hosted APIs?" (self-hosting)

1. **Abstraction first:** a provider interface, e.g. `class ExtractionModel(Protocol): def extract(images, ocr_text, schema) -> dict`, with `GeminiExtractor`,
   `VLLMExtractor` and `OllamaExtractor` implementations. (Same pattern as `DocumentEmbeddingModel` / `LLMService` in Project 1.) Config selects the provider.
2. **Open-source replacements:**
   - OCR: PaddleOCR / docTR / Tesseract (CPU); GOT-OCR2.0, olmOCR, Nougat, Marker, **Docling** (IBM; layout + tables + OCR), **MinerU**.
   - VLM for extraction: **Qwen2.5-VL / Qwen3-VL** (very strong at documents), InternVL, Llama 3.2 Vision, Pixtral, MiniCPM-V, Florence-2 (small; OCR/grounding), PaliGemma.
   - OCR-free KIE: **Donut** or LayoutLMv3 **fine-tuned** on your document types (cheap at inference, very accurate on fixed templates).
   - Text LLM (if OCR → text → LLM): Llama 3.x, Qwen 2.5/3, Mistral, gpt-oss.
3. **Serving:** **vLLM** (PagedAttention, continuous batching, OpenAI-compatible API, so client code barely changes), TGI, SGLang, Ollama/llama.cpp for small scale;
   Triton for OCR models. Use **structured/guided decoding** (vLLM guided JSON via outlines/xgrammar) to replace Gemini's `response_schema`.
4. **Optimization:** quantization (AWQ/GPTQ 4-bit, FP8), batching, image-resolution limits, a KV-cache budget; GPU sizing (a 7B VLM fits on one 24 GB GPU in fp16 with
   limited context, or more comfortably 4-bit).
5. **Quality parity:** run the same eval set (field-level F1, CER, exact match) on hosted vs self-hosted; fine-tune (LoRA) the open VLM on your corrected
   human-review data to close the gap.
6. **Ops:** autoscaling GPU workers on queue depth, model registry and versioning, monitoring (latency, GPU utilization, error rates), fallback to the hosted API during an incident if policy allows.
7. **Trade-off summary:** self-hosting gives data control, predictable cost at high volume and no rate limits; it costs GPU ops, MLOps effort, and possibly lower quality without fine-tuning.

### 4.6 Other likely Project-2 questions

- **Handwriting?** → Cloud OCR / TrOCR-handwritten / VLMs do better than Tesseract; flag it for human review.
- **Tables?** → Detect the table region (TATR / PP-Structure), recognize structure (rows/cols/spans), output HTML/Markdown; evaluate with TEDS; or ask the VLM for a table JSON plus a totals check.
- **Multi-page invoices / line items spanning pages?** → Page-level extraction + a merge step (dedupe headers, concatenate line items); or a long-context VLM with the whole document.
- **Confidence for LLM outputs?** → Combine OCR confidence of the matched tokens, VLM logprobs (if available), self-consistency across 2 runs, and validation rules into a field score; threshold → review.
- **PII?** → Detect (Presidio / regex / NER), mask before sending to a hosted API, encrypt at rest, retention policy.
- **Throughput?** → Async queue, page-level parallelism, OCR on CPU pools, batched VLM calls, caching by document hash.
- **Evaluation dataset?** → Labeled documents per type (fields + bboxes); public: FUNSD (forms), CORD/SROIE (receipts), DocVQA, RVL-CDIP (classification), PubTabNet (tables).
- **Connecting it to RAG (Project 1)?** → OCR output becomes text chunks with page + bbox metadata → same embedding/indexing pipeline, so users can chat with scanned documents.

---

## Part 5: Rapid-fire Q&A and interview strategy

### 5.1 Rapid fire (short, confident answers)

**RAG and retrieval**
- *What is top-k and how did you choose 5?* The number of chunks sent to the LLM. 5 × ~280 tokens covers most single-fact questions within a small budget; tune it with Recall@k vs faithfulness.
- *What happens if a question needs info from 10 chunks?* Top-5 may miss some. Use a larger k + reranker, query decomposition, or a parent-document (bigger context) strategy.
- *Why overlap?* So a sentence on a boundary appears whole in at least one chunk. The cost is ~20% more chunks.
- *Why normalize embeddings?* So cosine = dot product, scores land in a stable range, and ANN metrics agree.
- *Chroma distance vs similarity?* Chroma returns cosine **distance** (1 − cos); I convert with `1 − distance`, clamped to [0, 1].
- *What is HNSW in one line?* A multi-layer small-world graph you search greedily from coarse to fine, giving roughly logarithmic search time.
- *ANN recall?* The fraction of true nearest neighbours that ANN returns; tuned with efSearch / nprobe.
- *Why not FAISS here?* FAISS is a library without persistence management or metadata filtering. I'd have to build tenant filtering and storage myself. Chroma gives both out of the box.
- *Embedding dimension trade-off?* Higher dimension can capture more nuance but costs memory and latency; Matryoshka/PCA can reduce it.
- *What is the context window and token budget?* System prompt + metadata + 5 chunks + question ≈ 2k tokens; output capped at 1024.
- *Why temperature 0.2, not 0?* Near-deterministic factual answers with slightly more natural phrasing; 0 is also fine.
- *Lost in the middle?* LLMs use info at the start and end of the context better. Put the best chunks first (reranked order), keep the context short.
- *Bi-encoder vs cross-encoder in one line?* Bi-encoder: separate encodings, fast, for recall. Cross-encoder: joint encoding, accurate, for precision on a shortlist.
- *Sparse vs dense in one line?* Sparse matches words exactly (BM25, inverted index); dense matches meaning (embeddings, ANN). Hybrid gets both.
- *RRF formula?* Σ 1/(60 + rank).
- *What is MMR?* Re-selection that balances relevance against diversity, to avoid near-duplicate chunks.
- *What is HyDE?* Embed an LLM-generated hypothetical answer instead of the raw question.
- *What is GraphRAG?* Build a knowledge graph plus community summaries for global, multi-document questions.
- *What is agentic RAG?* An LLM controls retrieval (when, what, how many times) with tools and loops.
- *What is a semantic cache?* Reuse answers for semantically equivalent questions (cosine above a threshold), scoped by tenant.

**Engineering**
- *Why FastAPI `Depends`?* DI for auth and services; easy to override in tests (`app.dependency_overrides`).
- *Why `@lru_cache` on providers?* Load the 420 MB model, Chroma client and Groq client once per process.
- *Sync vs async routes?* Routes doing CPU/blocking work (`process`, `index`, `chat`) are plain `def`, so FastAPI runs them in a threadpool and they don't block the event loop. `upload` is `async` for `await file.read()`; auth is async (httpx).
- *Why UUIDs?* Globally unique, no enumeration, generated without a DB round-trip.
- *Why SHA-256?* Collision-resistant content fingerprint for exact-duplicate detection per user.
- *How is tenant isolation tested?* Storage tests (SQLite: owner can't access another user's paper; Chroma: search returns only the owner's vectors) and service tests (same PDF allowed for two users, duplicate rejected for one). Next step: an API test where user B gets 404 on user A's paper.
- *Why store chunks in SQLite if Chroma stores documents too?* SQLite is the source of truth for re-embedding, migrations and display; the vector store is a derived index.
- *What happens on delete?* Vectors (filtered by paper+user) → file → DB row (chunks cascade). Not atomic; a reconciliation job would fix orphans.
- *How does the frontend know progress?* It chains upload → process → index mutations and updates a progress label after each step.
- *Error handling?* Domain exceptions are mapped in routes: `PDFProcessingError` → 422, `DuplicatePaperError` → 409, `ValueError` → 400. A global handler turns anything else (including `LLMError`) into a generic 500 "An unexpected error occurred." so internals never leak. Improvement: map `LLMError` to 502/503 with a retry hint.

**LLM choice**
- *Why Groq?* Very fast inference (LPU hardware, high tokens/s), OpenAI-compatible API, low cost, open-weight models available.
- *Why gpt-oss-20b?* An open-weight MoE (≈21B total, ≈3.6B active params): good reasoning for its size, cheap and fast, and **self-hostable later**, so there's no lock-in to the model.
- *How to swap models?* Change `GROQ_MODEL` in env; for another provider, implement the same `generate(system, user)` adapter.

### 5.2 Questions you might get on the "why" of your decisions

| They ask | You say |
|---|---|
| Why build a layered architecture for a small app? | Each layer is testable and replaceable. The V2 migrations (Postgres, Qdrant, queue) touch adapters, not business logic. |
| Why not do everything in one `/upload` call? | Independent failure and retry per stage, explicit status machine, and a direct path to background jobs. |
| Why Supabase? | Managed, secure auth (OAuth, email, sessions, refresh tokens) without me storing passwords; the backend just verifies identity. |
| Why page-level citations instead of chunk text? | Users verify in the original PDF by page; it's also impossible for the LLM to fabricate them because code builds them. |
| Why metadata context in chat? | Author/title/year questions fail with pure semantic retrieval; trusted extracted metadata answers them precisely. |
| What would you do differently? | Start with an evaluation set on day 1, token-based structure-aware chunking, hybrid + reranker, background jobs, and local JWT verification. |
| What was the hardest part? | Metadata extraction from heterogeneous paper layouts (font-size heuristics, affiliations mixed with names, footnote markers); and keeping ownership consistent across three stores. |
| A bug you fixed? | Adjust to your real experience. Candidates from the git history: ownership enforcement across services, duplicate uploads (added hash + unique index), legacy rows without `user_id` (schema migration hides them), year extraction picking citation years (prefer the arXiv stamp). |

### 5.3 Strategy for a colleague-style interview

1. **Think aloud with trade-offs.** "Option A gives X but costs Y; for my constraints (single node, MVP, privacy) I chose A; at scale I'd switch to B."
2. **Always tie the answer to your code:** "In my `RetrievalService`, I'd add…". It proves you built it.
3. **Separate retrieval from generation** in every debugging answer: "First I'd check whether the right chunk was retrieved (Recall@k), then whether the LLM used it faithfully."
4. **Quantify:** dims (768), chunk size (200 words, 40 overlap), top-k (5), temperature (0.2), latency budget, storage maths.
5. **Be honest about scope:** "V1 does X; Y is designed but not shipped." Then show you know exactly how to build Y.
6. **For any "what if" question,** use: requirement → what breaks today → change (component + tech) → trade-off → how to measure.
7. **Ask clarifying questions in scenarios:** scale? latency SLO? data sensitivity? budget? That's what a colleague does.

### 5.4 Good questions to ask the interviewer

- How do you evaluate your RAG/LLM systems today: offline golden sets, online feedback, or both?
- Do you self-host models or use hosted APIs, and what drove that decision?
- What does your document ingestion look like: mostly digital PDFs or scans?
- How do you handle multi-tenancy and access control in retrieval?

### 5.5 One-page cheat sheet

```text
PROJECT 1: AI Research Copilot (full-stack, multi-tenant RAG)
React19+TS+Vite+Tailwind+TanStack Query | Supabase Auth | FastAPI+Pydantic | PyMuPDF
Chunk: page-aware, 200 words, 40 overlap (20%), never crosses pages
Embed: all-mpnet-base-v2, 768-d, max 384 tokens, normalized, encode_query/encode_document
Store: ChromaDB PersistentClient, 1 collection, HNSW cosine, metadata {user_id,paper_id,page,chunk_index}
       SQLite: papers + chunks, UNIQUE(user_id,content_hash), FK cascade, foreign_keys=ON
Search: where user_id (+paper_id), top_k=5 (1..20), score = 1 - cosine distance
LLM: Groq openai/gpt-oss-20b, temp 0.2, max_tokens 1024, "answer only from context"
Citations: built by code from retrieval → paper title + page (can't be hallucinated)
Confidence: max cosine (uncalibrated; hidden in UI) → would use reranker/faithfulness
Security: owner_id on every query/filter/delete; 404 for others' resources; UUID filenames
Tests: 63 pytest (models/services/storage/api), DI + Protocol fakes
Deploy: Docker python:3.13-slim + named volume; Vercel SPA
NOT YET: LangChain, hybrid BM25, reranker, RAGAS eval, memory, streaming, OCR, queue
V2: S3 + Postgres + Qdrant/pgvector + Redis + workers + hybrid/RRF + cross-encoder
    + RAGAS CI gate + streaming SSE + local JWKS JWT + observability

PROJECT 2: OCR Document-AI
preprocess (deskew/binarize/denoise) → detect (DBNet/CRAFT) → recognize (CRNN+CTC/TrOCR)
→ layout (LayoutLMv3/TATR) → Gemini JSON-schema extraction → validate → human review
Not VLM-only: hallucinated digits, no bboxes/confidence, cost, privacy, determinism
Gemini: native multimodal, PDFs, long context, response_schema JSON, cheap Flash
Self-host: Protocol adapter → PaddleOCR/Docling + Qwen-VL on vLLM (guided JSON), AWQ, eval parity
Metrics: CER/WER, field F1/exact match, ANLS, TEDS
```

---

*Generated from the code in this repository (backend/app, frontend/src, docker and config files). Part 4 is a reference design,
because Project 2's code isn't in this repo. Replace it with your actual implementation details.*
