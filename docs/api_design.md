# API Design

This document describes the **current implemented HTTP API** in this repository.

## Base URL and Versioning

- API router base path: `/api/v1`
- Health endpoint: `/api/v1/health`
- Root ping endpoint (outside API router): `/`

## Authentication

Most API endpoints require a Supabase access token.

- Header: `Authorization` with an access token value
- Enforced by `get_current_user` in `backend/app/core/auth.py`
- Authenticated endpoints are user-scoped (papers, vectors, and retrieval are isolated per user)

Auth failures:
- `401` when token is missing/invalid/expired
- `503` when Supabase user lookup is unavailable

## Implemented Response Conventions

There is no global `success/data/error` envelope today. Endpoints return route-specific JSON objects.

- Successful responses return JSON matching each route's Pydantic response model.
- Errors generally return:

```json
{ "detail": "..." }
```

- FastAPI validation errors return `422` with `detail` as a list of validation issues.
- Unhandled exceptions are normalized to:

```json
{ "detail": "An unexpected error occurred." }
```

## Core Models (as implemented)

### Paper

```json
{
  "id": "7e8f5c4d-1111-2222-3333-444455556666",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani"],
    "abstract": "...",
    "year": 2017,
    "doi": null,
    "journal": null,
    "keywords": []
  },
  "processing": {
    "stored_filename": "7e8f5c4d-1111-2222-3333-444455556666.pdf",
    "total_pages": 15,
    "total_chunks": 42,
    "uploaded_at": "2026-10-22T17:00:00+00:00",
    "status": "processed"
  }
}
```

`processing.status` enum values:
- `uploaded`
- `processing`
- `processed`
- `failed`

### Query body (used by `/search` and `/chat`)

```json
{
  "question": "Summarize the core contribution.",
  "top_k": 5,
  "paper_id": "7e8f5c4d-1111-2222-3333-444455556666",
  "conversation_id": "11111111-2222-3333-4444-555555555555"
}
```

Validation:
- `question`: required, min length 1
- `top_k`: optional, default `5`, range `1..20`
- `paper_id`: optional UUID
- `conversation_id`: optional UUID (currently accepted but not used for stateful chat storage)

## Implemented Endpoints

### GET `/`
Simple service ping.

Response `200`:

```json
{ "message": "AI Research Copilot API is running" }
```

---

### GET `/api/v1/health`
Health check.

Response `200`:

```json
{ "status": "healthy", "service": "AI Research Copilot" }
```

---

### GET `/api/v1/auth/me` (authenticated)
Returns current user identity derived from Supabase token.

Response `200`:

```json
{
  "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
  "email": "user@example.com"
}
```

---

### POST `/api/v1/papers` (authenticated)
Upload one PDF.

Request:
- `multipart/form-data`
- file field name: `file`

Response `201`:

```json
{
  "message": "Paper uploaded successfully",
  "paper": { "id": "...", "metadata": { "title": null, "authors": [], "abstract": null, "year": null, "doi": null, "journal": null, "keywords": [] }, "processing": { "stored_filename": "....pdf", "total_pages": 0, "total_chunks": 0, "uploaded_at": "2026-10-22T17:00:00+00:00", "status": "uploaded" } }
}
```

Possible errors:
- `400` missing filename or non-PDF upload
- `409` duplicate upload for same authenticated user (same content hash)
- `401`, `503`, `422`, `500`

---

### GET `/api/v1/papers` (authenticated)
List current user's papers, ordered by `uploaded_at DESC`.

Response `200`:

```json
{ "papers": [/* Paper[] */] }
```

Notes:
- No pagination implemented
- No filtering query params implemented

---

### GET `/api/v1/papers/{paper_id}` (authenticated)
Fetch one paper by UUID for current user.

Response `200`:

```json
{ "paper": { /* Paper */ } }
```

Possible errors:
- `404` paper not found for current user
- `401`, `503`, `422`, `500`

---

### DELETE `/api/v1/papers/{paper_id}` (authenticated)
Delete one paper for current user.

Response `200`:

```json
{ "message": "Paper deleted successfully" }
```

Possible errors:
- `404` paper not found for current user
- `401`, `503`, `422`, `500`

---

### POST `/api/v1/papers/{paper_id}/process` (authenticated)
Process a PDF into cleaned chunks + extracted metadata and store result in SQLite.

Response `200`:

```json
{ "paper": { /* Paper with updated totals/status */ } }
```

Processing/status behavior:
- Status transitions to `processing` before extraction
- On success, saved metadata/chunks set status to `processed`
- On PDF parsing failure, status is set to `failed` and route returns `422`

Possible errors:
- `404` paper not found for current user
- `422` uploaded file could not be processed as PDF
- `401`, `503`, `500`

---

### POST `/api/v1/papers/{paper_id}/index` (authenticated)
Generate embeddings for processed chunks and upsert into ChromaDB.

Response `200`:

```json
{
  "message": "Paper indexed successfully.",
  "indexed_chunks": 42
}
```

Possible errors:
- `404` paper not found for current user
- `409` paper is not in `processed` state
- `401`, `503`, `500`

---

### POST `/api/v1/search` (authenticated)
Semantic vector search over the authenticated user's indexed chunks.

Request body: `Query`

Response `200`:

```json
{
  "results": [
    {
      "chunk_id": "...",
      "paper_id": "...",
      "page_number": 3,
      "text": "retrieved chunk text",
      "score": 0.91
    }
  ]
}
```

Possible errors:
- `401`, `503`, `422`, `500`

---

### POST `/api/v1/chat` (authenticated)
RAG question answering using retrieved chunks (and optionally selected paper metadata).

Request body: `Query`

Response `200`:

```json
{
  "answer_text": "...",
  "citations": [
    {
      "paper_id": "...",
      "paper_title": "...",
      "chunk_id": "...",
      "page_number": 2
    }
  ],
  "retrieved_chunks": [
    { "chunk_id": "...", "score": 0.88 }
  ],
  "confidence": 0.88,
  "generation_time": 1.42
}
```

Possible errors:
- `401`, `503`, `422`, `500`

## Frontend Usage Notes (current project)

`frontend/src/api/client.ts` calls these implemented paths:
- `GET /papers`
- `POST /papers`
- `POST /papers/{paper_id}/process`
- `POST /papers/{paper_id}/index`
- `DELETE /papers/{paper_id}`
- `POST /chat`

`/search` and `/auth/me` exist on backend but are not currently used by this client module.

## Not Yet Implemented (Planned/Future)

The following are discussed in project docs/README themes but are **not implemented as API routes in the current router**:

- Recommendations endpoints (for example `/api/v1/recommendations/{paper_id}`)
- Analytics endpoints (for example topic/citation/gap analytics routes)
- Server-side pagination/filtering parameters for paper listing
- Streaming chat endpoint (SSE/WebSocket)

Keep this document aligned with `backend/app/api/router.py` and route modules whenever API behavior changes.
