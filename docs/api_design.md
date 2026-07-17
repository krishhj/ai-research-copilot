# API Design

This document defines the backend HTTP contract for the AI Research Copilot.

## API Principles
- Clear request and response models
- Consistent error format
- Predictable status codes
- Versioned endpoints
- Frontend-friendly responses

## Base Path
/api/v1

## Standard Response Shape
All endpoints should return structured JSON.

### Success Example
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully"
}

### Error Example
{
  "success": false,
  "error": {
    "type": "PDFProcessingError",
    "message": "Failed to extract text from PDF"
  }
}

---

## Planned Endpoints

### 1. Upload Paper
POST /api/v1/papers/upload

Purpose:
- Upload a PDF
- Extract metadata
- Store paper
- Trigger processing

Request:
- multipart/form-data

Response:
- paper_id
- status
- filename

### 2. List Papers
GET /api/v1/papers

Purpose:
- Return all uploaded papers
- Support filtering and pagination later

### 3. Get Paper Details
GET /api/v1/papers/{paper_id}

Purpose:
- Return metadata, status, and processing details

### 4. Search Papers
POST /api/v1/search

Purpose:
- Run hybrid retrieval against the paper corpus

Request:
{
  "query": "What are the main contributions of transformers?",
  "top_k": 5
}

### 5. Ask Question
POST /api/v1/chat

Purpose:
- Run RAG over the corpus and return an answer with citations

Request:
{
  "question": "Summarize the paper's method",
  "conversation_id": "optional-id"
}

### 6. Recommendations
GET /api/v1/recommendations/{paper_id}

Purpose:
- Return similar papers

### 7. Analytics
GET /api/v1/analytics/topics
GET /api/v1/analytics/citations
GET /api/v1/analytics/gaps

## Error Status Codes
- 400: Invalid request
- 404: Not found
- 422: Validation error
- 500: Unexpected server error

## Notes
- Keep request/response models in sync with frontend needs
- Update this document whenever endpoints change