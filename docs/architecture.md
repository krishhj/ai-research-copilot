# AI Research Copilot Architecture

## Project overview

AI Research Copilot lets authenticated users upload research PDFs, search their papers semantically, and ask questions grounded in retrieved passages. Answers include citations to the source paper and page.

## Implemented architecture

- **Frontend:** React and TypeScript application for authentication, paper uploads, library browsing, and chat.
- **API:** FastAPI routes and request/response schemas.
- **Authentication:** Supabase access tokens are validated by the backend. The authenticated user ID is passed to paper and retrieval operations.
- **Document processing:** PDFs are parsed page by page, cleaned, and split into overlapping, page-aware chunks. Metadata and chunks are stored in SQLite.
- **Embeddings and retrieval:** Sentence Transformers creates normalized document and query embeddings. ChromaDB stores vectors and performs cosine-distance search. Queries are filtered by user ID and, when selected, paper ID.
- **Reranking:** The top 20 vector results are rescored with the `cross-encoder/ms-marco-MiniLM-L6-v2` cross-encoder, and the best 5 are passed to the LLM. Reranking can be turned off with `RERANK_ENABLED=false`.
- **Answer generation:** Retrieved passages and selected-paper metadata are sent to the configured Groq model with instructions to answer from that context and say when information is insufficient.
- **Citations:** Retrieved chunk metadata is used to return paper and page citations.
- **Testing and local deployment:** Pytest covers services, storage, processing, and ownership behavior. Docker Compose supports local deployment.

## Core workflows

### Ingestion

PDF upload -> SHA-256 duplicate check -> page-level text extraction -> text cleaning -> page-aware overlapping chunks -> metadata and file storage -> embedding generation -> ChromaDB indexing

### Question answering

Authenticated question -> query embedding -> user-scoped dense vector search (20 candidates) -> cross-encoder reranking (top 5) -> retrieved context and paper metadata -> Groq answer generation -> answer with citations and relevance-score signal

## Retrieval scope

Retrieval runs in two stages: dense semantic vector search, then cross-encoder reranking. It does **not** currently implement BM25 keyword search, hybrid score fusion, query rewriting, or multi-turn conversational memory. The answer score is the best reranker score (or the best ChromaDB score when reranking is off); it is a relevance heuristic, not a calibrated probability that an answer is correct.

## Retrieval evaluation

`backend/evaluation/` holds 25 questions about the original RAG paper (Lewis et al., 2020), each labelled with the page that answers it. A result counts as relevant when it comes from a labelled page.

| Setup | Recall@1 | Recall@3 | Recall@5 | MRR | Latency per query |
|---|---|---|---|---|---|
| Dense only | 0.64 | 0.92 | 0.92 | 0.77 | ~0.1 s |
| Dense + rerank | 0.88 | 0.96 | 0.96 | 0.92 | ~2.1 s |

Run it from `backend/` with `python -m evaluation.evaluate_retrieval data/papers/<paper>.pdf`.

## Storage and privacy

- SQLite stores paper metadata and extracted chunks.
- ChromaDB stores embeddings, chunk text, and ownership metadata.
- Uploaded PDFs are stored locally in the current deployment.
- Backend authentication and owner filters constrain paper operations and vector searches to the authenticated user.

## Potential future improvements

- Add hybrid lexical and dense retrieval, then evaluate whether it improves retrieval quality.
- Extend the evaluation set beyond one paper and add checks for citation correctness, groundedness, and abstention.
- Move PDFs and persistent stores to production-ready managed storage.
- Add background processing for large documents, observability, rate limits, and deployment automation.
- Calibrate or rename the retrieval score so users do not mistake it for answer confidence.
