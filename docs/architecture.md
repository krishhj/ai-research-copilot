# AI Research Copilot Architecture

## 1. Project Overview
AI Research Copilot is a research paper analysis platform that lets users upload papers, extract structured knowledge, search semantically, ask questions using RAG, and explore paper-level analytics such as recommendations, topics, citations, and research gaps.

## 2. Goals
- Build a production-style AI backend from first principles
- Separate ingestion, retrieval, generation, and analytics
- Keep the system modular, testable, and easy to explain
- Support a modern frontend through a clean API layer

## 3. High-Level System Design
The system is organized into these layers:

- Frontend: React UI for uploading papers, searching, chatting, and exploring analytics
- API Layer: FastAPI routes and request/response handling
- Service Layer: Business logic for ingestion, retrieval, QA, recommendations, and analytics
- Storage Layer: SQLite for metadata and ChromaDB for vector storage
- Core Layer: Configuration, constants, logging, and custom exceptions

## 4. Core Workflows

### 4.1 Ingestion Workflow
PDF Upload -> PDF Processing -> Metadata Extraction -> Chunking -> Embedding -> Storage

### 4.2 Retrieval Workflow
User Query -> Query Embedding -> Dense Search + BM25 -> Fusion -> Reranking -> Context Building -> LLM Answer

### 4.3 Analytics Workflow
Stored Papers -> Topic Modeling / Citation Graph / Gap Analysis / Recommendations -> UI Visualization

## 5. Responsibilities by Module
- `core/`: shared infrastructure
- `models/`: domain data structures
- `services/`: application logic
- `storage/`: persistence and retrieval backends
- `api/`: HTTP interface
- `docs/`: architecture and decisions
- `tests/`: unit and integration tests

## 6. Design Principles
- Single Responsibility Principle
- Separation of Concerns
- Composition over large monolithic classes
- Centralized configuration
- Explicit error handling
- Test-driven development for critical models and services

## 7. Future Enhancements
- Authentication and user accounts
- Background job processing
- Streaming responses
- Better evaluation dashboards
- Docker + CI/CD
- Multi-user collaboration