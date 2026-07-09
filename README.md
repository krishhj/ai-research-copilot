# AI Research Copilot

> A production-style AI-powered research assistant that enables semantic search, Retrieval-Augmented Generation (RAG), research analytics, and paper recommendations from scientific literature.

---

## Overview

AI Research Copilot is an end-to-end AI application designed to help researchers, students, and engineers efficiently interact with research papers.

Users can upload papers, perform semantic search, ask questions using Retrieval-Augmented Generation (RAG), receive source-grounded answers, discover related papers, analyze research topics, and identify potential research gaps.

Unlike many tutorial-based projects, this application focuses on implementing the complete AI pipeline with professional software engineering practices rather than relying heavily on high-level frameworks.

---

# Objectives

- Build a production-style AI backend
- Implement Retrieval-Augmented Generation (RAG) from first principles
- Learn modern Information Retrieval techniques
- Practice scalable software architecture
- Develop a modern React frontend
- Apply software engineering best practices
- Understand AI system design end-to-end

---

# Features

## Document Processing

- Upload research papers (PDF)
- Metadata extraction
- Text extraction
- Intelligent text chunking
- Persistent document storage

---

## Semantic Search

- Dense Vector Search
- BM25 Keyword Search
- Hybrid Retrieval
- Cross-Encoder Reranking
- Source-aware retrieval

---

## AI Assistant

- Chat with research papers
- Context-aware answers
- Source citations
- Conversation memory
- Paper summarization

---

## Research Analytics

- Topic Modeling (BERTopic)
- Citation Network Analysis
- Research Gap Analysis
- Similar Paper Recommendations
- Research Trend Visualization

---

## Evaluation

- Retrieval quality evaluation
- RAG evaluation
- Latency benchmarking
- Embedding quality experiments

---

# Tech Stack

## Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- React Query
- Axios

---

## Backend

- FastAPI
- Uvicorn
- Pydantic v2
- SQLAlchemy
- SQLite

---

## AI / Machine Learning

- Sentence Transformers
- ChromaDB
- Rank-BM25
- Cross Encoder Reranker
- BERTopic
- NetworkX
- PyVis

---

## Large Language Model

- Groq API
- Llama 3.x

---

## Development

- Git
- Pytest
- Ruff
- Black
- isort
- mypy

---

# Project Architecture

```
React Frontend
        │
        ▼
FastAPI Backend
        │
──────────────────────────────────────
│              API Layer             │
──────────────────────────────────────
│           Service Layer            │
──────────────────────────────────────
│           Storage Layer            │
──────────────────────────────────────
│        SQLite + ChromaDB           │
──────────────────────────────────────
│              Groq LLM              │
```

---

# Backend Structure

```
backend/
│
├── app/
│   ├── core/
│   ├── models/
│   ├── api/
│   ├── services/
│   ├── storage/
│   ├── prompts/
│   └── utils/
│
├── tests/
├── docs/
├── logs/
└── data/
```

---

# RAG Pipeline

```
PDF Upload
      │
      ▼
PDF Processing
      │
      ▼
Metadata Extraction
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
      │
      ▼
ChromaDB

────────────────────────────

User Query
      │
      ▼
Query Embedding
      │
      ▼
Dense Search

+

BM25 Search
      │
      ▼
Hybrid Retrieval
      │
      ▼
Cross Encoder Reranker
      │
      ▼
Context Builder
      │
      ▼
Prompt Builder
      │
      ▼
Groq LLM
      │
      ▼
Answer + Citations
```

---

# Engineering Principles

This project follows several software engineering principles:

- Single Responsibility Principle (SRP)
- Separation of Concerns
- Composition over Inheritance
- Don't Repeat Yourself (DRY)
- Centralized Configuration
- Explicit Error Handling
- Type Safety
- Test-Driven Development (TDD mindset)
- Layered Architecture
- Modular Design

---

# Technical Decisions

### Why FastAPI?

- Excellent performance
- Native Pydantic support
- Automatic API documentation
- Strong typing

---

### Why React?

- Modern production frontend
- Better user experience
- Clean frontend/backend separation

---

### Why SQLite?

- Lightweight
- Serverless
- Perfect for project-scale metadata storage

---

### Why ChromaDB?

- Simple persistent vector database
- Easy integration
- Great developer experience

---

### Why Build RAG Without LangChain?

The objective is to understand how modern RAG systems actually work.

Instead of hiding complexity behind frameworks, this project implements:

- Chunking
- Embeddings
- Retrieval
- Prompt construction
- Context building
- Conversation memory

from first principles.

---

### Why Not LlamaIndex?

LlamaIndex abstracts document indexing and retrieval.

Since one goal of this project is to understand and implement these components ourselves, a custom implementation is preferred.

---

# Current Progress

## Sprint 1 — Core Infrastructure

- [x] Project Architecture
- [x] Configuration System
- [x] Logging
- [x] Constants
- [x] Custom Exceptions

---

## Sprint 2 — Domain Models

- [x] PaperStatus Enum
- [x] Paper Model
- [x] Paper Unit Tests
- [ ] Chunk Model
- [ ] Query Model
- [ ] Answer Model
- [ ] Conversation Model

---

## Upcoming

- FastAPI
- SQLite
- PDF Processing
- Chunking
- Embeddings
- ChromaDB
- Hybrid Retrieval
- Reranking
- RAG
- Analytics
- React Frontend
- Docker
- Deployment

---

# Learning Outcomes

This project is designed to strengthen skills in:

- AI Engineering
- Retrieval-Augmented Generation
- Information Retrieval
- Software Architecture
- FastAPI
- React
- Vector Databases
- Prompt Engineering
- Testing
- MLOps Fundamentals

---

# Project Status

🚧 **Actively Under Development**

The project is being built incrementally with a focus on clean architecture, maintainability, and production-ready engineering practices.

---

# License

This project is licensed under the MIT License.