<div align="center">

# AI Research Copilot

### AI-powered research assistant for semantic search, Retrieval-Augmented Generation (RAG), and research analytics.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Under%20Development-orange)

</div>

---

## Overview

AI Research Copilot is a full-stack AI application for exploring scientific literature through semantic search and Retrieval-Augmented Generation (RAG).

The system processes research papers into searchable knowledge, retrieves the most relevant information using hybrid search techniques, and generates source-grounded responses powered by Large Language Models.

The project is designed with a strong emphasis on modular architecture, maintainability, and production-oriented software engineering practices.

---

## Features

### Document Processing

- PDF ingestion
- Metadata extraction
- Intelligent document chunking
- Persistent document storage

### Retrieval

- Semantic vector search
- BM25 lexical search
- Hybrid retrieval
- Cross-encoder reranking

### AI Assistant

- Retrieval-Augmented Generation (RAG)
- Context-aware question answering
- Source citations
- Multi-turn conversations

### Research Analytics

- Paper recommendations
- Topic modeling
- Citation network visualization
- Research gap analysis

---

# System Architecture

```text
                           React Frontend
                                  │
                                  ▼
                           FastAPI Backend
                                  │
 ┌──────────────────────────────────────────────────┐
 │                   API Layer                      │
 ├──────────────────────────────────────────────────┤
 │                 Service Layer                    │
 ├──────────────────────────────────────────────────┤
 │              Retrieval Pipeline                  │
 ├──────────────────────────────────────────────────┤
 │      SQLite Metadata + ChromaDB Vector Store     │
 ├──────────────────────────────────────────────────┤
 │            Large Language Model (Groq)           │
 └──────────────────────────────────────────────────┘
```

---

# Retrieval Pipeline

```text
Research Paper (PDF)
        │
        ▼
Text Extraction
        │
        ▼
Metadata Extraction
        │
        ▼
Document Chunking
        │
        ▼
Embedding Generation
        │
        ▼
Vector Database
──────────────────────────────────────────────
        │
User Question
        │
        ▼
Query Embedding
        │
        ▼
Hybrid Retrieval
(Dense + BM25)
        │
        ▼
Cross-Encoder Reranking
        │
        ▼
Context Construction
        │
        ▼
Large Language Model
        │
        ▼
Answer with Citations
```

---

# Technology Stack

## Backend

- FastAPI
- Uvicorn
- Pydantic v2

## Artificial Intelligence

- Sentence Transformers
- ChromaDB
- Rank-BM25
- Cross Encoder
- BERTopic
- NetworkX

## Large Language Models

- Groq API
- Llama 3

## Database

- SQLite
- ChromaDB

## Frontend

- React
- TypeScript
- Tailwind CSS
- Vite

## Development

- Pytest
- Ruff
- Black
- isort
- mypy
- Git

---

# Project Structure

```text
backend/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── services/
│   ├── storage/
│   ├── prompts/
│   └── utils/
│
├── docs/
├── tests/
├── logs/
├── pyproject.toml
└── requirements.txt

frontend/
│
├── src/
├── public/
└── package.json
```

---

# Software Architecture

The project follows a layered architecture with clearly separated responsibilities.

- Domain-Driven Design (DDD)
- Separation of Concerns
- Single Responsibility Principle
- Modular Service Architecture
- Type-Safe Data Models
- Dependency Injection
- Structured Logging
- Centralized Configuration
- Unit Testing
- Clean API Design

---

# Technical Decisions

### FastAPI

FastAPI provides high performance, automatic OpenAPI documentation, native Pydantic integration, and excellent support for asynchronous APIs.

### SQLite

SQLite is used for structured metadata storage due to its simplicity, portability, and minimal operational overhead.

### ChromaDB

ChromaDB provides persistent vector storage for semantic retrieval while remaining lightweight and easy to integrate.

### Custom Retrieval Pipeline

Instead of relying on high-level orchestration frameworks, the retrieval pipeline is implemented from first principles to provide greater transparency, flexibility, and control over each stage of the system.

---

# Research Foundations

The implementation draws inspiration from established work in information retrieval and natural language processing, including:

- Retrieval-Augmented Generation (Lewis et al.)
- Attention Is All You Need
- Sentence-BERT
- Dense Passage Retrieval
- BM25 Ranking
- Cross-Encoder Reranking
- BERTopic
- RAGAS

---

# Development

## Clone the repository

```bash
git clone https://github.com/<your-username>/ai-research-copilot.git
```

## Create a virtual environment

```bash
python -m venv .venv
```

## Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Configure environment variables

```bash
cp .env.example .env
```

Update the required API keys inside `.env`.

## Run the application

```bash
uvicorn app.main:app --reload
```

---

# Testing

Run the test suite

```bash
pytest
```

Format the code

```bash
black .
```

Sort imports

```bash
isort .
```

Run static analysis

```bash
ruff check .
```

Run type checking

```bash
mypy app
```

---

# License

This project is licensed under the MIT License.