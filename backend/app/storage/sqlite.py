import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.core.exceptions import DatabaseError
from app.models.chunk import Chunk
from app.models.enums import PaperStatus
from app.models.paper import Paper, PaperMetaData, ProcessingMetadata


class SQLitePaperRepository:
    """Persist papers and enforce ownership at the database boundary."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create tables and safely add ownership to existing databases."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    title TEXT,
                    authors TEXT NOT NULL,
                    abstract TEXT,
                    year INTEGER,
                    doi TEXT,
                    journal TEXT,
                    keywords TEXT NOT NULL,
                    stored_filename TEXT NOT NULL UNIQUE,
                    total_pages INTEGER NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

            existing_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(papers)")
            }

            # Supports the database created before user accounts existed.
            # Legacy rows have user_id = NULL and are intentionally hidden.
            if "user_id" not in existing_columns:
                connection.execute(
                    "ALTER TABLE papers ADD COLUMN user_id TEXT"
                )

            if "content_hash" not in existing_columns:
                connection.execute(
                    "ALTER TABLE papers ADD COLUMN content_hash TEXT"
                )

            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                idx_papers_owner_content_hash
                ON papers(user_id, content_hash)
                WHERE content_hash IS NOT NULL
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_papers_user_id
                ON papers(user_id)
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    embedding_created INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (paper_id, chunk_index),
                    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
                )
                """
            )

    def add(self, paper: Paper, owner_id: UUID, content_hash: str) -> None:
        """Save one paper for its authenticated owner."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO papers (
                    id, user_id, content_hash, title, authors, abstract, year, doi,
                    journal, keywords, stored_filename, total_pages,
                    total_chunks, uploaded_at, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(paper.id),
                    str(owner_id),
                    content_hash,
                    paper.metadata.title,
                    json.dumps(paper.metadata.authors),
                    paper.metadata.abstract,
                    paper.metadata.year,
                    paper.metadata.doi,
                    paper.metadata.journal,
                    json.dumps(paper.metadata.keywords),
                    paper.processing.stored_filename,
                    paper.processing.total_pages,
                    paper.processing.total_chunks,
                    paper.processing.uploaded_at.isoformat(),
                    paper.processing.status.value,
                ),
            )

    def list_by_owner(self, owner_id: UUID) -> list[Paper]:
        """Return only papers owned by one authenticated user."""
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT * FROM papers
                WHERE user_id = ?
                ORDER BY uploaded_at DESC
                """,
                (str(owner_id),),
            ).fetchall()

        return [self._row_to_paper(row) for row in rows]

    def get_by_id_and_owner(
        self,
        paper_id: UUID,
        owner_id: UUID,
    ) -> Paper | None:
        """Return a paper only when it belongs to the requested owner."""
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                """
                SELECT * FROM papers
                WHERE id = ? AND user_id = ?
                """,
                (str(paper_id), str(owner_id)),
            ).fetchone()

        return self._row_to_paper(row) if row else None

    def get_by_content_hash_and_owner(
        self,
        content_hash: str,
        owner_id: UUID,
    ) -> Paper | None:
        """Return a matching paper only for the specified owner."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM papers
                WHERE content_hash = ? AND user_id = ?
                """,
                (content_hash, str(owner_id)),
            ).fetchone()

        return self._row_to_paper(row) if row else None

    def delete_by_id_and_owner(
        self,
        paper_id: UUID,
        owner_id: UUID,
    ) -> bool:
        """Delete a paper only when it belongs to the requested owner."""

        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM papers
                WHERE id = ? AND user_id = ?
                """,
                (str(paper_id), str(owner_id)),
            )

        return cursor.rowcount == 1

    def save_processing_result(
        self,
        paper_id: UUID,
        owner_id: UUID,
        total_pages: int,
        chunks: tuple[Chunk, ...],
        metadata: PaperMetaData,
    ) -> None:
        """Save chunks and metadata only for the paper owner."""
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM chunks
                WHERE paper_id = ?
                  AND EXISTS (
                      SELECT 1 FROM papers
                      WHERE papers.id = chunks.paper_id
                        AND papers.user_id = ?
                  )
                """,
                (str(paper_id), str(owner_id)),
            )

            connection.executemany(
                """
                INSERT INTO chunks (
                    id, paper_id, chunk_index, page_number,
                    chunk_text, token_count, embedding_created
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(chunk.id),
                        str(chunk.paper_id),
                        chunk.chunk_index,
                        chunk.page_number,
                        chunk.chunk_text,
                        chunk.token_count,
                        int(chunk.embedding_created),
                    )
                    for chunk in chunks
                ],
            )

            cursor = connection.execute(
                """
                UPDATE papers
                SET
                    title = ?,
                    authors = ?,
                    abstract = ?,
                    year = ?,
                    doi = ?,
                    journal = ?,
                    keywords = ?,
                    total_pages = ?,
                    total_chunks = ?,
                    status = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    metadata.title,
                    json.dumps(metadata.authors),
                    metadata.abstract,
                    metadata.year,
                    metadata.doi,
                    metadata.journal,
                    json.dumps(metadata.keywords),
                    total_pages,
                    len(chunks),
                    PaperStatus.PROCESSED.value,
                    str(paper_id),
                    str(owner_id),
                ),
            )

            if cursor.rowcount != 1:
                raise DatabaseError("Paper not found.")

    def list_chunks_by_paper_id(
        self,
        paper_id: UUID,
        owner_id: UUID,
    ) -> list[Chunk]:
        """Return chunks only when their paper belongs to the owner."""
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT chunks.*
                FROM chunks
                JOIN papers ON papers.id = chunks.paper_id
                WHERE chunks.paper_id = ?
                  AND papers.user_id = ?
                ORDER BY chunks.chunk_index
                """,
                (str(paper_id), str(owner_id)),
            ).fetchall()

        return [
            Chunk(
                id=row["id"],
                paper_id=row["paper_id"],
                chunk_index=row["chunk_index"],
                page_number=row["page_number"],
                chunk_text=row["chunk_text"],
                token_count=row["token_count"],
                embedding_created=bool(row["embedding_created"]),
            )
            for row in rows
        ]

    def update_status(
        self,
        paper_id: UUID,
        owner_id: UUID,
        status: PaperStatus,
    ) -> None:
        """Update status only when the paper belongs to the owner."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE papers
                SET status = ?
                WHERE id = ? AND user_id = ?
                """,
                (status.value, str(paper_id), str(owner_id)),
            )

        if cursor.rowcount != 1:
            raise DatabaseError("Paper not found.")

    def mark_chunks_embedded(
        self,
        paper_id: UUID,
        owner_id: UUID,
    ) -> int:
        """Mark chunks embedded only when their paper belongs to the owner."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE chunks
                SET embedding_created = 1
                WHERE paper_id = ?
                  AND EXISTS (
                      SELECT 1 FROM papers
                      WHERE papers.id = chunks.paper_id
                        AND papers.user_id = ?
                  )
                """,
                (str(paper_id), str(owner_id)),
            )

        return cursor.rowcount

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Paper:
        """Convert one database row into a Paper domain object."""
        return Paper(
            id=row["id"],
            metadata=PaperMetaData(
                title=row["title"],
                authors=json.loads(row["authors"]),
                abstract=row["abstract"],
                year=row["year"],
                doi=row["doi"],
                journal=row["journal"],
                keywords=json.loads(row["keywords"]),
            ),
            processing=ProcessingMetadata(
                stored_filename=row["stored_filename"],
                total_pages=row["total_pages"],
                total_chunks=row["total_chunks"],
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
                status=PaperStatus(row["status"]),
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection with foreign-key support."""
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection