import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.models.enums import PaperStatus
from app.models.paper import Paper, PaperMetaData, ProcessingMetadata
from app.core.exceptions import DatabaseError
from app.models.chunk import Chunk
class SQLitePaperRepository:
    """Persist Paper metadata in SQLite database"""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create database and paper tables when they do not exists"""

        self._database_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
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

    def add(self, paper: Paper) -> None:
        """Save one paper's metadata"""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO papers(
                    id, title, authors, abstract, year, doi, journal, keywords, stored_filename, total_pages, total_chunks, uploaded_at, status
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(paper.id),
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

    def list_all(self) -> list[Paper]:
        """Return all stored papers, newest first"""
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                "SELECT * FROM papers ORDER BY uploaded_at DESC"
            ).fetchall()

        return [self._row_to_paper(row) for row in rows]

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Paper:
        """Convert one database row into a Paper domain object"""
        return Paper(
            id = row["id"],
            metadata=PaperMetaData(
                title=row["title"],
                authors = json.loads(row["authors"]),
                abstract=row["abstract"],
                year= row["year"],
                doi= row["doi"],
                journal= row["journal"],
                keywords=json.loads(row["keywords"]),
            ),
            processing=ProcessingMetadata(
                stored_filename=row["stored_filename"],
                total_pages=row["total_pages"],
                total_chunks=row["total_chunks"],
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
                status=PaperStatus(row["status"])
            )
        )

    def get_by_id(self, paper_id: UUID) -> Paper | None:
        """Return one paper by ID, or None when it does not exist"""
        with self._connect() as connection :
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                "SELECT * FROM papers WHERE id = ?",(str(paper_id),)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_paper(row)

    def delete(self, paper_id: UUID) -> bool:
        """Delete one paper record and report wether it existed"""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM papers WHERE id = ?",
                (str(paper_id),),
            )

        return cursor.rowcount == 1

    def _connect(self) -> sqlite3.Connection:
        """Create a sqlite connection with Foreign-key support enabled"""
        connection = sqlite3.connect(self._database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def save_processing_result(self, paper_id: UUID, total_pages: int, chunks: tuple[Chunk, ...]) -> None:
        """Save processed chunk and update the paper's processing metadata"""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM chunks WHERE paper_id = ?",
                (str(paper_id),)
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
                SET total_pages = ?, total_chunks = ?, status = ?
                WHERE id = ?
                """,
                (
                    total_pages,
                    len(chunks),
                    PaperStatus.PROCESSED.value,
                    str(paper_id),
                ),
            )

            if cursor.rowcount != 1:
                raise DatabaseError(f"Paper not found: {paper_id}")

    def list_chunks_by_paper_id(self, paper_id: UUID) -> list[Chunk]:
        """Return all chunks for one paper in their original order"""
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                    SELECT * FROM chunks
                    WHERE paper_id = ?
                    ORDER BY chunk_index
                    """,
                    (str(paper_id),),
                ).fetchall()

        return [
            Chunk(
                id = row["id"],
                paper_id= row["paper_id"],
                chunk_index=row["chunk_index"],
                page_number=row["page_number"],
                chunk_text=row["chunk_text"],
                token_count=row["token_count"],
                embedding_created=bool(row["embedding_created"]),
            )
            for row in rows
        ]

    def update_status(self, paper_id: UUID, status: PaperStatus) -> None:
        """Update one paper's processing status"""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE papers SET status = ? WHERE id = ?",
                (status.value, str(paper_id)),
            )

        if cursor.rowcount != 1:
            raise DatabaseError(f"Paper not found: {paper_id}")