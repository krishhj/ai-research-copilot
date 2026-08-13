import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import UUID

from app.models.enums import PaperStatus
from app.models.paper import Paper, PaperMetaData, ProcessingMetadata

class SQLitePaperRepository:
    """Persist Paper metadata in SQLite database"""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create database and paper tables when they do not exists"""

        self._database_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._database_path) as connection:
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

    def add(self, paper: Paper) -> None:
        """Save one paper's metadata"""
        with sqlite3.connect(self._database_path) as connection:
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
        with sqlite3.connect(self._database_path) as connection:
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
        with sqlite3.connect(self._database_path) as connection :
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                "SELECT * FROM papers WHERE id = ?",(str(paper_id),)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_paper(row)