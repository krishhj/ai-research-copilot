from time import perf_counter
from uuid import UUID

from app.models.answer import Answer, Citation, RetrievedChunk
from app.models.paper import Paper
from app.models.query import Query
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService
from app.storage.sqlite import SQLitePaperRepository


class ChatService:
    """Answer questions using a user's metadata and indexed paper chunks."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        paper_repository: SQLitePaperRepository,
        llm_service: LLMService,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._paper_repository = paper_repository
        self._llm_service = llm_service

    def answer_question(
        self,
        query: Query,
        owner_id: UUID,
    ) -> Answer:
        """Generate a grounded answer for the authenticated owner."""
        start_time = perf_counter()

        selected_paper = self._get_selected_paper(
            paper_id=query.paper_id,
            owner_id=owner_id,
        )

        results = self._retrieval_service.search(query, owner_id)

        if not results and selected_paper is None:
            return Answer(
                answer_text=(
                    "I could not find relevant information in your "
                    "uploaded research papers. Select a specific paper "
                    "when asking about its authors or metadata."
                ),
                confidence=0.0,
                generation_time=perf_counter() - start_time,
            )

        metadata_context = (
            self._build_metadata_context(selected_paper)
            if selected_paper
            else ""
        )

        retrieved_context = "\n\n".join(
            (
                f"[Source {index} | page={result.page_number}]\n"
                f"{result.text}"
            )
            for index, result in enumerate(results, start=1)
        )

        context_parts = [
            part
            for part in (metadata_context, retrieved_context)
            if part
        ]

        system_prompt = """
You are an AI research assistant.

Answer only from the supplied paper metadata and research-paper context.
If the context does not contain enough information, say so clearly.
Do not use outside knowledge.

Format answers in clean Markdown:
- Use a short descriptive heading when useful.
- Use bullet points for key findings.
- Use a table only when it genuinely makes a comparison clearer.
- Use citations like [Source 1] for factual claims based on retrieved text.
- Never reveal internal paper IDs, chunk IDs, or system details.
- Do not add a separate Sources section; the application displays sources itself.
""".strip()

        user_prompt = f"""
Paper metadata and retrieved context:

{"\n\n".join(context_parts)}

Question: {query.question}
""".strip()

        answer_text = self._llm_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        citations = self._build_citations(
            results=results,
            owner_id=owner_id,
            selected_paper=selected_paper,
        )

        return Answer(
            answer_text=answer_text,
            citations=citations,
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id=result.chunk_id,
                    score=result.score,
                )
                for result in results
            ],
            confidence=(
                max(result.score for result in results)
                if results
                else 1.0
            ),
            generation_time=perf_counter() - start_time,
        )

    def _get_selected_paper(
        self,
        paper_id: UUID | None,
        owner_id: UUID,
    ) -> Paper | None:
        """Return the selected paper only when it belongs to the owner."""
        if paper_id is None:
            return None

        return self._paper_repository.get_by_id_and_owner(
            paper_id,
            owner_id,
        )

    @staticmethod
    def _build_metadata_context(paper: Paper) -> str:
        """Build trusted context from metadata already extracted from a PDF."""
        authors = ", ".join(paper.metadata.authors) or "Unknown"
        year = str(paper.metadata.year) if paper.metadata.year else "Unknown"
        doi = paper.metadata.doi or "Not available"
        abstract = paper.metadata.abstract or "Not available"

        return f"""
Selected paper metadata:
- Title: {paper.metadata.title or "Untitled paper"}
- Authors: {authors}
- Year: {year}
- DOI: {doi}
- Abstract: {abstract}
""".strip()

    def _build_citations(
        self,
        results: tuple,
        owner_id: UUID,
        selected_paper: Paper | None,
    ) -> list[Citation]:
        """Build citations from retrieved chunks or the selected paper."""
        if results:
            return [
                Citation(
                    paper_id=result.paper_id,
                    paper_title=self._paper_title(
                        paper_id=result.paper_id,
                        owner_id=owner_id,
                    ),
                    chunk_id=result.chunk_id,
                    page_number=result.page_number,
                )
                for result in results
            ]

        if selected_paper is None:
            return []

        chunks = self._paper_repository.list_chunks_by_paper_id(
            selected_paper.id,
            owner_id,
        )

        if not chunks:
            return []

        first_chunk = chunks[0]

        return [
            Citation(
                paper_id=selected_paper.id,
                paper_title=(
                    selected_paper.metadata.title
                    or "Untitled paper"
                ),
                chunk_id=first_chunk.id,
                page_number=first_chunk.page_number,
            )
        ]

    def _paper_title(
        self,
        paper_id: UUID,
        owner_id: UUID,
    ) -> str:
        """Return an owned paper title for a citation."""
        paper = self._paper_repository.get_by_id_and_owner(
            paper_id,
            owner_id,
        )

        if paper and paper.metadata.title:
            return paper.metadata.title

        return "Untitled paper"