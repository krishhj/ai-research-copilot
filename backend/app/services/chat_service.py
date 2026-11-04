import re
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
- Never reveal internal paper IDs, chunk IDs, or system details.
- Do not add a separate Sources section; the application displays sources itself.

Citations:
- End every sentence, bullet, sub-bullet, and table row that uses the retrieved text with its own source marker, for example: "RAG-Token can use a different document for each token [Source 2]."
- Use several markers when a point uses several sources, for example [Source 1][Source 3].
- Never collect markers at the end of the answer, and never write page numbers yourself.
""".strip()

        user_prompt = f"""
Paper metadata and retrieved context:

{"\n\n".join(context_parts)}

Question: {query.question}

Remember: put the matching [Source N] marker at the end of each sentence or bullet you write, not at the end of the answer.
""".strip()

        answer_text = self._llm_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        answer_text = self._add_missing_citations(answer_text, results)

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
                else 0.0
            ),
            generation_time=perf_counter() - start_time,
        )

    @staticmethod
    def _add_missing_citations(answer_text: str, results: tuple) -> str:
        """Put a [Source N] marker on each line that is missing one."""
        # The model sometimes writes markers with special spaces or repeats them
        answer_text = re.sub(r"\[Source\s*(\d+)\]", r"[Source \1]", answer_text)
        answer_text = re.sub(r"(\[Source (\d+)\])(\s*\[Source \2\])+", r"\1", answer_text)

        if not results:
            return answer_text

        source_words = [
            set(re.findall(r"[a-z0-9]{4,}", result.text.lower()))
            for result in results
        ]
        answer_lines = answer_text.splitlines()
        lines = []

        for index, line in enumerate(answer_lines):
            text = line.strip()
            next_line = answer_lines[index + 1].strip() if index + 1 < len(answer_lines) else ""

            # A table header row is followed by a |---|---| line and needs no marker
            if text.startswith("|") and next_line and set(next_line) <= set("|-: "):
                lines.append(line)
                continue

            # The model sometimes collects every marker on its own line at the end
            if re.fullmatch(r"(\[Source \d+\]\s*)+", text):
                continue

            if (
                len(text) < 25
                or text.startswith("#")
                or re.search(r"\[Source \d+\]", text)
                or set(text) <= set("|-: ")
            ):
                lines.append(line)
                continue

            line_words = set(re.findall(r"[a-z0-9]{4,}", text.lower()))
            overlaps = [len(line_words & words) for words in source_words]
            best = max(range(len(results)), key=lambda index: overlaps[index])

            if not line_words or overlaps[best] / len(line_words) < 0.3:
                lines.append(line)
                continue

            marker = f"[Source {best + 1}]"
            if text.endswith("|"):
                line = line.rstrip()[:-1].rstrip() + f" {marker} |"
            else:
                line = line.rstrip() + f" {marker}"
            lines.append(line)

        return "\n".join(lines)

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