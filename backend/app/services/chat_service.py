from time import perf_counter

from app.models.answer import Answer, Citation, RetrievedChunk
from app.models.query import Query
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService
from app.storage.sqlite import SQLitePaperRepository

class ChatService:
    """Answer research questions using retrieved paper chunks"""

    def __init__(self, retrieval_service: RetrievalService, paper_repository: SQLitePaperRepository, llm_service: LLMService) -> None:
        self._retrieval_service = retrieval_service
        self._paper_repository = paper_repository
        self._llm_service = llm_service

    def answer_question(self, query: Query) -> Answer:
        """Retrieve context and generate a grounded answer"""
        start_time = perf_counter()
        results = self._retrieval_service.search(query=query)

        if not results:
            return Answer(
                answer_text=("I could not find relevant information in the uploaded research papers"),
                confidence=0.0,
                generation_time= perf_counter() - start_time
            )

        context = "\n\n".join(
            (
                f"[Source {index} | paper_id={result.paper_id} | page={result.page_number}\n{result.text}]"
            )
            for index, result in enumerate(results, start=1)
        )

        system_prompt = """
You are an AI research assistant.

Answer only from the supplied research-paper context.
If the context does not contain enough information, say so clearly.
Do not use outside knowledge.
Use citations like [Source 1] when making factual claims.
""".strip()

        user_prompt = f"""
Research-paper context:

{context}

Question: {query.question}
""".strip()

        answer_text = self._llm_service.generate(system_prompt=system_prompt, user_prompt=user_prompt)

        citations = []
        for result in results:
            paper = self._paper_repository.get_by_id(result.paper_id)

            citations.append(
                Citation(
                    paper_id=result.paper_id,
                    paper_title=(
                        paper.metadata.title
                        if paper and paper.metadata.title
                        else "Untitled paper"
                    ),
                    chunk_id= result.chunk_id,
                    page_number=result.page_number
                )
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
            confidence= max(result.score for result in results),
            generation_time=perf_counter() - start_time
        )