from collections.abc import Sequence


def first_relevant_rank(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> int | None:
    """Return the 1-based position of the first chunk from an expected page, or None"""
    for rank, page_number in enumerate(retrieved_pages, start=1):
        if page_number in expected_pages:
            return rank

    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of questions whose answer page appears in the top k results"""
    if not ranks:
        return 0.0

    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Average of 1/rank of the first relevant result (0 when it was never found)"""
    if not ranks:
        return 0.0

    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
