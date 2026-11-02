import pytest

from evaluation.metrics import first_relevant_rank, mean_reciprocal_rank, recall_at_k


def test_first_relevant_rank_returns_position_of_first_match():
    assert first_relevant_rank([2, 7, 4, 4], expected_pages=[4]) == 3


def test_first_relevant_rank_returns_none_when_missing():
    assert first_relevant_rank([1, 2, 3], expected_pages=[9]) is None


def test_recall_at_k_counts_questions_found_within_k():
    ranks = [1, 3, None, 5]

    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.5
    assert recall_at_k(ranks, 5) == 0.75


def test_mean_reciprocal_rank_rewards_higher_positions():
    assert mean_reciprocal_rank([1, 2, None, 4]) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_metrics_handle_empty_input():
    assert recall_at_k([], 5) == 0.0
    assert mean_reciprocal_rank([]) == 0.0
