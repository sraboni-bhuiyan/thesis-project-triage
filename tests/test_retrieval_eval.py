import math

import pytest

from retrieval_eval import ndcg_at_k, precision_at_k, reciprocal_rank, score_ratings


def test_precision_at_k():
    assert precision_at_k([1, 0, 1], 3) == pytest.approx(2 / 3)
    assert precision_at_k([0, 0, 0], 3) == 0


def test_reciprocal_rank():
    assert reciprocal_rank([0, 1, 1]) == 0.5
    assert reciprocal_rank([1, 0, 0]) == 1.0
    assert reciprocal_rank([0, 0, 0]) == 0.0


def test_ndcg():
    assert ndcg_at_k([1, 1, 0], 3) == 1.0                       # already ideal
    assert ndcg_at_k([0, 0, 0], 3) == 0.0
    assert ndcg_at_k([0, 0, 1], 3) == pytest.approx((1 / math.log2(4)) / 1.0)
    assert ndcg_at_k([0, 1, 1], 3) == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))


def test_score_ratings():
    r = score_ratings({"a": [1, 0, 0], "b": [0, 0, 0]}, k=3)
    assert r["n_cases"] == 2 and r["n_judgments"] == 6 and r["n_relevant"] == 1
    assert r["p_at_3"] == pytest.approx(1 / 6)
    assert r["mrr"] == 0.5
    assert r["ndcg_at_3"] == 0.5
    assert r["precision_by_rank"] == {"1": 0.5, "2": 0.0, "3": 0.0}
    assert r["cases_with_no_relevant"] == 1
