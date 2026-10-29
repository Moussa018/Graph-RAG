import pytest

from graph_rag.retrieval.fusion import reciprocal_rank_fusion


def test_document_in_both_rankings_wins():
    fused = reciprocal_rank_fusion(["a", "b", "c"], ["c", "d"], k=60)
    assert fused[0][0] == "c"
    assert fused[0][1] == pytest.approx(1 / 63 + 1 / 61)


def test_single_ranking_keeps_its_order():
    fused = reciprocal_rank_fusion(["x", "y", "z"])
    assert [node for node, _ in fused] == ["x", "y", "z"]
