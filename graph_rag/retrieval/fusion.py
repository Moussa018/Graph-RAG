def reciprocal_rank_fusion(*rankings, k=60):
    """Fusionne plusieurs classements : score(d) = somme de 1 / (k + rang(d)).

    Seuls les rangs comptent, donc des scores d'echelles differentes
    (cosinus, BM25) se combinent sans calibration.
    Retourne [(node_id, score)] trie par score decroissant.
    """
    scores = {}
    for ranking in rankings:
        for rank, node_id in enumerate(ranking, start=1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
