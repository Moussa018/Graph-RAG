from dataclasses import dataclass, field

import networkx as nx

from graph_rag.retrieval.bm25 import BM25Index
from graph_rag.retrieval.fusion import reciprocal_rank_fusion


@dataclass
class RetrievalResult:
    subgraph: nx.DiGraph
    trace: list[dict]                          # une ligne par candidat, pour l'affichage
    warnings: list[str] = field(default_factory=list)


class HybridRetriever:
    """Recherche hybride puis expansion dans le graphe.

    1. pgvector (cosinus) et BM25 donnent chacun leurs candidats.
    2. Les deux classements sont fusionnes par RRF.
    3. Le cross-encoder reclasse les candidats fusionnes, les top_k deviennent les seeds.
    4. Le graph store (Neo4j ou Postgres) ajoute les voisins a graph_depth sauts.
    """

    def __init__(self, settings, embedder, vector_store, graph_store, reranker):
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.reranker = reranker
        self._bm25_indexes = {}

    def bm25_index(self, project_id):
        if project_id not in self._bm25_indexes:
            rows = self.vector_store.load_corpus(project_id)
            self._bm25_indexes[project_id] = BM25Index(
                [node_id for node_id, _ in rows], [content for _, content in rows]
            )
        return self._bm25_indexes[project_id]

    def retrieve(self, project_id, question):
        s = self.settings
        warnings = []
        bm25_index = self.bm25_index(project_id)

        dense = self.vector_store.dense_search(project_id, self.embedder.embed_query(question), s.dense_k)
        lexical = bm25_index.search(question, s.bm25_k)

        fused = reciprocal_rank_fusion([n for n, _ in dense], [n for n, _ in lexical], k=s.rrf_k)
        candidates = fused[:s.rerank_candidates]

        try:
            reranked = self.reranker.rerank(
                question, [(n, bm25_index.texts[n]) for n, _ in candidates], s.top_k
            )
        except Exception as e:
            warnings.append(f"Reranker indisponible, classement RRF utilisé : {e}")
            reranked = [(n, None) for n, _ in candidates[:s.top_k]]

        seeds = [n for n, _ in reranked]
        subgraph = self.graph_store.expand(project_id, seeds, s.graph_depth)
        return RetrievalResult(subgraph, _build_trace(candidates, dense, lexical, reranked), warnings)


def _build_trace(candidates, dense, lexical, reranked):
    """Rang de chaque candidat dans chaque etape ; les seeds d'abord, dans l'ordre du reranker."""
    dense_rank = {n: i for i, (n, _) in enumerate(dense, start=1)}
    bm25_rank = {n: i for i, (n, _) in enumerate(lexical, start=1)}
    seed_order = {n: i for i, (n, _) in enumerate(reranked)}
    rerank_score = dict(reranked)
    trace = [
        {
            "noeud": n,
            "rang pgvector": dense_rank.get(n),
            "rang BM25": bm25_rank.get(n),
            "score RRF": round(score, 4),
            "score reranker": None if rerank_score.get(n) is None else round(rerank_score[n], 3),
            "seed": n in seed_order,
        }
        for n, score in candidates
    ]
    trace.sort(key=lambda row: seed_order.get(row["noeud"], len(seed_order)))
    return trace
