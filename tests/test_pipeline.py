import networkx as nx

from graph_rag.config import Settings
from graph_rag.retrieval.pipeline import HybridRetriever

SETTINGS = Settings(
    google_api_key=None, llm_model="", embed_model="", embed_dim=1,
    database_url="", graph_backend="postgres",
    neo4j_uri="", neo4j_user="", neo4j_password="", neo4j_database="",
    reranker_model="", dense_k=3, bm25_k=3, rrf_k=60, rerank_candidates=3, top_k=1, graph_depth=1,
)


class FakeEmbedder:
    def embed_query(self, text):
        return [0.0]


class FakeVectorStore:
    def dense_search(self, project_id, query_vector, limit):
        return [("render", 0.9), ("save", 0.8)]

    def load_corpus(self, project_id):
        return [("load_config", "Nom: load_config"), ("render", "Nom: render"), ("save", "Nom: save")]


class FakeGraphStore:
    def __init__(self):
        self.seeds = None

    def expand(self, project_id, seeds, max_depth):
        self.seeds = seeds
        graph = nx.DiGraph()
        for seed in seeds:
            graph.add_node(seed, depth=0)
        return graph


class KeywordReranker:
    def rerank(self, query, candidates, top_k):
        return sorted(((n, float(n in query)) for n, _ in candidates), key=lambda x: -x[1])[:top_k]


class BrokenReranker:
    def rerank(self, query, candidates, top_k):
        raise OSError("model not found")


def test_reranker_picks_seed_from_fused_candidates():
    graph_store = FakeGraphStore()
    retriever = HybridRetriever(SETTINGS, FakeEmbedder(), FakeVectorStore(), graph_store, KeywordReranker())

    result = retriever.retrieve("p1", "que fait load_config ?")

    assert graph_store.seeds == ["load_config"]
    assert result.trace[0]["noeud"] == "load_config"
    assert result.trace[0]["rang BM25"] == 1
    assert result.trace[0]["rang pgvector"] is None
    assert result.warnings == []


def test_falls_back_to_rrf_when_reranker_fails():
    graph_store = FakeGraphStore()
    retriever = HybridRetriever(SETTINGS, FakeEmbedder(), FakeVectorStore(), graph_store, BrokenReranker())

    result = retriever.retrieve("p1", "render")

    assert graph_store.seeds == ["render"]  # 1er dans pgvector et dans BM25
    assert "model not found" in result.warnings[0]
