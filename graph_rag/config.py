import os
from dataclasses import dataclass


def _env_int(name, default):
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    """Configuration de l'application, lue depuis les variables d'environnement (.env)."""

    google_api_key: str | None
    llm_model: str
    embed_model: str
    embed_dim: int

    # Stockage
    database_url: str
    graph_backend: str  # "neo4j" ou "postgres"
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str

    # Recherche
    reranker_model: str
    dense_k: int            # candidats pgvector
    bm25_k: int             # candidats BM25
    rrf_k: int              # constante de la Reciprocal Rank Fusion
    rerank_candidates: int  # candidats fusionnes envoyes au cross-encoder
    top_k: int              # noeuds "seeds" retenus apres reranking
    graph_depth: int        # sauts d'expansion dans le graphe autour des seeds

    @classmethod
    def from_env(cls):
        graph_backend = os.environ.get("GRAPH_BACKEND", "neo4j").lower()
        if graph_backend not in ("neo4j", "postgres"):
            raise ValueError(f"GRAPH_BACKEND doit valoir 'neo4j' ou 'postgres', pas {graph_backend!r}")
        return cls(
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            llm_model=os.environ.get("LLM_MODEL", "gemini-3.5-flash"),
            embed_model=os.environ.get("EMBED_MODEL", "text-embedding-004"),
            embed_dim=_env_int("EMBED_DIM", 768),
            database_url=os.environ.get("DATABASE_URL", "postgresql://graphrag:graphrag@localhost:5432/graphrag"),
            graph_backend=graph_backend,
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", "graphrag-password"),
            neo4j_database=os.environ.get("NEO4J_DATABASE", "neo4j"),
            reranker_model=os.environ.get("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
            dense_k=_env_int("DENSE_K", 20),
            bm25_k=_env_int("BM25_K", 20),
            rrf_k=_env_int("RRF_K", 60),
            rerank_candidates=_env_int("RERANK_CANDIDATES", 20),
            top_k=_env_int("TOP_K", 5),
            graph_depth=_env_int("GRAPH_DEPTH", 1),
        )
