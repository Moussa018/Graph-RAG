from typing import Protocol

import networkx as nx


class VectorStore(Protocol):
    """Noeuds + embeddings d'un projet (implementation : PgVectorStore)."""

    def has_project(self, project_id: str) -> bool: ...

    def write_nodes(self, project_id: str, nodes: list, texts: list[str], vectors: list) -> None: ...

    def dense_search(self, project_id: str, query_vector, limit: int) -> list[tuple[str, float]]:
        """[(node_id, similarite cosinus)] trie par similarite decroissante."""

    def load_corpus(self, project_id: str) -> list[tuple[str, str]]:
        """[(node_id, texte indexe)] de tout le projet, pour BM25 et le reranker."""


class GraphStore(Protocol):
    """Graphe d'un projet (implementations : Neo4jGraphStore, PostgresGraphStore)."""

    name: str

    def has_project(self, project_id: str) -> bool: ...

    def write_graph(self, project_id: str, graph: nx.DiGraph) -> None: ...

    def expand(self, project_id: str, seeds: list[str], max_depth: int) -> nx.DiGraph:
        """Sous-graphe des seeds et de leurs voisins a max_depth sauts (dans les deux sens).

        Chaque noeud porte `depth` (0 pour un seed) et ses attributs (type, file, doc, source).
        """
