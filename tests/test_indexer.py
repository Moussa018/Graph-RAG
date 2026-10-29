import networkx as nx

from graph_rag.ingestion.indexer import graph_fingerprint, index_project


def make_graph():
    graph = nx.DiGraph()
    graph.add_node("main", type="Function", file="app.py", doc="Point d'entree")
    graph.add_node("helper", type="Function", file="app.py", doc="Aide")
    graph.add_edge("main", "helper", relation="CALLS")
    return graph


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[0.0] for _ in texts]


class FakeStore:
    def __init__(self, projects=()):
        self.projects = set(projects)

    def has_project(self, project_id):
        return project_id in self.projects

    def write_nodes(self, project_id, nodes, texts, vectors):
        self.projects.add(project_id)

    def write_graph(self, project_id, graph):
        self.projects.add(project_id)


def test_fingerprint_depends_on_edges():
    graph = make_graph()
    other = make_graph()
    other.add_edge("helper", "main", relation="CALLS")
    assert graph_fingerprint(graph) == graph_fingerprint(make_graph())
    assert graph_fingerprint(graph) != graph_fingerprint(other)


def test_index_project_writes_each_store_once():
    embedder, vectors, graphs = FakeEmbedder(), FakeStore(), FakeStore()

    first = index_project(make_graph(), embedder, vectors, graphs)
    second = index_project(make_graph(), embedder, vectors, graphs)

    assert (first.vectors_written, first.graph_written) == (True, True)
    assert (second.vectors_written, second.graph_written) == (False, False)
    assert embedder.calls == 1


def test_index_project_fills_only_missing_graph_store():
    """Passer de Postgres a Neo4j ne refait pas les embeddings."""
    graph = make_graph()
    embedder = FakeEmbedder()
    report = index_project(graph, embedder, FakeStore({graph_fingerprint(graph)}), FakeStore())
    assert (report.vectors_written, report.graph_written) == (False, True)
    assert embedder.calls == 0
