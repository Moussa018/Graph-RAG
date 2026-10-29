from graph_rag.ingestion.parser import build_graph

SOURCE = '''
class Base:
    pass


class Loader(Base):
    """Charge des fichiers."""

    def load(self, path):
        return read_file(path)


def read_file(path):
    return open(path).read()
'''


def test_build_graph_extracts_nodes_and_relations():
    graph = build_graph([("loader.py", SOURCE.encode("utf-8"))])

    assert graph.nodes["Loader"]["type"] == "Class"
    assert graph.nodes["Loader"]["doc"] == "Charge des fichiers."
    assert graph.nodes["Loader.load"]["type"] == "Function"
    assert graph.nodes["read_file"]["file"] == "loader.py"

    assert graph.edges["Loader", "Base"]["relation"] == "INHERITS_FROM"
    assert graph.edges["Loader", "Loader.load"]["relation"] == "HAS_METHOD"
    assert graph.edges["Loader.load", "read_file"]["relation"] == "CALLS"


def test_build_graph_keeps_source_code():
    graph = build_graph([("loader.py", SOURCE)])
    assert graph.nodes["read_file"]["source"].startswith("def read_file(path):")


def test_build_graph_skips_invalid_files():
    graph = build_graph([("broken.py", b"def oops(:"), ("ok.py", b"def fine():\n    pass\n")])
    assert list(graph.nodes) == ["fine"]
