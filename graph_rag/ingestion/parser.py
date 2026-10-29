import ast
import zipfile

import networkx as nx

MAX_SOURCE_CHARS = 1500  # code source conserve par noeud (BM25, embedding, reranker et contexte LLM)


class CodeVisitor(ast.NodeVisitor):
    """Construit le graphe d'un fichier : classes, fonctions, heritages et appels."""

    def __init__(self, filename, content=""):
        self.filename = filename
        self.content = content
        self.graph = nx.DiGraph()
        self.current_class = None
        self.current_function = None

    def _source(self, node):
        return (ast.get_source_segment(self.content, node) or "")[:MAX_SOURCE_CHARS]

    def visit_ClassDef(self, node):
        class_name = node.name
        docstring = ast.get_docstring(node) or "Aucune doc"
        self.graph.add_node(class_name, type="Class", file=self.filename, doc=docstring, source=self._source(node))
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.graph.add_edge(class_name, base.id, relation="INHERITS_FROM")
        prev_class = self.current_class
        self.current_class = class_name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        func_name = node.name
        node_name = f"{self.current_class}.{func_name}" if self.current_class else func_name
        docstring = ast.get_docstring(node) or "Aucune doc"
        self.graph.add_node(node_name, type="Function", file=self.filename, doc=docstring, source=self._source(node))
        if self.current_class:
            self.graph.add_edge(self.current_class, node_name, relation="HAS_METHOD")
        prev_func = self.current_function
        self.current_function = node_name
        self.generic_visit(node)
        self.current_function = prev_func

    def visit_Call(self, node):
        if self.current_function:
            called_func = None
            if isinstance(node.func, ast.Name):
                called_func = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called_func = node.func.attr
            if called_func:
                self.graph.add_edge(self.current_function, called_func, relation="CALLS")
        self.generic_visit(node)


def iter_python_sources(uploaded_files):
    """(chemin, octets) de chaque fichier .py uploade, y compris ceux contenus dans une archive .zip."""
    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        uploaded_file.seek(0)
        if name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as z:
                for z_name in z.namelist():
                    if z_name.endswith(".py"):
                        yield z_name, z.read(z_name)
        elif name.endswith(".py"):
            yield name, uploaded_file.read()


def build_graph(sources):
    """Graphe combine de tous les fichiers ; les fichiers illisibles ou invalides sont ignores."""
    graph = nx.DiGraph()
    for filename, raw in sources:
        try:
            content = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        visitor = CodeVisitor(filename, content)
        visitor.visit(tree)
        graph = nx.compose(graph, visitor.graph)
    return graph
