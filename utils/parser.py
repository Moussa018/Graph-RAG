import ast
import zipfile
import networkx as nx
import streamlit as st

class CodeVisitor(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.graph = nx.DiGraph()
        self.current_class = None
        self.current_function = None

    def visit_ClassDef(self, node):
        class_name = node.name
        docstring = ast.get_docstring(node) or "Aucune doc"
        self.graph.add_node(class_name, type="Class", file=self.filename, doc=docstring)
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
        self.graph.add_node(node_name, type="Function", file=self.filename, doc=docstring)
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

@st.cache_data
def process_files_to_graph(uploaded_files):
    combined_graph = nx.DiGraph()
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        
        if filename.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as z:
                for z_filename in z.namelist():
                    if z_filename.endswith(".py"):
                        try:
                            content = z.read(z_filename).decode('utf-8')
                            tree = ast.parse(content)
                            visitor = CodeVisitor(z_filename)
                            visitor.visit(tree)
                            combined_graph = nx.compose(combined_graph, visitor.graph)
                        except Exception:
                            pass
        elif filename.endswith('.py'):
            try:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8')
                tree = ast.parse(content)
                visitor = CodeVisitor(filename)
                visitor.visit(tree)
                combined_graph = nx.compose(combined_graph, visitor.graph)
            except Exception:
                pass
    return combined_graph
