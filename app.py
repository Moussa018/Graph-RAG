import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from google import genai
import os
import zipfile
import ast
from pyvis.network import Network
from dotenv import load_dotenv

# Extrait de code intégré pour rendre l'app 100% autonome et lire les commentaires
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

# ----- INTERFACE PRINCIPALE -----
load_dotenv()
st.set_page_config(page_title="Ultimate Graph RAG", layout="wide", page_icon="🌌")
st.title("Graph RAG (Code Explorer)")

if not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "votre_cle_api_google_ici":
    st.error("Clé API Google manquante dans le fichier .env")
    st.stop()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Gestion de la mémoire du Chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_data
def process_zip_to_graph(zip_bytes):
    combined_graph = nx.DiGraph()
    with zipfile.ZipFile(zip_bytes, 'r') as z:
        for filename in z.namelist():
            if filename.endswith(".py"):
                try:
                    content = z.read(filename).decode('utf-8')
                    tree = ast.parse(content)
                    visitor = CodeVisitor(filename)
                    visitor.visit(tree)
                    combined_graph = nx.compose(combined_graph, visitor.graph)
                except Exception:
                    pass
    return combined_graph

def generate_interactive_graph(g):
    net = Network(height="500px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)
    for node, data in g.nodes(data=True):
        color = "#ff4b4b" if data.get("type") == "Class" else "#00b4d8"
        title = f"Fichier: {data.get('file', 'inconnu')}\nDescription: {data.get('doc', 'Aucune')}"
        net.add_node(node, label=node, title=title, color=color)
    for u, v, data in g.edges(data=True):
        net.add_edge(u, v, title=data.get("relation", ""))
    
    net.save_graph("graph_vis.html")
    with open("graph_vis.html", "r", encoding="utf-8") as f:
        return f.read()

# Sidebar pour l'upload
st.sidebar.markdown("###1. Uploadez un projet ZIP")
uploaded_file = st.sidebar.file_uploader("Fichier .zip de code Python", type=["zip"])

graph = None
if uploaded_file:
    with st.spinner("Extraction du code à la volée..."):
        graph = process_zip_to_graph(uploaded_file)
elif os.path.exists("code_graph.graphml"):
    graph = nx.read_graphml("code_graph.graphml")

if graph and graph.number_of_nodes() > 0:
    st.sidebar.success(f"Graphe en mémoire : {graph.number_of_nodes()} Noeuds et {graph.number_of_edges()} Relations")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Carte Interactive de l'Architecture")
        st.caption("Vous pouvez zoomer, déplacer les noeuds, et les survoler pour lire le code.")
        html_data = generate_interactive_graph(graph)
        components.html(html_data, height=520)
    
    with col2:
        st.markdown("### Assistant")
        
        # Affichage du chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Préparation du contexte du graphe
        graph_context = ""
        for u, v, data in graph.edges(data=True):
            graph_context += f"- '{u}' {data.get('relation', 'appels')} '{v}'\n"
            
        if prompt := st.chat_input("Posez votre question à l'architecte..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyse du graphe..."):
                    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
                    
                    full_prompt = f"""
                    Tu es un architecte logiciel IA. Tu vas répondre au développeur en t'appuyant sur l'architecture ci-dessous.
                    Graphe des relations :
                    {graph_context}
                    
                    Historique du chat :
                    {history_text}
                    
                    Réponds à la question de façon concise et experte.
                    """
                    try:
                        response = client.models.generate_content(
                            model='gemini-3.5-flash',
                            contents=full_prompt
                        )
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"Erreur API : {e}")

else:
    st.info("Veuillez générer un graphe local ou uploader un ZIP dans le panneau latéral gauche.")
