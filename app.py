"""Interface Streamlit : upload, indexation, carte du graphe, chat et trace de recherche."""
import os

from dotenv import load_dotenv

load_dotenv()

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components

from graph_rag.config import Settings
from graph_rag.generation.prompt import build_graph_context, build_prompt
from graph_rag.ingestion.indexer import index_project
from graph_rag.ingestion.parser import build_graph, iter_python_sources
from graph_rag.services import build_services
from graph_rag.ui.visualizer import generate_interactive_graph

HISTORY_MESSAGES = 3  # messages precedents envoyes au LLM


@st.cache_resource
def get_services():
    return build_services(Settings.from_env())


@st.cache_data
def load_graph(uploaded_files):
    return build_graph(iter_python_sources(uploaded_files))


# ----- INTERFACE PRINCIPALE -----
st.set_page_config(page_title="Ultimate Graph RAG", layout="wide", page_icon="🌌")
st.title("Graph RAG (Code Explorer)")

if not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "votre_cle_api_google_ici":
    st.error("Clé API Google manquante dans le fichier .env")
    st.stop()

try:
    services = get_services()
except Exception as e:
    st.error(f"Configuration invalide : {e}")
    st.stop()

# Gestion de la mémoire du Chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar pour l'upload
st.sidebar.markdown("### 1. Uploadez vos fichiers")
uploaded_files = st.sidebar.file_uploader(
    "Fichiers .py ou archive .zip",
    type=["zip", "py"],
    accept_multiple_files=True
)

graph = None
if uploaded_files:
    with st.spinner("Extraction du code à la volée..."):
        graph = load_graph(uploaded_files)
elif os.path.exists("code_graph.graphml"):
    graph = nx.read_graphml("code_graph.graphml")

if not graph or graph.number_of_nodes() == 0:
    st.info("Veuillez générer un graphe local ou uploader un ZIP dans le panneau latéral gauche.")
    st.stop()

st.sidebar.success(f"Graphe en mémoire : {graph.number_of_nodes()} Noeuds et {graph.number_of_edges()} Relations")

# Indexation : vecteurs dans pgvector, graphe dans Neo4j (ou Postgres) ; chaque store est ignore s'il a deja le projet
project_id = None
try:
    with st.spinner(f"Indexation (pgvector + {services.graph_store.name})..."):
        report = index_project(graph, services.embedder, services.vector_store, services.graph_store)
    project_id = report.project_id
    st.sidebar.caption(f"Projet `{project_id}` — vecteurs : pgvector, graphe : {services.graph_store.name}")
except Exception as e:
    st.sidebar.error(f"Indexation impossible : {e}")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Carte Interactive de l'Architecture")
    st.caption("Vous pouvez zoomer, déplacer les noeuds, et les survoler pour lire le code.")
    components.html(generate_interactive_graph(graph), height=520)

with col2:
    st.markdown("### Assistant")

    # Affichage du chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Posez votre question à l'architecte..."):
        history = st.session_state.messages[-HISTORY_MESSAGES:]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = None
            with st.spinner("Recherche hybride (pgvector + BM25), reranking et expansion du graphe..."):
                try:
                    if project_id is None:
                        raise RuntimeError("projet non indexé")
                    result = services.retriever.retrieve(project_id, prompt)
                    subgraph = result.subgraph
                    for warning in result.warnings:
                        st.warning(warning)
                except Exception as e:
                    st.error(f"Erreur de recherche : {e}")
                    subgraph = graph  # Fallback sur le graphe complet

                full_prompt = build_prompt(prompt, build_graph_context(subgraph), history)
                try:
                    answer = services.generate_answer(full_prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Erreur API : {e}")

            if result and result.trace:
                with st.expander(f"Détail de la recherche ({subgraph.number_of_nodes()} noeuds dans le contexte)"):
                    st.dataframe(result.trace, hide_index=True, use_container_width=True)
