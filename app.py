import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from google import genai
import os
from dotenv import load_dotenv

from utils.parser import process_files_to_graph
from utils.rag import build_node_embeddings, retrieve_subgraph
from utils.visualizer import generate_interactive_graph

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
        graph = process_files_to_graph(uploaded_files)
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
                
        if prompt := st.chat_input("Posez votre question à l'architecte..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Recherche RAG dans le graphe..."):
                    # Récupération ou calcul des embeddings
                    node_embeddings = build_node_embeddings(client, graph)
                    
                    try:
                        # Embedding de la question
                        query_response = client.models.embed_content(
                            model='text-embedding-004',
                            contents=prompt
                        )
                        query_embedding = query_response.embeddings[0].values
                        subgraph = retrieve_subgraph(graph, query_embedding, node_embeddings)
                    except Exception as e:
                        st.error(f"Erreur d'embedding: {e}")
                        subgraph = graph # Fallback sur le graphe complet
                        
                    # Préparation du contexte du sous-graphe pertinent
                    graph_context = "### Noeuds pertinents (avec descriptions):\n"
                    for n, data in subgraph.nodes(data=True):
                        graph_context += f"- [{data.get('type', 'Node')}] {n} : {data.get('doc', 'Aucune doc')}\n"
                    
                    graph_context += "\n### Relations dans ce contexte:\n"
                    for u, v, data in subgraph.edges(data=True):
                        graph_context += f"- '{u}' {data.get('relation', 'appels')} '{v}'\n"
                        
                    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
                    
                    full_prompt = f"""
                    Tu es un architecte logiciel IA expert. Réponds au développeur en t'appuyant sur l'extrait de l'architecture ci-dessous (sous-graphe RAG).
                    
                    Contexte RAG :
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
