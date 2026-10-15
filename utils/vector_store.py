import hashlib
import os

import chromadb
import streamlit as st

CHROMA_PATH = os.environ.get("CHROMA_PATH", ".chroma")
EMBED_MODEL = "text-embedding-004"
BATCH_SIZE = 100


@st.cache_resource
def get_chroma_client():
    """Client ChromaDB persistant : les embeddings survivent au redemarrage de l'app."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _node_text(node, data):
    doc = data.get("doc", "")
    node_type = data.get("type", "")
    return f"Nom: {node}\nType: {node_type}\nDescription: {doc}"


def graph_fingerprint(graph):
    """Empreinte du contenu du graphe : un meme projet reutilise la meme collection."""
    digest = hashlib.sha1()
    for node, data in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
        digest.update(_node_text(node, data).encode("utf-8"))
    return digest.hexdigest()[:16]


def get_or_build_collection(genai_client, graph):
    """Retourne la collection ChromaDB du graphe, en l'indexant si necessaire.

    Si le projet a deja ete indexe (meme empreinte), aucun appel a l'API
    d'embeddings n'est refait.
    Retourne None si l'indexation echoue (l'appelant retombe sur le graphe complet).
    """
    nodes = list(graph.nodes(data=True))
    if not nodes:
        return None

    chroma = get_chroma_client()
    collection = chroma.get_or_create_collection(
        name=f"codegraph_{graph_fingerprint(graph)}",
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == len(nodes):
        return collection

    ids = [str(node) for node, _ in nodes]
    texts = [_node_text(node, data) for node, data in nodes]
    metadatas = [{"type": data.get("type") or "External"} for _, data in nodes]

    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            response = genai_client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch_texts,
            )
            collection.upsert(
                ids=ids[i:i + BATCH_SIZE],
                embeddings=[list(emb.values) for emb in response.embeddings],
                documents=batch_texts,
                metadatas=metadatas[i:i + BATCH_SIZE],
            )
    except Exception as e:
        st.warning(f"Erreur embeddings: {e}")
        chroma.delete_collection(collection.name)
        return None
    return collection


def search_nodes(collection, query_embedding, top_k=5):
    """Les top_k noeuds les plus proches de la question (distance cosinus)."""
    result = collection.query(
        query_embeddings=[list(query_embedding)],
        n_results=min(top_k, collection.count()),
    )
    return result["ids"][0]
