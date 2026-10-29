import hashlib
from dataclasses import dataclass


def node_text(node, data):
    """Texte indexe pour un noeud : sert a l'embedding, a BM25 et au reranker."""
    parts = [
        f"Nom: {node}",
        f"Type: {data.get('type') or 'External'}",
    ]
    if data.get("file"):
        parts.append(f"Fichier: {data['file']}")
    parts.append(f"Description: {data.get('doc') or 'Aucune doc'}")
    if data.get("source"):
        parts.append(f"Code:\n{data['source']}")
    return "\n".join(parts)


def graph_fingerprint(graph):
    """Empreinte des noeuds et des aretes : un meme projet reutilise le meme project_id."""
    digest = hashlib.sha1()
    for node, data in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
        digest.update(node_text(node, data).encode("utf-8"))
        digest.update(b"\0")
    for u, v, data in sorted(graph.edges(data=True), key=lambda e: (str(e[0]), str(e[1]))):
        digest.update(f"{u}\0{v}\0{data.get('relation', '')}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


@dataclass
class IndexReport:
    project_id: str
    vectors_written: bool  # False : embeddings deja presents dans pgvector
    graph_written: bool    # False : graphe deja present dans le graph store


def index_project(graph, embedder, vector_store, graph_store):
    """Indexe le projet dans les deux stores, chacun seulement s'il ne l'a pas deja.

    Les vecteurs sont ecrits en premier : le backend graphe Postgres lit les
    attributs des noeuds dans la table des vecteurs.
    """
    project_id = graph_fingerprint(graph)

    vectors_written = False
    if not vector_store.has_project(project_id):
        nodes = list(graph.nodes(data=True))
        texts = [node_text(node, data) for node, data in nodes]
        vectors = embedder.embed_documents(texts)
        vector_store.write_nodes(project_id, nodes, texts, vectors)
        vectors_written = True

    graph_written = False
    if not graph_store.has_project(project_id):
        graph_store.write_graph(project_id, graph)
        graph_written = True

    return IndexReport(project_id, vectors_written, graph_written)
