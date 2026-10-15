from utils.vector_store import search_nodes


def retrieve_subgraph(graph, query_embedding, collection, top_k=5):
    """Sous-graphe forme des top_k noeuds les plus proches (base vectorielle) et de leurs voisins."""
    if collection is None:
        return graph

    top_nodes = [node for node in search_nodes(collection, query_embedding, top_k) if graph.has_node(node)]

    subgraph_nodes = set(top_nodes)
    for node in top_nodes:
        subgraph_nodes.update(graph.successors(node))
        subgraph_nodes.update(graph.predecessors(node))

    return graph.subgraph(subgraph_nodes)
