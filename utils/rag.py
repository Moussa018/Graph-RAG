import math
import streamlit as st

@st.cache_data
def build_node_embeddings(_client, _graph):
    nodes = list(_graph.nodes(data=True))
    if not nodes:
        return {}
    
    node_names = [n[0] for n in nodes]
    texts_to_embed = []
    for node, data in nodes:
        doc = data.get("doc", "")
        node_type = data.get("type", "")
        texts_to_embed.append(f"Nom: {node}\nType: {node_type}\nDescription: {doc}")
    
    embeddings = {}
    try:
        batch_size = 100
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i+batch_size]
            response = _client.models.embed_content(
                model='text-embedding-004',
                contents=batch_texts
            )
            for j, emb in enumerate(response.embeddings):
                embeddings[node_names[i+j]] = emb.values
    except Exception as e:
        st.warning(f"Erreur embeddings: {e}")
    return embeddings

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return dot_product / (norm_v1 * norm_v2)

def retrieve_subgraph(graph, query_embedding, node_embeddings, top_k=5):
    if not node_embeddings:
        return graph
    
    similarities = []
    for node, emb in node_embeddings.items():
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((sim, node))
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_nodes = [node for sim, node in similarities[:top_k]]
    
    subgraph_nodes = set(top_nodes)
    for node in top_nodes:
        if graph.has_node(node):
            subgraph_nodes.update(graph.successors(node))
            subgraph_nodes.update(graph.predecessors(node))
            
    return graph.subgraph(subgraph_nodes)
