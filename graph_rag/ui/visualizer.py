from pyvis.network import Network


def generate_interactive_graph(g):
    """HTML autonome du graphe (JS integre : rien n'est ecrit sur disque, pas de dossier lib/)."""
    net = Network(height="500px", width="100%", bgcolor="#0E1117", font_color="white",
                  directed=True, cdn_resources="in_line")
    for node, data in g.nodes(data=True):
        color = "#ff4b4b" if data.get("type") == "Class" else "#00b4d8"
        title = f"Fichier: {data.get('file', 'inconnu')}\nDescription: {data.get('doc', 'Aucune')}"
        net.add_node(node, label=node, title=title, color=color)
    for u, v, data in g.edges(data=True):
        net.add_edge(u, v, title=data.get("relation", ""))
    return net.generate_html()
