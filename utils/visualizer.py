from pyvis.network import Network

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
