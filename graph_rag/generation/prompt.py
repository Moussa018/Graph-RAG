SYSTEM_INSTRUCTIONS = """Tu es un architecte logiciel IA expert. Réponds au développeur uniquement à partir de l'extrait de l'architecture ci-dessous (sous-graphe RAG).

Règles :
- Cite les noms exacts des fonctions/classes et leur fichier (ex: `CodeVisitor.visit_Call` dans `graph_rag/ingestion/parser.py`).
- Appuie-toi sur le code source fourni et sur les relations (CALLS, INHERITS_FROM, HAS_METHOD) pour expliquer les flux.
- Si le contexte ne suffit pas pour répondre, dis-le explicitement et indique quel élément manque au lieu d'inventer."""


def build_graph_context(subgraph):
    """Contexte envoye au LLM : code des seeds, resume des voisins, puis relations.

    Un noeud sans attribut `depth` (graphe complet en secours) n'est que resume,
    pour ne pas envoyer le code de tout le projet.
    """
    seeds = [(n, d) for n, d in subgraph.nodes(data=True) if d.get("depth") == 0]
    neighbours = [(n, d) for n, d in subgraph.nodes(data=True) if d.get("depth") != 0]

    context = ""
    if seeds:
        context += "### Noeuds les plus pertinents (retenus par le reranker)\n"
    for n, data in seeds:
        context += f"\n#### [{data.get('type') or 'Node'}] {n} — {data.get('file') or 'externe'}\n"
        context += f"Docstring : {data.get('doc') or 'Aucune doc'}\n"
        if data.get("source"):
            context += f"```python\n{data['source']}\n```\n"

    if neighbours:
        context += "\n### Autres noeuds (appelants, appelés, classes parentes, méthodes)\n"
        for n, data in neighbours:
            context += f"- [{data.get('type') or 'Node'}] {n} ({data.get('file') or 'externe'}) : {data.get('doc') or 'Aucune doc'}\n"

    context += "\n### Relations dans ce contexte\n"
    for u, v, data in subgraph.edges(data=True):
        context += f"- '{u}' {data.get('relation', 'CALLS')} '{v}'\n"
    return context


def build_prompt(question, graph_context, history):
    """history : [{"role", "content"}] des messages precedents (sans la question courante)."""
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history) or "(vide)"
    return f"""{SYSTEM_INSTRUCTIONS}

Contexte RAG :
{graph_context}

Historique du chat :
{history_text}

Question : {question}

Réponds de façon concise et précise."""
