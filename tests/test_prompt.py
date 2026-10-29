import networkx as nx

from graph_rag.generation.prompt import build_graph_context, build_prompt


def test_context_includes_seed_source_and_neighbour_summary():
    subgraph = nx.DiGraph()
    subgraph.add_node("load", type="Function", file="io.py", doc="Charge", source="def load(): ...", depth=0)
    subgraph.add_node("open", type="External", file=None, doc=None, source=None, depth=1)
    subgraph.add_edge("load", "open", relation="CALLS")

    context = build_graph_context(subgraph)

    assert "#### [Function] load — io.py" in context
    assert "def load(): ..." in context
    assert "- [External] open (externe) : Aucune doc" in context
    assert "- 'load' CALLS 'open'" in context


def test_full_graph_fallback_does_not_send_source_code():
    graph = nx.DiGraph()
    graph.add_node("load", type="Function", file="io.py", doc="Charge", source="def load(): ...")
    assert "def load()" not in build_graph_context(graph)


def test_prompt_contains_question_and_history():
    prompt = build_prompt("Que fait load ?", "ctx", [{"role": "user", "content": "Bonjour"}])
    assert "Question : Que fait load ?" in prompt
    assert "user: Bonjour" in prompt
