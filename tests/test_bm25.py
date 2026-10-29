from graph_rag.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_splits_camel_and_snake_case():
    assert tokenize("getUserById") == ["getuserbyid", "get", "user", "by", "id"]
    assert tokenize("parse_files") == ["parse_files", "parse", "files"]


def test_tokenize_removes_accents():
    assert tokenize("Où est l'élément ?") == ["ou", "est", "l", "element"]


def test_search_ranks_exact_identifier_first():
    index = BM25Index(
        ["load_config", "save_report", "render_chart"],
        [
            "Nom: load_config\nDescription: lit le fichier de configuration",
            "Nom: save_report\nDescription: ecrit le rapport",
            "Nom: render_chart\nDescription: dessine le graphique",
        ],
    )
    results = index.search("où est load_config ?", limit=3)
    assert results[0][0] == "load_config"


def test_search_ignores_documents_without_shared_terms():
    index = BM25Index(["a", "b", "c"], ["alpha beta", "gamma delta", "epsilon zeta"])
    assert [node for node, _ in index.search("alpha", limit=5)] == ["a"]
