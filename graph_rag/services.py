from dataclasses import dataclass

from google import genai

from graph_rag.config import Settings
from graph_rag.retrieval.embeddings import GeminiEmbedder
from graph_rag.retrieval.pipeline import HybridRetriever
from graph_rag.retrieval.reranker import CrossEncoderReranker
from graph_rag.storage.base import GraphStore
from graph_rag.storage.postgres import PgVectorStore, PostgresGraphStore


@dataclass
class Services:
    """Toutes les dependances de l'application, construites une seule fois."""

    settings: Settings
    client: genai.Client
    embedder: GeminiEmbedder
    vector_store: PgVectorStore
    graph_store: GraphStore  # Neo4jGraphStore ou PostgresGraphStore
    retriever: HybridRetriever

    def generate_answer(self, prompt):
        response = self.client.models.generate_content(model=self.settings.llm_model, contents=prompt)
        return response.text


def build_graph_store(settings):
    if settings.graph_backend == "neo4j":
        from graph_rag.storage.neo4j_graph import Neo4jGraphStore

        return Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password,
                               settings.neo4j_database)
    return PostgresGraphStore(settings.database_url, settings.embed_dim)


def build_services(settings):
    client = genai.Client(api_key=settings.google_api_key)
    embedder = GeminiEmbedder(client, settings.embed_model)
    vector_store = PgVectorStore(settings.database_url, settings.embed_dim)
    graph_store = build_graph_store(settings)
    reranker = CrossEncoderReranker(settings.reranker_model)
    retriever = HybridRetriever(settings, embedder, vector_store, graph_store, reranker)
    return Services(settings, client, embedder, vector_store, graph_store, retriever)
