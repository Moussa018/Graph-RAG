import psycopg
from pgvector.psycopg import register_vector

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

-- Vecteurs (PgVectorStore)
CREATE TABLE IF NOT EXISTS code_projects (
    project_id  TEXT PRIMARY KEY,
    node_count  INTEGER NOT NULL,
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS code_nodes (
    project_id  TEXT NOT NULL REFERENCES code_projects ON DELETE CASCADE,
    node_id     TEXT NOT NULL,
    node_type   TEXT NOT NULL,
    file        TEXT,
    doc         TEXT,
    source      TEXT,
    content     TEXT NOT NULL,
    embedding   vector({embed_dim}) NOT NULL,
    PRIMARY KEY (project_id, node_id)
);

CREATE INDEX IF NOT EXISTS code_nodes_embedding_hnsw
    ON code_nodes USING hnsw (embedding vector_cosine_ops);

"""

_initialized_urls = set()


def _ensure_schema(database_url, embed_dim):
    """Cree l'extension pgvector et les tables (une fois par processus et par base)."""
    if database_url in _initialized_urls:
        return
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(SCHEMA_SQL.format(embed_dim=int(embed_dim)))
    _initialized_urls.add(database_url)


class PgVectorStore:
    """Noeuds et embeddings dans Postgres, recherche par index HNSW (distance cosinus)."""

    def __init__(self, database_url, embed_dim):
        self.database_url = database_url
        self.embed_dim = embed_dim

    def _connect(self):
        _ensure_schema(self.database_url, self.embed_dim)
        conn = psycopg.connect(self.database_url, autocommit=True)
        register_vector(conn)
        return conn

    def has_project(self, project_id):
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM code_projects WHERE project_id = %s", (project_id,)
            ).fetchone() is not None

    def write_nodes(self, project_id, nodes, texts, vectors):
        """Ecrit tous les noeuds en une transaction : un projet present dans code_projects est complet."""
        rows = [
            (project_id, str(node), data.get("type") or "External", data.get("file"),
             data.get("doc"), data.get("source"), text, vector)
            for (node, data), text, vector in zip(nodes, texts, vectors)
        ]
        with self._connect() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO code_projects (project_id, node_count) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (project_id, len(rows)),
            )
            cur.executemany(
                "INSERT INTO code_nodes (project_id, node_id, node_type, file, doc, source, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                rows,
            )

    def dense_search(self, project_id, query_vector, limit):
        with self._connect() as conn:
            try:
                # pgvector >= 0.8 : continue le parcours HNSW tant que le filtre project_id
                # n'a pas fourni assez de resultats.
                conn.execute("SET hnsw.iterative_scan = strict_order")
            except psycopg.Error:
                pass
            rows = conn.execute(
                "SELECT node_id, 1 - (embedding <=> %(q)s) FROM code_nodes "
                "WHERE project_id = %(p)s ORDER BY embedding <=> %(q)s LIMIT %(k)s",
                {"q": query_vector, "p": project_id, "k": limit},
            ).fetchall()
        return [(node_id, float(similarity)) for node_id, similarity in rows]

    def load_corpus(self, project_id):
        with self._connect() as conn:
            return conn.execute(
                "SELECT node_id, content FROM code_nodes WHERE project_id = %s ORDER BY node_id",
                (project_id,),
            ).fetchall()

