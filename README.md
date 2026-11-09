# Graph RAG (Code Explorer)

Upload a Python project and chat with an AI that understands its architecture. The app parses your code into a dependency graph. It stores the graph in **Neo4j** and the embeddings in **PostgreSQL + pgvector**. Each question is answered from a sub-graph chosen in four steps:

1. Hybrid search: pgvector + BM25.
2. Fusion of the two rankings with Reciprocal Rank Fusion (RRF).
3. A cross-encoder reranker.
4. A graph walk in Neo4j.

## Features

- **Code-to-graph**: Python's `ast` module extracts classes, functions and methods, together with their docstring, file and source code (first 1,500 characters). It also records `CALLS`, `INHERITS_FROM` and `HAS_METHOD` relations. Uploaded code is processed in memory and never written to disk.
- **Graph in Neo4j**: typed nodes (`:CodeNode:Class`, `:Function`, `:External`) and typed relations. Graph expansion is a Cypher variable-length path. The graph can be explored in Neo4j Browser.
- **Vectors in pgvector**: 768-dimension Gemini embeddings with an HNSW index (`vector_cosine_ops`).
- **Hybrid retrieval**: dense search (pgvector) and lexical search (BM25 Okapi, with code-aware tokenisation) are merged with RRF.
- **Reranker**: a multilingual cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) keeps the 5 most relevant nodes.
- **Graph in Neo4j**: nodes and relations (calls, methods, inheritance) are stored and walked in Neo4j.
- **Incremental indexing**: each project is identified by a fingerprint of its nodes and edges. Each store is only written if it doesn't already have the project.
- **Retrieval trace**: under each answer, a table shows every candidate's pgvector rank, BM25 rank, RRF score and reranker score.

## Architecture

```
                 ┌──────────── ingestion ────────────┐
.py / .zip ──► parser.py (AST) ──► NetworkX graph ──► indexer.py
                                                      │        │
                                     Gemini embeddings│        │nodes + relations
                                                      ▼        ▼
                                          PostgreSQL+pgvector  Neo4j
                                          (code_nodes, HNSW)   (:CodeNode)-[:CALLS]->(...)
                                                      │        │
                 ┌──────────── retrieval ─────────────┼────────┼──┐
question ──► pgvector top 20 ─┐                       │        │  │
         └─► BM25 top 20 ─────┴─► RRF ─► cross-encoder top 5 ──┴─► sub-graph (1 hop)
                 └────────────────────────────────────────────────┘
                                                      │
                                    generation ──► prompt.py ──► Gemini 3.5 Flash
```

Why each stage:

| Stage | Catches | Misses on its own |
|---|---|---|
| pgvector (dense) | Intent and paraphrases ("where are files read?" → `iter_python_sources`) | Exact identifiers and rare tokens |
| BM25 | Exact names: `visit_Call`, `MAX_SOURCE_CHARS`, `getUserById` | Synonyms and phrasing that differs from the code |
| RRF | Merges both without calibrating their scores (cosine and BM25 are on different scales) | Fine-grained relevance |
| Cross-encoder | Reads the question and the code together, so it ranks most precisely | Too slow to run on the whole project, so it only sees the RRF candidates |
| Graph walk (Neo4j) | Context the question doesn't mention: who calls the function, what it calls, parent class | — |

BM25 tokenisation is tailored to code. Accents are removed, and each identifier is kept whole and also split on `snake_case` and `camelCase`. For example, `getUserById` gives `getuserbyid`, `get`, `user`, `by`, `id`.

If the reranker cannot be loaded, the app falls back to the RRF ranking and shows a warning. If the whole search fails, the full graph is sent to the LLM as summaries, without source code.

## Data model

**Neo4j**

```cypher
(:CodeNode:Function {uid, project_id, node_id, type, file, doc, source})
(:CodeNode:Class)-[:HAS_METHOD]->(:CodeNode:Function)
(:CodeNode)-[:CALLS]->(:CodeNode)
(:CodeNode:Class)-[:INHERITS_FROM]->(:CodeNode:Class)
(:CodeProject {project_id, node_count, edge_count, indexed_at})   // written last: project complete
```

`uid = "<project_id>:<node_id>"` has a uniqueness constraint, and `project_id` is indexed. Here is an example query to run in Neo4j Browser (http://localhost:7474):

```cypher
MATCH (f:Function {node_id: 'build_graph'})-[:CALLS*1..2]->(g) RETURN f, g
```

**PostgreSQL**

| Table | Contents |
|---|---|
| `code_projects` | Projects whose vectors are indexed |
| `code_nodes` | `node_id`, `node_type`, `file`, `doc`, `source`, `content` (indexed text), `embedding vector(768)` + HNSW index |

Each write (vectors, Neo4j graph) is a single transaction, and the project marker is written in that same transaction. A store that lists a project therefore always holds it in full. Tables and constraints are created automatically.

## Run locally

Requirements: Python 3.10+, Docker.

```bash
git clone https://github.com/Moussa018/Graph-RAG.git
cd Graph-RAG
docker compose up -d          # PostgreSQL 17 + pgvector 0.8 (5432), Neo4j 5 (7474, 7687)
python -m venv venv
# Windows: venv\Scripts\activate   |   Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set GOOGLE_API_KEY
streamlit run app.py
```

The reranker (~470 MB) is downloaded from Hugging Face on the first question, then cached.

To re-index from scratch, run `docker compose down -v`.

## Configuration

All variables are listed in [`.env.example`](.env.example) and read by `graph_rag/config.py`.

| Variable | Default | Role |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini key (required) |
| `DATABASE_URL` | `postgresql://graphrag:graphrag@localhost:5432/graphrag` | pgvector |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` / `neo4j` / `graphrag-password` | Neo4j |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder |
| `DENSE_K` / `BM25_K` | 20 / 20 | Candidates from each search |
| `RRF_K` | 60 | RRF constant (higher values flatten the weight of top ranks) |
| `RERANK_CANDIDATES` | 20 | Pairs scored by the cross-encoder |
| `TOP_K` | 5 | Seeds kept after reranking |
| `GRAPH_DEPTH` | 1 | Hops of graph expansion (2+ can pull in many nodes through hubs such as `append`) |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The unit tests cover parsing, BM25, RRF, idempotent indexing, the retrieval pipeline (including the fallback when the reranker fails) and prompt building. They use in-memory fakes and need neither Postgres, Neo4j nor the Gemini API.

## Project structure

```
app.py                          Streamlit interface (UI only)
docker-compose.yml              PostgreSQL + pgvector, Neo4j
.env.example                    All configuration variables
graph_rag/
  config.py                     Settings read from the environment
  services.py                   Builds the dependencies (stores, embedder, reranker, retriever)
  ingestion/
    parser.py                   AST → NetworkX graph (doc + source code per node)
    indexer.py                  Fingerprint, indexed text, idempotent indexing into both stores
  storage/
    base.py                     VectorStore / GraphStore interfaces
    postgres.py                 PgVectorStore (pgvector)
    neo4j_graph.py              Neo4jGraphStore (Cypher)
  retrieval/
    embeddings.py               Gemini embeddings (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)
    bm25.py                     Code-aware tokenisation + BM25 Okapi
    fusion.py                   Reciprocal Rank Fusion
    reranker.py                 Cross-encoder
    pipeline.py                 HybridRetriever: dense + BM25 → RRF → reranker → graph
  generation/
    prompt.py                   LLM context and prompt
  ui/
    visualizer.py               Interactive PyVis map (self-contained HTML)
tests/                          Unit tests (pytest)
```

## Tech stack

Python, Streamlit, NetworkX, PyVis, Neo4j 5, PostgreSQL 17 + pgvector 0.8, psycopg 3, rank-bm25, sentence-transformers, Google GenAI SDK (Gemini 3.5 Flash, `text-embedding-004`)
