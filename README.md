# Graph RAG (Code Explorer)

Upload a Python project and chat with an AI that understands its architecture. The app parses your code into a dependency graph, shows it as an interactive map, and answers questions using only the most relevant part of that graph.

## Features

- **Code-to-graph**: extracts classes, functions, inheritances and function calls with Python's `ast` module. Uploaded code is processed in memory and never written to disk.
- **Interactive map**: zoomable, draggable graph of your architecture (PyVis).
- **Vector search**: every node is embedded and stored in a persistent ChromaDB collection, so a project that was already indexed is never re-embedded.
- **Architecture assistant**: Gemini answers from the retrieved sub-graph (top 5 nodes plus their callers and callees), which keeps the context small and precise.

## How it works

1. Upload `.py` files or a `.zip` archive.
2. The code is parsed into a NetworkX graph (classes, functions, `CALLS`, `INHERITS_FROM`, `HAS_METHOD`).
3. Each node (name, type, docstring) is embedded with `text-embedding-004` and stored in ChromaDB.
4. Your question is embedded and ChromaDB returns the 5 nearest nodes (cosine distance).
5. Those nodes and their direct neighbours form a sub-graph, which is sent to Gemini 3.5 Flash as context.

## Tech stack

Python, Streamlit, NetworkX, PyVis, ChromaDB, Google GenAI SDK (Gemini 3.5 Flash, `text-embedding-004`)

## Run locally

```bash
git clone https://github.com/Moussa018/Graph-RAG.git
cd Graph-RAG
python -m venv venv
# Windows: venv\Scripts\activate   |   Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Then launch the app:

```bash
streamlit run app.py
```

Embeddings are stored in `.chroma/` (change it with the optional `CHROMA_PATH` variable). Delete this folder to force a full re-indexing.

## Project structure

```
app.py                  Streamlit interface: upload, graph map, chat
utils/parser.py         AST parsing and graph construction
utils/vector_store.py   ChromaDB indexing and nearest-neighbour search
utils/rag.py            Sub-graph extraction around the retrieved nodes
utils/visualizer.py     Interactive graph rendering
```
