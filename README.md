#  Graph RAG (Code Explorer)

Welcome to **Graph RAG**, an advanced Retrieval-Augmented Generation (RAG) application designed to explore, visualize, and understand the software architecture of your Python projects.

Unlike traditional text-based RAGs that simply chunk files into plain text, this tool reads raw source code, extracts its **Abstract Syntax Tree (AST)**, and builds a **true dependency graph** (Function Calls, Class Inheritances). 

With our recent updates, the project now integrates a **True Vector Search** directly on the graph, making it a state-of-the-art "True Graph RAG".

## Key Features

- **True Graph RAG with Embeddings**: When you ask a question, the app generates mathematical vectors (Embeddings using `text-embedding-004`) to isolate relevant components and their dependencies, providing ultra-precise context to the AI without exceeding token limits.
- **Code-to-Graph Extraction**: Parses `.py` files inside ZIP archives directly in memory (Zero-Disk Usage).
- **Interactive Visualization**: Renders an interactive, physics-based 3D map of your architecture using `PyVis`.
- **Intelligent Architectural Assistant**: An AI agent powered by **Gemini 3.5 Flash** that answers complex technical questions based on graph topology, docstrings, and code signatures.

## System Workflow: How It Works Under the Hood

1. **Ingestion & Parsing (AST)**: The user uploads a `.zip` file of a Python project. The system reads the archive directly in RAM. Using Python's native `ast` module, it deterministically identifies all Classes, Functions, and their docstrings.
2. **Graph Construction**: Each class and function becomes a "Node" in a `NetworkX` graph. Directional "Edges" are created to map inheritances and function calls. This graph is visualized interactively via `PyVis`.
3. **Vector Indexing (Embeddings)**: The system iterates through the graph nodes (name, type, and docstring) and sends them to Google GenAI (`text-embedding-004`) to generate semantic vector embeddings, which are stored in a persistent **ChromaDB** collection (one collection per project, identified by a fingerprint of the graph). A project that was already indexed is never re-embedded.
4. **Retrieval (RAG)**: When a user asks a question, the query is converted into a vector. ChromaDB runs a *cosine* nearest-neighbour search to find the Top 5 most semantically relevant nodes in the codebase.
5. **Contextual Sub-Graph Extraction**: The system isolates these Top 5 nodes along with their immediate neighbors (callers/callees) to maintain architectural logic, creating a highly focused sub-graph.
6. **AI Generation**: This targeted sub-graph is passed as context to **Gemini 3.5 Flash**. The AI analyzes this precise architectural slice and formulates an expert, context-aware answer for the user.

## Tech Stack

- **Frontend**: Streamlit
- **Graph Engine**: NetworkX
- **Visualization**: PyVis
- **AI / Embeddings / LLM**: Google GenAI SDK (Gemini 3.5 Flash & text-embedding-004)
- **Vector Database**: ChromaDB (persistent, stored in `.chroma/`)
- **Parser**: Python native `ast` module

## How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Moussa018/Graph-RAG.git
   cd Graph-RAG
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   # On Windows: venv\Scripts\activate
   # On Mac/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Add your Google API Key**
   Create a `.env` file in the root directory and add your key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```
