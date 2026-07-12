#  Graph RAG (Code Explorer)

This project is an advanced **Graph RAG (Retrieval-Augmented Generation)** application designed to explore and understand software architecture. 
Unlike traditional text-based RAGs, this tool reads raw Python source code, extracts its Abstract Syntax Tree (AST), and builds a real-time **dependency graph** (Function Calls, Class Inheritances).

## Features
- **Code-to-Graph Extraction**: Parses `.py` files inside ZIP archives directly in memory.
- **Interactive Visualization**: Renders an interactive, physics-based 3D graph of the codebase architecture using `PyVis`.
- **Intelligent Architectural Assistant**: Uses Google's **Gemini 3.5 Flash** to answer complex reverse-engineering questions based purely on the graph topology and docstrings.
- **Zero-Disk Usage**: The ZIP code is unzipped and parsed on the fly in RAM.

## Tech Stack
- **Frontend**: Streamlit, PyVis
- **Graph Engine**: NetworkX
- **AI / LLM**: Google GenAI SDK (Gemini 3.5 Flash)
- **Parser**: Python `ast` module

## How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/Moussa018/Graph-RAG.git
   cd Graph-RAG
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Add your API Key**
   Create a `.env` file in the root directory and add your Google API Key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Launch the App**
   ```bash
   streamlit run app.py
   ```
