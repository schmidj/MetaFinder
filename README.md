# MetaFinder AI

MetaFinder AI is an AI-powered data discovery tool that helps you find and summarize existing datasets, technical reports, and research papers on any topic. It uses Retrieval-Augmented Generation (RAG) and integrates with platforms like Zenodo for real-time metadata extraction. You can also receive SMS updates for new datasets using the MCP integration.

## Features
- **Modern Flask Web App**: Search for data sources and view AI-extracted metadata summaries.
- **RAG Pipeline**: Extracts and summarizes metadata from research papers and datasets.
- **Zenodo Integration**: Finds and analyzes real datasets and papers from Zenodo.
- **SMS Notifications**: Get notified about new datasets via MCP/Surge SMS integration.

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MetaFinder
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask Web App
```bash
python app.py
```
Visit [http://localhost:5000](http://localhost:5000) in your browser.

### 4. Run the RAG Example Scripts
- **Original RAG Example (local PDFs):**
  ```bash
  python scripts/rag_example.py
  ```
- **Zenodo RAG Example (current):**
  ```bash
  python scripts/rag_zenodo.py
  ```

### 5. MCP SMS Integration (optional)
- **Start the MCP server:**
  ```bash
  python scripts/mcp_server_example.py
  ```
- **Send a test SMS from the client:**
  ```bash
  python scripts/mcp_client_example.py
  ```

## How it Works
- Enter a research topic in the web app to discover and summarize relevant datasets and papers.
- The app fetches results from Zenodo and uses RAG to extract metadata for each result.
- Optionally, enter your phone number to get SMS updates when new datasets are published.

---

**Get started and accelerate your data-driven research with MetaFinder AI!**