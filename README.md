# Wikipedia RAG System

A fully functional, locally-hosted RAG (Retrieval Augmented Generation) system using Wikipedia as knowledge base, with Ollama for LLM serving and optional MCP for agentic capabilities.

## Overview

This project demonstrates a complete end-to-end RAG pipeline that:
- Processes Wikipedia dumps into searchable chunks
- Generates semantic embeddings for intelligent retrieval
- Uses FAISS for lightning-fast vector similarity search
- Connects to local Ollama LLMs for generating contextual answers
- Provides both CLI and web interfaces for interaction


## Features

- **Wikipedia Processing**: Parse and clean 262K+ articles from Simple English Wikipedia
- **Semantic Search**: FAISS-powered vector search over 569K text chunks
- **Local LLM**: Ollama integration with Mistral/Llama2 models
- **Web Interface**: Beautiful chat UI via OpenWebUI or Flask
- **Source Attribution**: Every answer includes relevant Wikipedia citations
- **Offline Operation**: Complete system runs locally without internet
- **Memory Efficient**: Optimized for 8GB RAM with batch processing

## 🏗️ Architecture

```
Wikipedia XML → Parser → Chunker → Embeddings → FAISS Index
                                                      ↓
                                              RAG Retrieval ← User Query
                                                      ↓
                                            Context Formation
                                                      ↓
                                            Ollama LLM (Mistral)
                                                      ↓
                                              Generated Answer + Sources
```

## 📊 Current Status

**Data Processing:**
- ✅ Wikipedia parsing: 262,105 articles extracted
- ✅ Text chunking: 569,456 searchable segments
- ✅ Embeddings: 384-dim vectors using sentence-transformers
- ✅ FAISS index: 834MB, cosine similarity search

**LLM Integration:**
- ✅ Ollama setup with Mistral 7B model
- ✅ RAG pipeline connecting search to generation
- ✅ Context-aware response generation
- ✅ Source citation system

**Interfaces:**
- ✅ Interactive CLI chat
- ✅ Flask web interface
- ✅ OpenWebUI Docker integration

**Next Steps:**
- 🔄 MCP agents for autonomous reasoning
- 🔄 Scale to full English Wikipedia (6.7M articles)
- 🔄 Conversation memory and history
- 🔄 Advanced retrieval strategies

## 🚀 Quick Start

### What i'm using
- MacBook Air M1 (8GB RAM, 256GB storage)
- Python 3.9+, Docker (for OpenWebUI)
- ~5GB for Simple Wikipedia, ~150GB for full Wikipedia

### Installation

```bash
# Clone the repository
git clone https://github.com/bhavyashah10/wikipedia-rag-project.git
cd wikipedia-rag-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py
```

### Data Processing Pipeline

```bash
# 1. Download Simple English Wikipedia (~200MB)
curl -O https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
mv simplewiki-latest-pages-articles.xml.bz2 data/raw/

# 2. Parse articles from XML (~5 minutes)
python test_parser.py

# 3. Create text chunks
python src/data_processing/text_chunker.py

# 4. Generate embeddings (~15 minutes)
python src/embeddings/embedding_generator.py

# 5. Build FAISS index (~3 minutes)
python src/retrieval/faiss_indexer.py
```

### LLM Setup

```bash
# Install Ollama
brew install ollama

# Start Ollama service
brew services start ollama

# Pull Mistral model (~4GB, takes 5-10 minutes)
ollama pull mistral

# Or use Llama2
# ollama pull llama2:7b-chat
```

### Running the System

#### Option 1: CLI Chat Interface

```bash
# Start interactive chat
python src/llm_integration/rag_pipeline.py

# Example queries:
# - "What is artificial intelligence?"
# - "Explain quantum computing"
# - "Tell me about the Roman Empire"
```


#### Option 2: OpenWebUI (Recommended)

```bash
# Install and run OpenWebUI with Docker
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# Open browser to http://localhost:3000
# Login and start chatting
```

#### Option 3: Flask Web Interface (Optional - Use if not using Option 2)

```bash
# Start web server
cd src/llm_integration
python web_interface.py

# Open browser to http://localhost:5000
```

## 📁 Project Structure

```
wikipedia-rag-project/
├── data/
│   ├── raw/                    # Wikipedia XML dumps
│   ├── processed/              # 569K clean chunks (JSON)
│   └── embeddings/             # FAISS index + vectors (1.6GB)
├── src/
│   ├── data_processing/
│   │   ├── wikipedia_parser.py    # XML parsing & cleaning
│   │   └── text_chunker.py        # Semantic text chunking
│   ├── embeddings/
│   │   └── embedding_generator.py # Vector embeddings (sentence-transformers)
│   ├── retrieval/
│   │   └── faiss_indexer.py       # FAISS index & similarity search
│   ├── llm_integration/
│   │   ├── rag_pipeline.py        # Complete RAG pipeline
│   │   ├── web_interface.py       # Flask web server
│   │   └── templates/
│   │       └── chat.html          # Web UI
│   └── mcp_agents/                # MCP tools (future)
├── config/
│   └── config.yaml             # System configuration
├── logs/                       # Application logs
├── requirements.txt            # Python dependencies
├── test_setup.py              # Setup verification
├── test_parser.py             # Parser testing
└── check_setup.py             # System status check

```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

**Processing Settings:**
```yaml
processing:
  chunk_size: 1000              # Characters per chunk
  chunk_overlap: 200            # Overlap between chunks
  min_article_length: 100       # Filter short articles
```

**Embedding Settings:**
```yaml
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  normalize: true
```

**RAG Settings:**
```yaml
rag:
  top_k: 5                      # Number of chunks to retrieve
  score_threshold: 0.7          # Minimum similarity score
  max_context_length: 4000      # Max characters in context
```

**LLM Settings:**
```yaml
llm:
  model: "mistral:latest"       # Ollama model to use
  temperature: 0.7              # Response creativity
  max_tokens: 2048              # Max response length
```