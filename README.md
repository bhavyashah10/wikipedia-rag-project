# Wikipedia RAG System

A locally-hosted RAG (Retrieval Augmented Generation) system using Wikipedia as knowledge base, with Ollama for LLM serving and MCP for agentic capabilities.

## 🎯 Project Goals

- **Download & Process**: Full Wikipedia dump (~50GB) → Clean, searchable text chunks
- **Vector Search**: FAISS-powered semantic search over Wikipedia content
- **Local LLM**: Ollama + OpenWebUI for private, offline AI assistance  
- **Agentic Tools**: MCP (Model Context Protocol) for autonomous reasoning and tool use
- **Self-Contained**: Complete knowledge system running locally on MacBook Air M1

## 🏗️ Architecture

```
Wikipedia XML → Parser → Chunker → Embeddings → FAISS Index
                                                      ↓
                                              RAG Retrieval
                                                      ↓
                                            Ollama LLM + MCP Agents
```

## 📊 Current Progress

- ✅ **Wikipedia Download**: Simple English Wikipedia (200MB → 800MB uncompressed)
- ✅ **XML Parsing**: 262,105 articles extracted and cleaned
- ✅ **Text Chunking**: 569,456 chunks created (avg 2.2 chunks/article)
- ✅ **Embedding Generation**: 569K chunks → 384-dim vectors (834MB)
- ✅ **FAISS Index**: Semantic search over 569K vectors (sub-second queries)
- 🔄 **Next**: Ollama LLM integration for RAG conversations
- 🔄 **Next**: OpenWebUI web interface
- 🔄 **Next**: MCP agentic layer

## 🚀 Quick Start

### Prerequisites
- MacBook Air M1 (8GB RAM, 256GB+ storage)
- Python 3.9+
- ~150GB free space for full Wikipedia (current test uses ~5GB)

### Setup
```bash
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

### Run Wikipedia Processing
```bash
# Download Simple English Wikipedia (for testing)
curl -O https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
mv simplewiki-latest-pages-articles.xml.bz2 data/raw/

# Parse articles from XML
python test_parser.py

# Create text chunks
python src/data_processing/text_chunker.py

# Generate embeddings (10-15 minutes)
python src/embeddings/embedding_generator.py

# Build FAISS index for search
python src/retrieval/faiss_indexer.py
```

### Test the RAG System
```bash
# Test semantic search
python src/retrieval/faiss_indexer.py

# Example queries that work:
# - "What is artificial intelligence?"
# - "How do computers work?" 
# - "Tell me about space exploration"
# - "What is machine learning?"
```

## 📁 Project Structure

```
wikipedia-rag-project/
├── data/
│   ├── raw/                    # Wikipedia XML dumps
│   ├── processed/              # Clean articles & chunks (569K chunks)
│   └── embeddings/             # FAISS indices & vectors (1.6GB)
├── src/
│   ├── data_processing/        # XML parser & text chunker
│   ├── embeddings/             # Embedding generation (sentence-transformers)
│   ├── retrieval/              # FAISS search & RAG
│   ├── llm_integration/        # Ollama interface (coming next)
│   └── mcp_agents/             # MCP tools & agents (coming next)
├── config/                     # YAML configurations
├── notebooks/                  # Jupyter experiments
└── logs/                       # Application logs
```

## ⚙️ Configuration

See `config/config.yaml` for:
- Processing parameters (chunk size, overlap, filters)
- Embedding model settings (sentence-transformers)
- FAISS index configuration
- LLM settings (Ollama models)
- RAG retrieval parameters

## 🧪 Current Test Results

**Simple English Wikipedia Processing:**
- **Input**: 200MB compressed XML
- **Parsed**: 262,105 articles
- **Chunks**: 569,456 text segments
- **Average**: 2.2 chunks per article
- **Processing Time**: ~5 minutes on M1 MacBook Air

**Embedding Generation:**
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vectors**: 569,456 × 384 dimensions
- **Size**: 834MB embeddings + 132MB metadata
- **Processing Time**: ~15 minutes on M1 MacBook Air

**FAISS Search Performance:**
- **Index Size**: 834MB (cosine similarity, flat index)
- **Search Speed**: Sub-second queries over 569K vectors
- **Quality**: Excellent semantic relevance for test queries
- **Memory Usage**: ~2GB RAM during search

## 🛠️ Tech Stack

- **Data Processing**: Python, lxml, BeautifulSoup
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS (CPU-optimized for M1)
- **LLM Serving**: Ollama (Llama2-7B, Mistral-7B)
- **Web Interface**: OpenWebUI
- **Agents**: MCP (Model Context Protocol)
- **Hardware**: Optimized for Apple Silicon M1

## 📈 Next Steps

1. **Ollama Integration** - Local LLM serving (Llama2-7B, Mistral-7B)
2. **RAG Pipeline** - Connect FAISS search to LLM responses
3. **OpenWebUI** - Web-based chat interface
4. **MCP Agents** - Tool use and autonomous reasoning
5. **Scale to Full Wikipedia** - Process complete English Wikipedia (~6.7M articles)

## 💡 Example Search Results

The system already demonstrates excellent semantic search:

**Query: "What is artificial intelligence?"**
- ✅ Returns actual AI definition articles
- ✅ Similarity scores: 0.540-0.640 (high relevance)
- ✅ Sub-second response time

**Query: "How do computers work?"**  
- ✅ Finds computer architecture explanations
- ✅ Returns technical details about processors, circuits
- ✅ Perfect semantic matching (not just keyword search)

## 🔧 Hardware Considerations

**Current (Simple Wiki + RAG Search):**
- Storage: ~5GB total (chunks + embeddings + index)
- RAM: ~2GB during search, ~4GB during embedding generation
- Processing: ~20 minutes total setup time

**Full Wikipedia (Estimated):**
- Storage: ~120-150GB total  
- RAM: ~4-6GB during processing
- Processing: ~2-4 hours total setup time
