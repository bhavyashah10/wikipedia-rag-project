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
- 🔄 **Next**: Embedding generation with sentence-transformers
- 🔄 **Next**: FAISS index creation
- 🔄 **Next**: Ollama LLM integration
- 🔄 **Next**: MCP agentic layer

## 🚀 Quick Start

### Prerequisites
- MacBook Air M1 (8GB RAM, 256GB+ storage)
- Python 3.9+
- ~150GB free space for full Wikipedia (current test uses ~5GB)

### Setup
```bash
git clone <this-repo>
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
```

## 📁 Project Structure

```
wikipedia-rag-project/
├── data/
│   ├── raw/                    # Wikipedia XML dumps
│   ├── processed/              # Clean articles & chunks
│   └── embeddings/             # FAISS indices & vectors
├── src/
│   ├── data_processing/        # XML parser & text chunker
│   ├── embeddings/             # Embedding generation
│   ├── retrieval/              # FAISS search & RAG
│   ├── llm_integration/        # Ollama interface
│   └── mcp_agents/             # MCP tools & agents
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

## 🛠️ Tech Stack

- **Data Processing**: Python, lxml, BeautifulSoup
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS (CPU-optimized for M1)
- **LLM Serving**: Ollama (Llama2-7B, Mistral-7B)
- **Web Interface**: OpenWebUI
- **Agents**: MCP (Model Context Protocol)
- **Hardware**: Optimized for Apple Silicon M1

## 📈 Next Steps

1. **Embedding Generation** - Convert chunks to 384-dim vectors
2. **FAISS Index** - Build searchable vector database  
3. **RAG Pipeline** - Connect retrieval to LLM
4. **Ollama Integration** - Local model serving
5. **MCP Agents** - Tool use and reasoning
6. **Scale to Full Wikipedia** - Process complete English Wikipedia

## 🔧 Hardware Considerations

**Current (Simple Wiki):**
- Storage: ~5GB total
- RAM: ~2GB during processing
- Processing: ~5 minutes

**Full Wikipedia (Estimated):**
- Storage: ~120-150GB total  
- RAM: ~4-6GB during processing
- Processing: ~2-4 hours

## 📝 License

MIT License - See LICENSE file for details.