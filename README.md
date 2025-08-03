# RAG for LLMs: Production-Ready Implementation

A comprehensive, modular Retrieval-Augmented Generation (RAG) system built from scratch with advanced features, optimizations, and production-ready capabilities.

## 🚀 Key Features

### Core RAG Capabilities
- **Modular Architecture**: Factory pattern for easy component swapping
- **Multiple Vector Stores**: ChromaDB (default) and FAISS support
- **Advanced Text Processing**: Fixed-size and sentence-based chunking with overlap
- **Smart Embeddings**: SentenceTransformer with built-in caching for efficiency
- **Local LLM Support**: Ollama integration with lightweight models (gemma:2b)
- **Document Support**: PDF, TXT, and Markdown file processing

### Advanced Features
- **Query Optimization**: 
  - LLM-based query rewriting and expansion
  - Query decomposition for complex questions
  - Hypothetical Document Embeddings (HyDE)
- **Reranking**: Cross-encoder models for improved relevance
- **Hybrid Search**: Combines vector similarity with keyword matching
- **Context Compression**: Extractive summarization for longer contexts
- **Conversation Memory**: Multi-turn dialogue support
- **Performance Monitoring**: Built-in metrics and optimization tools

### Optimizations
- **Low RAM Usage**: Configured with gemma:2b (1.7GB) via Ollama
- **Embedding Cache**: Reduces redundant computations
- **Batch Processing**: Efficient document indexing
- **GPU Acceleration**: Optional CUDA support for embeddings
- **Dynamic Chunking**: Adaptive chunk sizes based on content

## 📋 System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Ollama installed for local LLM inference
- 5GB disk space for models and vector store

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rahulmehta25/Retrieval-Augmented-Generation-for-LLMs-A-Survey.git
cd "RAG for LLMs-  A Survey/rag-from-scratch"
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull the lightweight model
ollama pull gemma:2b
```

## 🚀 Quick Start

### Basic Usage
```python
from src.rag.naive_rag import NaiveRAG

# Initialize RAG system
rag = NaiveRAG('config.yaml')

# Index documents
rag.index_documents(['documents/doc1.pdf', 'documents/doc2.txt'])

# Query the system
answer = rag.query("What are the key concepts in this document?")
print(answer)
```

### Ollama Demo (Recommended)
```bash
# Run the interactive Ollama demo
python ollama_demo.py
```

This demo:
- Uses the lightweight gemma:2b model
- Provides an interactive question-answering interface
- Includes sample documents about AI, ML, and RAG
- Shows real-time responses with low memory usage

## 📁 Project Structure

```
rag-from-scratch/
├── src/                      # Core implementation
│   ├── chunking/            # Text splitting strategies
│   │   └── text_splitter.py # Fixed-size and sentence splitters
│   ├── embedding/           # Embedding generation
│   │   └── embedder.py      # SentenceTransformer wrapper
│   ├── retrieval/           # Vector stores and search
│   │   ├── vector_store.py  # ChromaDB and FAISS implementations
│   │   ├── reranker.py      # Cross-encoder reranking
│   │   └── hybrid_search.py # Dense + sparse retrieval
│   ├── generation/          # LLM integration
│   │   └── generator.py     # Ollama, OpenAI, HuggingFace
│   └── rag/                 # RAG pipelines
│       ├── naive_rag.py     # Basic implementation
│       ├── advanced_rag.py  # Advanced features
│       └── rag_factory.py   # Component factory
├── config.yaml              # System configuration
├── requirements.txt         # Python dependencies
├── ollama_demo.py          # Interactive demo
└── frontend_requirements.md # Frontend integration guide
```

## ⚙️ Configuration

The system is configured via `config.yaml`:

```yaml
# Text Processing
text_splitter:
  type: fixed_size
  chunk_size: 500
  chunk_overlap: 50

# Embeddings
embedder:
  type: sentence_transformer
  model_name: all-MiniLM-L6-v2
  cache_dir: ./embedding_cache

# Vector Storage
vector_store:
  type: chromadb
  path: ./chroma_db
  collection_name: rag_collection

# Generation (Ollama)
generator:
  type: ollama
  model_name: gemma:2b  # Lightweight model
  host: localhost
  port: 11434
```

## 🧠 Available Models

### Ollama Models (Local)
| Model | RAM Usage | Description |
|-------|-----------|-------------|
| gemma:2b | ~1.7GB | Lightweight, fast responses |
| phi | ~2.7GB | Microsoft's efficient model |
| tinyllama | ~637MB | Ultra-light for basic tasks |
| llama2:7b | ~3.8GB | Meta's Llama 2 |
| mistral:7b | ~4.1GB | High quality, efficient |

### Alternative Backends
- OpenAI API (gpt-3.5-turbo, gpt-4)
- HuggingFace models (distilgpt2, etc.)

## 🔧 Advanced Features

### Query Optimization Strategies
```python
from src.rag.advanced_rag import AdvancedRAG

rag = AdvancedRAG('config.yaml')

# Query expansion
answer = rag.query_optimized(
    "What is RAG?", 
    query_optimization_strategy="expansion"
)

# Query decomposition for complex questions
answer = rag.query_optimized(
    "Compare RAG and fine-tuning approaches", 
    query_optimization_strategy="decomposition"
)

# Hypothetical Document Embeddings
answer = rag.query_optimized(
    "How does vector search work?", 
    query_optimization_strategy="hyde"
)
```

### Reranking for Better Results
```python
# Enable reranking in config
rag.enable_reranking = True
answer = rag.query("Your question", k=20, rerank_top_k=5)
```

### Hybrid Search
```python
# Combines vector similarity with BM25 keyword search
rag.use_hybrid_search = True
answer = rag.query("Your question", alpha=0.5)  # 0.5 = equal weight
```

## 🎯 Performance Optimizations

1. **Embedding Cache**: Automatically caches embeddings to avoid recomputation
2. **Batch Processing**: Processes multiple documents efficiently
3. **Low Memory Models**: Default configuration uses gemma:2b (1.7GB RAM)
4. **Local Execution**: No API calls, all processing done locally
5. **Optimized Chunking**: Intelligent text splitting preserves context

## 🌐 Frontend Integration

See `frontend_requirements.md` for detailed specifications to build a web interface using frameworks like React or Vue.js. The system is designed to be easily wrapped in a Flask/FastAPI backend.

## 📊 Benchmarks

| Operation | Time | Memory |
|-----------|------|--------|
| Document Indexing (10 docs) | ~5s | 200MB |
| Query Processing | ~3s | 150MB |
| Embedding Generation | ~0.5s | 100MB |
| Reranking (20 docs) | ~1s | 50MB |

*Benchmarks with gemma:2b on M1 MacBook*

## 🔍 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Memory Issues
- Switch to smaller model: `ollama pull tinyllama`
- Reduce chunk_size in config.yaml
- Disable reranking for lower memory usage

### Performance Issues
- Enable GPU acceleration if available
- Increase embedding cache size
- Use FAISS instead of ChromaDB for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📚 Documentation

- [Frontend Integration Guide](frontend_requirements.md)
- [API Reference](docs/api.md) (coming soon)
- [Architecture Overview](docs/architecture.md) (coming soon)

## 📄 License

This project is for educational and research purposes. See LICENSE file for details.

## 🙏 Acknowledgments

- Built following best practices from "A Survey on Retrieval-Augmented Generation"
- Ollama for local LLM inference
- ChromaDB for vector storage
- Sentence-Transformers for embeddings

---

**Ready to build intelligent applications with RAG?** Start with `python ollama_demo.py` for an interactive experience!