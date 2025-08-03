# RAG for LLMs: Implementation from Scratch

This project implements a complete Retrieval-Augmented Generation (RAG) system from scratch, following the comprehensive implementation guide. The system is built in phases, starting with basic components and progressively adding advanced features.

## Project Structure

```
rag-from-scratch/
├── src/                                  # Source code for the RAG system
│   ├── chunking/                         # Text splitting and chunking modules
│   ├── embedding/                        # Text embedding generation
│   ├── retrieval/                        # Vector store and retrieval
│   ├── generation/                       # LLM-based text generation
│   └── rag/                              # Core RAG pipeline implementations
├── data/                                 # Storage for documents
├── tests/                                # Unit and integration tests
├── notebooks/                            # Jupyter notebooks for evaluation
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Setup Instructions

1. **Create Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory for API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Implementation Phases

### Phase 1: Project Setup & Basic Structure ✅
- [x] Initialize project structure
- [x] Set up virtual environment
- [x] Install dependencies
- [x] Create basic directory structure

### Phase 2: Naive RAG Implementation ✅
- [x] Text chunking module
- [x] Embedding module
- [x] Vector store module
- [x] Generation module
- [x] Naive RAG pipeline

### Phase 3: Testing & Evaluation ✅
- [x] Unit tests for all components
- [x] Integration tests
- [x] Performance evaluation
- [x] Error handling tests

### Phase 4: Advanced RAG Features ✅
- [x] Query optimization (LLM rewriting, expansion, decomposition, HyDE)
- [x] Reranking (Cross-encoder and simple heuristic)
- [x] Advanced retrieval strategies
- [x] Hybrid search capabilities

### Phase 5: Modular RAG Architecture ✅
- [x] Factory pattern for component creation
- [x] Context compression strategies
- [x] Hybrid search (dense + sparse)
- [x] Conversation memory and multi-turn dialogue
- [x] Production-ready modular architecture

### Phase 6: Advanced Applications ✅
- [x] Domain-specific RAG (Code RAG)
- [x] Performance optimization (quantization, GPU acceleration)
- [x] Dynamic chunk size optimization
- [x] Comprehensive performance monitoring
- [x] Production deployment features

## Usage

### Basic Usage

```python
from src.rag.naive_rag import NaiveRAG

# Initialize RAG system
rag = NaiveRAG(config_path='config.yaml')

# Index documents
rag.index_documents(['path/to/document1.txt', 'path/to/document2.pdf'])

# Query the system
answer = rag.query("What is the capital of France?")
print(answer)

### Advanced Usage

```python
from src.rag.advanced_rag import AdvancedRAG

# Initialize Advanced RAG system
advanced_rag = AdvancedRAG(config_path='config.yaml')

# Index documents
advanced_rag.index_documents(['path/to/document1.txt', 'path/to/document2.pdf'])

# Query with different optimization strategies
answer1 = advanced_rag.query_optimized("What is the capital of France?", 
                                      query_optimization_strategy="expansion")
answer2 = advanced_rag.query_optimized("How does AI work?", 
                                      query_optimization_strategy="hyde")
answer3 = advanced_rag.query_optimized("What are the components of RAG?", 
                                      query_optimization_strategy="decomposition")
```
```

### Configuration

Create a `config.yaml` file to configure the RAG components:

```yaml
text_splitter:
  type: fixed_size
  chunk_size: 500
  chunk_overlap: 50

embedder:
  type: sentence_transformer
  model_name: all-MiniLM-L6-v2
  cache_dir: ./embedding_cache

vector_store:
  type: chromadb
  path: ./chroma_db
  collection_name: rag_collection

generator:
  type: huggingface
  model_name: distilgpt2
  device: cpu
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Demo

### Basic Demo
Run the basic demo to see the RAG system in action:

```bash
python demo.py
```

This will:
- Initialize the RAG system
- Index sample documents about Paris, AI, and RAG
- Demonstrate question-answering capabilities
- Show retrieved documents and generated answers

### Advanced Demo
Run the advanced demo to see advanced features:

```bash
python advanced_demo.py
```

This will showcase:
- Query optimization strategies (expansion, decomposition, HyDE)
- Reranking with cross-encoders
- Performance comparisons
- Advanced retrieval techniques

### Complete Demo
Run the complete demo to see all phases:

```bash
python complete_demo.py
```

This comprehensive demo showcases:
- All 6 phases of the RAG implementation
- Modular architecture with factory pattern
- Domain-specific applications (Code RAG)
- Performance optimization features
- Production-ready capabilities

## Contributing

This project follows the implementation guide structure. Each phase builds upon the previous one, ensuring a solid foundation for advanced features.

## License

This project is for educational purposes and follows the implementation guide provided. 