# RAG for LLMs: An In-Depth Technical Implementation Guide

## Introduction
This guide provides a comprehensive, in-depth technical implementation roadmap for building Retrieval-Augmented Generation (RAG) systems for Large Language Models (LLMs). Based on the concepts outlined in "RAG for LLMs: a survey," this document will walk you through the process of constructing a robust RAG system from scratch, starting with foundational components and progressively integrating advanced features, optimization techniques, and deployment considerations. Each section will delve into the technical specifics, offering practical insights and implementation patterns to facilitate a successful RAG integration.




## Guide Outline

This guide is structured into several phases, mirroring a typical development lifecycle for a RAG system:

### Phase 1: Project Setup & Basic Structure
- Initialize Project
- Create Project Structure
- Initial Dependencies

### Phase 2: Naive RAG Implementation
- Text Chunking Module
- Embedding Module
- Vector Store Module
- Generation Module
- Naive RAG Pipeline

### Phase 3: Testing & Evaluation
- Unit Tests
- Evaluation Notebook

### Phase 4: Advanced RAG Features
- Query Optimization
- Retrieval Enhancements
- Iterative Retrieval

### Phase 5: Modular RAG Architecture
- Modular Components
- Production Features

### Phase 6: Advanced Applications
- Domain-Specific RAG
- Performance Optimization

### Best Practices for Using Claude in Cursor
- Iterative Development
- Code Review Prompts
- Debugging Prompts
- Documentation Prompts

### Testing Checklist

### Resources & References

### Common Pitfalls to Avoid




## Phase 1: Project Setup & Basic Structure

This initial phase lays the groundwork for our RAG system. A well-organized project structure and proper dependency management are crucial for scalability and maintainability. We will establish the core directories, set up a virtual environment, and install the necessary initial libraries.

### 1.1 Initialize Project

To begin, create a dedicated directory for your RAG project and navigate into it. It is highly recommended to use a virtual environment to manage project dependencies, ensuring that your project's libraries do not conflict with other Python projects on your system.

```bash
mkdir rag-from-scratch
cd rag-from-scratch
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

This sequence of commands first creates a directory named `rag-from-scratch`, then changes the current working directory to this new location. Subsequently, a virtual environment named `venv` is created within this directory, and finally, the virtual environment is activated. Activation modifies your shell's `PATH` variable to prioritize executables within the virtual environment, such as `python` and `pip`.

### 1.2 Create Project Structure

A clear and logical project structure is vital for managing complexity, especially as the RAG system evolves with more advanced features. The proposed structure separates concerns into distinct modules for chunking, embedding, retrieval, and generation, alongside dedicated directories for data, tests, and notebooks.

```
rag-from-scratch/
├── src/                                  # Source code for the RAG system
│   ├── __init__.py                       # Makes `src` a Python package
│   ├── chunking/                         # Modules for text splitting and chunking
│   │   ├── __init__.py
│   │   └── text_splitter.py              # Handles various text splitting strategies
│   ├── embedding/                        # Modules for generating text embeddings
│   │   ├── __init__.py
│   │   └── embedder.py                   # Manages different embedding models and caching
│   ├── retrieval/                        # Modules for document retrieval and vector storage
│   │   ├── __init__.py
│   │   └── vector_store.py               # Implements vector database interactions
│   ├── generation/                       # Modules for LLM-based text generation
│   │   ├── __init__.py
│   │   └── generator.py                  # Manages LLM backends and prompt templating
│   └── rag/                              # Core RAG pipeline implementations
│       ├── __init__.py
│       └── naive_rag.py                  # Basic retrieve-then-generate pipeline
├── data/                                 # Storage for raw and processed data
│   └── documents/                        # Directory for source documents (e.g., PDFs, text files)
├── tests/                                # Unit and integration tests
├── notebooks/                            # Jupyter notebooks for experimentation and evaluation
├── requirements.txt                      # List of Python dependencies
├── .env                                  # Environment variables (e.g., API keys)
└── README.md                             # Project overview and setup instructions
```

This structure promotes modularity, making it easier to develop, test, and debug individual components. For instance, all text processing logic resides within `src/chunking`, while `src/embedding` is solely responsible for embedding generation.

### 1.3 Initial Dependencies

Before implementing any code, we need to define and install the core Python libraries that will be used across the project. These dependencies cover fundamental data manipulation, environment variable management, and initial components for embeddings, vector stores, and LLM interaction.

Create a `requirements.txt` file at the root of your `rag-from-scratch` directory with the following content:

```txt
# Core dependencies
numpy==1.24.3
pandas==2.0.3
python-dotenv==1.0.0

# Embeddings
sentence-transformers==2.2.2
openai==1.3.0  # If using OpenAI embeddings

# Vector Store
faiss-cpu==1.7.4
chromadb==0.4.15

# LLM
transformers==4.35.0
torch==2.1.0
langchain==0.0.340  # Optional, for LLM abstraction

# Utils
tiktoken==0.5.1
tqdm==4.66.1
pytest==7.4.3
```

After creating `requirements.txt`, install these dependencies using pip within your activated virtual environment:

```bash
pip install -r requirements.txt
```

This command reads the `requirements.txt` file and installs each listed package with its specified version. It's crucial to pin exact versions (`==`) for reproducibility across different environments. `numpy` and `pandas` are essential for data handling. `python-dotenv` allows for loading environment variables from a `.env` file, which is critical for securely managing API keys. `sentence-transformers` and `openai` provide options for generating embeddings. `faiss-cpu` and `chromadb` are selected as initial vector store backends. `transformers` and `torch` are fundamental for working with large language models, while `langchain` offers a higher-level abstraction for LLM applications. Finally, `tiktoken` is useful for token counting, `tqdm` for progress bars, and `pytest` for unit testing.




## Phase 2: Naive RAG Implementation

This phase focuses on building the foundational components of a Retrieval-Augmented Generation (RAG) system. We will implement modules for text chunking, embedding generation, vector storage and retrieval, and LLM-based text generation. Finally, these modules will be integrated into a cohesive Naive RAG pipeline.

### 2.1 Text Chunking Module

Text chunking is a critical preprocessing step in RAG. Large documents need to be broken down into smaller, manageable pieces (chunks) that can be effectively embedded and retrieved. The chunking strategy significantly impacts retrieval quality. We will implement both fixed-size and sentence-based chunking, ensuring metadata preservation and handling of various file formats.

**File:** `src/chunking/text_splitter.py`

```python
import abc
from typing import List, Dict, Any

class TextSplitter(abc.ABC):
    """
    Abstract base class for text splitting strategies.
    """
    @abc.abstractmethod
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Splits the input text into chunks.

        Args:
            text: The input text to split.
            metadata: Optional metadata associated with the text (e.g., source, page_number).

        Returns:
            A list of dictionaries, where each dictionary represents a chunk
            and contains 'content' and 'metadata' keys.
        """
        pass

class FixedSizeTextSplitter(TextSplitter):
    """
    Implements fixed-size chunking with overlap.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be non-negative and less than chunk size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_content = text[start:end]
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['start_index'] = start
            chunk_metadata['end_index'] = end
            chunks.append({"content": chunk_content, "metadata": chunk_metadata})
            start += self.chunk_size - self.chunk_overlap
        return chunks

class SentenceTextSplitter(TextSplitter):
    """
    Implements sentence-based chunking.
    """
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # A more robust sentence splitter would use a library like NLTK or spaCy.
        # For simplicity, we'll use a basic regex split.
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i, sentence in enumerate(sentences):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['sentence_index'] = i
            chunks.append({"content": sentence.strip(), "metadata": chunk_metadata})
        return chunks

# Example of how to handle different file formats (simplified)
# In a real application, you'd use libraries like PyPDF2, python-docx, etc.
class DocumentLoader:
    def load_text_from_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_pdf_text(self, file_path: str) -> str:
        # Placeholder for PDF text extraction logic
        # Requires libraries like PyPDF2 or pdfminer.six
        return f"Content from PDF: {file_path}"

    def load_markdown_text(self, file_path: str) -> str:
        # Placeholder for Markdown text extraction logic
        return f"Content from Markdown: {file_path}"

```

**Explanation:**

*   **`TextSplitter` (Abstract Base Class):** Defines a common interface (`split_text` method) for all text splitting strategies. This allows for easy extensibility and switching between different chunking methods without altering the core RAG pipeline.
*   **`FixedSizeTextSplitter`:** Splits text into chunks of a predefined `chunk_size` with an optional `chunk_overlap`. Overlap is crucial for maintaining context across chunks, as relevant information might span across chunk boundaries. Each chunk includes its original content and metadata, such as `start_index` and `end_index` to trace back to the original document.
*   **`SentenceTextSplitter`:** Splits text based on sentence boundaries. This method often produces more semantically coherent chunks than fixed-size chunking, as it avoids breaking sentences mid-way. For a production-ready system, consider using advanced NLP libraries like NLTK or spaCy for more accurate sentence tokenization.
*   **`DocumentLoader` (Conceptual):** While not fully implemented here, this class illustrates how different file formats (`.txt`, `.pdf`, `.md`) would be handled. In a real-world scenario, you would integrate libraries like `PyPDF2` or `pdfminer.six` for PDF parsing, `python-docx` for Word documents, and `markdown` for Markdown files. The extracted text would then be passed to the chosen `TextSplitter`.

Metadata preservation is vital. Each chunk is returned as a dictionary containing the `content` and a `metadata` dictionary. This metadata can include the original source file, page numbers, section titles, or any other relevant information that aids in retrieval and attribution.

### 2.2 Embedding Module

Embeddings are numerical representations of text that capture semantic meaning. They are fundamental to RAG, enabling the system to find semantically similar documents to a given query. This module will support various embedding models, implement batch processing for efficiency, and include caching to avoid redundant computations.

**File:** `src/embedding/embedder.py`

```python
import abc
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
# from openai import OpenAI # Uncomment if using OpenAI

class Embedder(abc.ABC):
    """
    Abstract base class for embedding models.
    """
    @abc.abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of lists of floats, where each inner list is an embedding vector.
        """
        pass

class SentenceTransformerEmbedder(Embedder):
    """
    Embedder using Sentence Transformers models.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = './embedding_cache'):
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.cache = {}

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        texts_to_embed = []
        for text in texts:
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                texts_to_embed.append(text)

        if texts_to_embed:
            # Batch processing for efficiency
            new_embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True).tolist()
            for i, text in enumerate(texts_to_embed):
                self.cache[text] = new_embeddings[i]
                embeddings.append(new_embeddings[i])
        return embeddings

# class OpenAIEmbedder(Embedder):
#     """
#     Embedder using OpenAI API.
#     """
#     def __init__(self, model_name: str = 'text-embedding-ada-002'):
#         self.client = OpenAI()
#         self.model_name = model_name
#         self.cache = {}

#     def embed(self, texts: List[str]) -> List[List[float]]:
#         embeddings = []
#         texts_to_embed = []
#         for text in texts:
#             if text in self.cache:
#                 embeddings.append(self.cache[text])
#             else:
#                 texts_to_embed.append(text)

#         if texts_to_embed:
#             # OpenAI API handles batching internally up to a limit
#             response = self.client.embeddings.create(
#                 input=texts_to_embed,
#                 model=self.model_name
#             )
#             new_embeddings = [d.embedding for d in response.data]
#             for i, text in enumerate(texts_to_embed):
#                 self.cache[text] = new_embeddings[i]
#                 embeddings.append(new_embeddings[i])
#         return embeddings

# Note: For sparse embeddings (e.g., BM25, TF-IDF), a different approach would be needed,
# typically involving a sparse matrix representation and a different retrieval mechanism.
# This example focuses on dense embeddings.
```

**Explanation:**

*   **`Embedder` (Abstract Base Class):** Defines the `embed` method, which takes a list of texts and returns their corresponding embedding vectors. This abstract interface allows for seamless integration of different embedding models.
*   **`SentenceTransformerEmbedder`:** Implements embedding generation using models from the `sentence-transformers` library. These models are highly efficient for generating dense embeddings and can be run locally. The `cache_dir` parameter allows for local caching of model weights. Crucially, it includes a simple in-memory cache (`self.cache`) to store previously computed embeddings, preventing redundant API calls or computations for identical texts. This is particularly useful when dealing with large datasets or repeated queries.
*   **`OpenAIEmbedder` (Commented Out):** This class demonstrates how to integrate with OpenAI's embedding API. It would require an OpenAI API key and the `openai` Python package. Similar to the Sentence Transformer embedder, it includes caching to optimize API usage and reduce costs. The OpenAI API handles batching of inputs internally.

**Dense vs. Sparse Embeddings:** This implementation focuses on dense embeddings, which are typically generated by neural networks and capture semantic similarity effectively. Sparse embeddings (e.g., TF-IDF, BM25) are based on term frequency and inverse document frequency and are excellent for keyword matching. A comprehensive RAG system often benefits from a hybrid approach, combining both dense and sparse retrieval, which will be explored in later phases.

### 2.3 Vector Store Module

The vector store is the heart of the retrieval component. It stores the numerical embeddings of our document chunks and allows for efficient similarity search. This module will support multiple backends (FAISS and ChromaDB), provide CRUD operations, and implement similarity search with metadata filtering.

**File:** `src/retrieval/vector_store.py`

```python
import abc
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

class VectorStore(abc.ABC):
    """
    Abstract base class for vector store implementations.
    """
    @abc.abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Adds documents and their embeddings to the vector store.
        """
        pass

    @abc.abstractmethod
    def search(self, query_embedding: List[float], k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search.

        Args:
            query_embedding: The embedding of the query.
            k: The number of top similar documents to retrieve.
            filters: Optional metadata filters.

        Returns:
            A list of dictionaries, each representing a retrieved document with content and metadata.
        """
        pass

    @abc.abstractmethod
    def update_document(self, doc_id: str, new_content: str = None, new_metadata: Dict[str, Any] = None, new_embedding: List[float] = None):
        """
        Updates an existing document in the vector store.
        """
        pass

    @abc.abstractmethod
    def delete_document(self, doc_id: str):
        """
        Deletes a document from the vector store.
        """
        pass

class FAISSVectorStore(VectorStore):
    """
    Vector store implementation using FAISS.
    FAISS is an open-source library for efficient similarity search and clustering of dense vectors.
    """
    def __init__(self, embedding_dimension: int):
        self.index = faiss.IndexFlatL2(embedding_dimension) # L2 distance for similarity
        self.documents = [] # Stores {'id': ..., 'content': ..., 'metadata': ..., 'embedding': ...}
        self.doc_id_map = {}
        self.next_id = 0

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        if not documents or not embeddings:
            return
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")

        new_embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(new_embeddings_np)

        for i, doc in enumerate(documents):
            doc_id = str(self.next_id)
            self.documents.append({
                'id': doc_id,
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'embedding': embeddings[i]
            })
            self.doc_id_map[doc_id] = self.next_id # Map doc_id to internal FAISS index
            self.next_id += 1

    def search(self, query_embedding: List[float], k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        query_embedding_np = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding_np, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: # FAISS returns -1 for empty slots if k > num_vectors
                continue
            doc = self.documents[idx]
            # Apply filters if provided (FAISS itself doesn't support direct metadata filtering)
            if filters:
                match = True
                for key, value in filters.items():
                    if doc['metadata'].get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            results.append({"content": doc['content'], "metadata": doc['metadata'], "distance": distances[0][i]})
        return results

    def update_document(self, doc_id: str, new_content: str = None, new_metadata: Dict[str, Any] = None, new_embedding: List[float] = None):
        # FAISS does not support direct update/delete of individual vectors easily.
        # A common strategy is to rebuild the index or mark documents as 'deleted'.
        # For simplicity, this implementation will not fully support in-place updates/deletes
        # without rebuilding the index, which is inefficient for single updates.
        # For a production system, consider a vector database that supports these operations natively.
        print(f"Warning: FAISSVectorStore does not efficiently support updating/deleting individual documents. "
              f"Consider rebuilding the index or using a different vector store for frequent updates.")

    def delete_document(self, doc_id: str):
        print(f"Warning: FAISSVectorStore does not efficiently support updating/deleting individual documents. "
              f"Consider rebuilding the index or using a different vector store for frequent deletions.")


class ChromaDBVectorStore(VectorStore):
    """
    Vector store implementation using ChromaDB.
    ChromaDB is an open-source embedding database that makes it easy to build LLM apps.
    """
    def __init__(self, path: str = "./chroma_db", collection_name: str = "rag_collection"):
        self.client = chromadb.PersistentClient(path=path)
        # ChromaDB requires an embedding function, which it uses internally.
        # We'll use a dummy one here since embeddings are provided externally.
        # In a real scenario, you might let ChromaDB handle embeddings or use a consistent one.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        )

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        ids = [str(i) for i in range(self.collection.count(), self.collection.count() + len(documents))]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )

    def search(self, query_embedding: List[float], k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filters if filters else {},
            include=['documents', 'metadatas', 'distances']
        )
        retrieved_docs = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
        return retrieved_docs

    def update_document(self, doc_id: str, new_content: str = None, new_metadata: Dict[str, Any] = None, new_embedding: List[float] = None):
        update_data = {}
        if new_content is not None: update_data['documents'] = [new_content]
        if new_metadata is not None: update_data['metadatas'] = [new_metadata]
        if new_embedding is not None: update_data['embeddings'] = [new_embedding]
        self.collection.update(ids=[doc_id], **update_data)

    def delete_document(self, doc_id: str):
        self.collection.delete(ids=[doc_id])

```

**Explanation:**

*   **`VectorStore` (Abstract Base Class):** Defines the standard interface for vector store operations: `add_documents`, `search`, `update_document`, and `delete_document`. This abstraction allows for easy swapping of underlying vector database technologies.
*   **`FAISSVectorStore`:** Implements a vector store using FAISS (Facebook AI Similarity Search). FAISS is highly optimized for similarity search on large datasets of dense vectors. It uses an `IndexFlatL2` for L2 (Euclidean) distance similarity. A key limitation of `IndexFlatL2` is its lack of direct support for metadata filtering and efficient individual document updates/deletions. For these operations, a full index rebuild or a more complex management layer is often required. The `documents` list and `doc_id_map` are maintained to simulate document storage and retrieval, as FAISS itself only stores vectors.
*   **`ChromaDBVectorStore`:** Implements a vector store using ChromaDB. ChromaDB is designed to be a more user-friendly embedding database that integrates well with LLM applications. It supports persistent storage, metadata filtering (`where` clause in `query`), and native CRUD operations. ChromaDB can also handle embedding generation internally, but for consistency with our `Embedder` module, we provide embeddings externally.

**Key Considerations:**

*   **Similarity Metric:** The choice of similarity metric (e.g., L2 distance, cosine similarity) depends on the embedding model used. For many modern embedding models, cosine similarity is preferred, but L2 distance can also work effectively, especially when embeddings are normalized.
*   **Metadata Filtering:** Crucial for retrieving contextually relevant documents. For example, you might want to retrieve documents only from a specific source or published within a certain date range. ChromaDB supports this natively, while FAISS requires post-filtering.
*   **Scalability:** For very large datasets, more advanced FAISS indices (e.g., `IndexIVFFlat`, `IndexHNSW`) or distributed vector databases (e.g., Milvus, Pinecone, Weaviate) would be necessary.

### 2.4 Generation Module

The generation module is responsible for taking the retrieved context and the user's query, constructing a suitable prompt, and interacting with a Large Language Model (LLM) to generate a coherent and informative answer. It will support multiple LLM backends and manage prompt templates.

**File:** `src/generation/generator.py`

```python
import abc
from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# from openai import OpenAI # Uncomment if using OpenAI

class Generator(abc.ABC):
    """
    Abstract base class for LLM generation.
    """
    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates text based on the given prompt.
        """
        pass

    @abc.abstractmethod
    def stream_generate(self, prompt: str, **kwargs):
        """
        Generates text in a streaming fashion.
        """
        pass

class HuggingFaceGenerator(Generator):
    """
    Generator using HuggingFace models (e.g., local models).
    """
    def __init__(self, model_name: str = 'distilgpt2', device: str = 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, **kwargs) -> str:
        # Ensure prompt is a string
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        # The pipeline returns a list of dictionaries, take the generated_text from the first one
        result = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True, # Enable sampling for temperature to have effect
            pad_token_id=self.tokenizer.eos_token_id, # Handle padding for batching
            **kwargs
        )[0]['generated_text']

        # The pipeline returns the prompt concatenated with the generated text.
        # We need to extract only the new generated part.
        return result[len(prompt):].strip()

    def stream_generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, **kwargs):
        # HuggingFace pipeline does not directly support streaming in a simple way for all models.
        # For true streaming, you would typically use the model's generate method with `stream=True`
        # or integrate with a framework like FastAPI for SSE.
        # This is a simplified placeholder.
        full_response = self.generate(prompt, max_new_tokens, temperature, **kwargs)
        yield full_response # Yield the full response as a single chunk for simplicity

# class OpenAIGenerator(Generator):
#     """
#     Generator using OpenAI API.
#     """
#     def __init__(self, model_name: str = 'gpt-3.5-turbo'):
#         self.client = OpenAI()
#         self.model_name = model_name

#     def generate(self, prompt: str, max_new_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=max_new_tokens,
#             temperature=temperature,
#             **kwargs
#         )
#         return response.choices[0].message.content.strip()

#     def stream_generate(self, prompt: str, max_new_tokens: int = 500, temperature: float = 0.7, **kwargs):
#         stream = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=max_new_tokens,
#             temperature=temperature,
#             stream=True,
#             **kwargs
#         )
#         for chunk in stream:
#             if chunk.choices[0].delta.content is not None:
#                 yield chunk.choices[0].delta.content

class PromptTemplate:
    """
    Manages prompt templates for RAG.
    """
    def __init__(self, template: str = """
    Use the following context to answer the question at the end. If you don't know the answer,
    just say that you don't know, don't try to make up an answer.

    Context:
    {context}

    Question: {question}

    Answer:
    """):
        self.template = template

    def format_prompt(self, question: str, context: List[str]) -> str:
        context_str = "\n\n".join(context)
        return self.template.format(context=context_str, question=question)

```

**Explanation:**

*   **`Generator` (Abstract Base Class):** Defines the `generate` and `stream_generate` methods, providing a consistent interface for interacting with different LLMs.
*   **`HuggingFaceGenerator`:** Utilizes the `transformers` library to load and run local or Hugging Face Hub models. It uses the `pipeline` abstraction for ease of use. Parameters like `max_new_tokens` and `temperature` control the generation process. `temperature` influences the randomness of the output: higher values make the output more creative, lower values make it more deterministic. Note that true streaming for all Hugging Face models can be more complex and might require direct interaction with the model's `generate` method or specific server implementations.
*   **`OpenAIGenerator` (Commented Out):** Demonstrates integration with OpenAI's Chat Completions API. It supports both standard generation and streaming, which is beneficial for user experience as it provides real-time output. This would require an OpenAI API key.
*   **`PromptTemplate`:** A crucial component for RAG. It defines how the retrieved context and the user's question are combined into a single prompt that is fed to the LLM. The example template clearly instructs the LLM to use the provided context and to avoid hallucinating answers. Effective prompt engineering is key to getting good results from RAG systems.

**Context Window Management:** LLMs have a limited context window (the maximum number of tokens they can process at once). The `PromptTemplate` implicitly handles this by concatenating the retrieved context. In more advanced scenarios, if the combined context and question exceed the LLM's context window, strategies like summarization, re-ranking, or dynamic truncation would be needed. This will be touched upon in later phases.

### 2.5 Naive RAG Pipeline

The Naive RAG pipeline integrates all the previously built modules into a single, cohesive system. It follows the basic retrieve-then-generate flow: given a query, it first retrieves relevant documents from the vector store and then uses an LLM to generate an answer based on these retrieved documents.

**File:** `src/rag/naive_rag.py`

```python
import os
import yaml
import logging
from typing import List, Dict, Any

from src.chunking.text_splitter import FixedSizeTextSplitter, DocumentLoader
from src.embedding.embedder import SentenceTransformerEmbedder
from src.retrieval.vector_store import ChromaDBVectorStore, FAISSVectorStore
from src.generation.generator import HuggingFaceGenerator, PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NaiveRAG:
    """
    Implements a basic Retrieve-Augmented Generation (RAG) pipeline.
    """
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)

        # Initialize components based on configuration
        self.text_splitter = self._initialize_text_splitter()
        self.embedder = self._initialize_embedder()
        self.vector_store = self._initialize_vector_store()
        self.generator = self._initialize_generator()
        self.prompt_template = PromptTemplate()
        self.document_loader = DocumentLoader() # For loading documents

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_text_splitter(self):
        splitter_config = self.config['text_splitter']
        splitter_type = splitter_config.get('type', 'fixed_size')
        if splitter_type == 'fixed_size':
            return FixedSizeTextSplitter(
                chunk_size=splitter_config.get('chunk_size', 500),
                chunk_overlap=splitter_config.get('chunk_overlap', 50)
            )
        elif splitter_type == 'sentence':
            # For simplicity, SentenceTextSplitter doesn't take params here
            return FixedSizeTextSplitter() # Using fixed size as a placeholder for now
        else:
            raise ValueError(f"Unknown text splitter type: {splitter_type}")

    def _initialize_embedder(self):
        embedder_config = self.config['embedder']
        embedder_type = embedder_config.get('type', 'sentence_transformer')
        if embedder_type == 'sentence_transformer':
            return SentenceTransformerEmbedder(
                model_name=embedder_config.get('model_name', 'all-MiniLM-L6-v2'),
                cache_dir=embedder_config.get('cache_dir', './embedding_cache')
            )
        # elif embedder_type == 'openai':
        #     return OpenAIEmbedder(model_name=embedder_config.get('model_name', 'text-embedding-ada-002'))
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")

    def _initialize_vector_store(self):
        vector_store_config = self.config['vector_store']
        store_type = vector_store_config.get('type', 'chromadb')
        if store_type == 'chromadb':
            return ChromaDBVectorStore(
                path=vector_store_config.get('path', './chroma_db'),
                collection_name=vector_store_config.get('collection_name', 'rag_collection')
            )
        elif store_type == 'faiss':
            # For FAISS, we need the embedding dimension. This should ideally come from the embedder.
            # For now, we'll hardcode or get from config if known.
            embedding_dimension = vector_store_config.get('embedding_dimension', 384) # all-MiniLM-L6-v2 dimension
            return FAISSVectorStore(embedding_dimension=embedding_dimension)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")

    def _initialize_generator(self):
        generator_config = self.config['generator']
        generator_type = generator_config.get('type', 'huggingface')
        if generator_type == 'huggingface':
            return HuggingFaceGenerator(
                model_name=generator_config.get('model_name', 'distilgpt2'),
                device=generator_config.get('device', 'cpu')
            )
        # elif generator_type == 'openai':
        #     return OpenAIGenerator(model_name=generator_config.get('model_name', 'gpt-3.5-turbo'))
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

    def index_documents(self, document_paths: List[str]):
        """
        Loads, chunks, embeds, and adds documents to the vector store.
        """
        logging.info(f"Indexing {len(document_paths)} documents...")
        all_chunks = []
        for doc_path in document_paths:
            try:
                # Determine file type and load content
                file_extension = os.path.splitext(doc_path)[1].lower()
                if file_extension == '.txt':
                    text = self.document_loader.load_text_from_file(doc_path)
                elif file_extension == '.pdf':
                    text = self.document_loader.load_pdf_text(doc_path)
                elif file_extension == '.md':
                    text = self.document_loader.load_markdown_text(doc_path)
                else:
                    logging.warning(f"Unsupported file type for {doc_path}. Skipping.")
                    continue

                metadata = {"source": doc_path, "file_type": file_extension}
                chunks = self.text_splitter.split_text(text, metadata=metadata)
                all_chunks.extend(chunks)
                logging.info(f"Processed {len(chunks)} chunks from {doc_path}")
            except Exception as e:
                logging.error(f"Error processing document {doc_path}: {e}")

        if not all_chunks:
            logging.warning("No chunks generated for indexing.")
            return

        logging.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        chunk_contents = [chunk['content'] for chunk in all_chunks]
        embeddings = self.embedder.embed(chunk_contents)

        logging.info(f"Adding {len(all_chunks)} chunks to vector store...")
        self.vector_store.add_documents(all_chunks, embeddings)
        logging.info("Document indexing complete.")

    def retrieve(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieves relevant documents for a given query.
        """
        logging.info(f"Retrieving top {k} documents for query: '{query}'")
        query_embedding = self.embedder.embed([query])[0]
        retrieved_docs = self.vector_store.search(query_embedding, k=k, filters=filters)
        logging.info(f"Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs

    def generate_answer(self, question: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using the LLM based on the question and retrieved contexts.
        """
        logging.info("Generating answer...")
        context_contents = [doc['content'] for doc in retrieved_contexts]
        formatted_prompt = self.prompt_template.format_prompt(question, context_contents)
        answer = self.generator.generate(formatted_prompt)
        logging.info("Answer generated.")
        return answer

    def query(self, question: str, k: int = 5, filters: Dict[str, Any] = None) -> str:
        """
        End-to-end query method for the Naive RAG pipeline.
        """
        retrieved_contexts = self.retrieve(question, k=k, filters=filters)
        if not retrieved_contexts:
            logging.warning("No relevant contexts found. Cannot generate an answer.")
            return "I am sorry, but I could not find enough relevant information to answer your question."
        answer = self.generate_answer(question, retrieved_contexts)
        return answer

# Example Configuration (config.yaml)
# Create this file in the root of your project: rag-from-scratch/config.yaml
"""
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
  # For FAISS, you might need:
  # type: faiss
  # embedding_dimension: 384 # For all-MiniLM-L6-v2

generator:
  type: huggingface
  model_name: distilgpt2
  device: cpu # or 'cuda' if GPU is available
  # For OpenAI, you might need:
  # type: openai
  # model_name: gpt-3.5-turbo
"""

```

**Explanation:**

*   **`NaiveRAG` Class:** The central orchestrator of the RAG pipeline. Its constructor loads configuration from a `config.yaml` file and initializes instances of the `TextSplitter`, `Embedder`, `VectorStore`, and `Generator` based on the specified types and parameters.
*   **Configuration Management:** The `_load_config` method reads a YAML file, allowing for flexible and external configuration of the RAG components. This is crucial for easy experimentation with different models, chunking strategies, and vector store backends without modifying the code.
*   **`index_documents` Method:** This is the ingestion pipeline. It takes a list of document paths, loads their content (handling different file types conceptually), splits them into chunks using the configured `text_splitter`, generates embeddings for these chunks using the `embedder`, and finally adds the chunks and their embeddings to the `vector_store`. Robust error handling and logging are included.
*   **`retrieve` Method:** Given a user `query`, it first generates an embedding for the query using the `embedder`. Then, it performs a similarity search in the `vector_store` to find the `k` most relevant document chunks. Optional `filters` can be applied to narrow down the search space based on metadata.
*   **`generate_answer` Method:** Takes the original `question` and the `retrieved_contexts` (the content of the relevant chunks). It uses the `prompt_template` to construct a single, coherent prompt that includes both the question and the context. This formatted prompt is then sent to the `generator` (LLM) to produce the final answer.
*   **`query` Method:** Provides an end-to-end interface for the RAG system. It orchestrates the retrieval and generation steps, returning the LLM's answer. It also includes a basic fallback message if no relevant contexts are found.
*   **Logging:** Integrated throughout the pipeline to provide visibility into the process, aiding in debugging and monitoring.

**Example `config.yaml`:** A sample `config.yaml` is provided to illustrate how to configure the different components. This file should be placed in the root of your `rag-from-scratch` project directory. You can easily switch between `chromadb` and `faiss` for the vector store, or `huggingface` and `openai` for the generator, by simply modifying this configuration file.

This Naive RAG implementation forms the bedrock upon which more advanced features and optimizations will be built in subsequent phases. It demonstrates the fundamental retrieve-then-generate pattern, showcasing the interaction between the core RAG components.




## Phase 3: Testing & Evaluation

Once the Naive RAG system is built, it is crucial to thoroughly test its components and evaluate its end-to-end performance. This phase ensures the reliability of the system and provides a baseline for future improvements. We will cover both unit testing for individual modules and a comprehensive evaluation of the entire RAG pipeline using a Jupyter notebook.

### 3.1 Unit Tests

Unit tests are essential for verifying the correctness of each component in isolation. They help catch bugs early, facilitate refactoring, and ensure that individual modules behave as expected under various conditions, including edge cases. We will use the `pytest` framework for writing and running our tests.

**Directory:** `tests/`

Create separate test files for each module (e.g., `tests/test_text_splitter.py`, `tests/test_embedder.py`, etc.).

**Example: `tests/test_text_splitter.py`**

```python
import pytest
from src.chunking.text_splitter import FixedSizeTextSplitter, SentenceTextSplitter

class TestFixedSizeTextSplitter:
    def test_simple_split(self):
        splitter = FixedSizeTextSplitter(chunk_size=10, chunk_overlap=2)
        text = "This is a test sentence for splitting."
        chunks = splitter.split_text(text)
        assert len(chunks) == 4
        assert chunks[0]["content"] == "This is a "
        assert chunks[1]["content"] == "a test sen"
        assert chunks[2]["content"] == "sentence f"
        assert chunks[3]["content"] == "or splitting."

    def test_no_overlap(self):
        splitter = FixedSizeTextSplitter(chunk_size=10, chunk_overlap=0)
        text = "This is a test sentence."
        chunks = splitter.split_text(text)
        assert len(chunks) == 3
        assert chunks[0]["content"] == "This is a "
        assert chunks[1]["content"] == "test sente"
        assert chunks[2]["content"] == "nce."

    def test_edge_case_empty_text(self):
        splitter = FixedSizeTextSplitter()
        chunks = splitter.split_text("")
        assert len(chunks) == 0

    def test_metadata_preservation(self):
        splitter = FixedSizeTextSplitter(chunk_size=10, chunk_overlap=2)
        text = "This is a test."
        metadata = {"source": "test.txt"}
        chunks = splitter.split_text(text, metadata=metadata)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test.txt"

class TestSentenceTextSplitter:
    def test_sentence_splitting(self):
        splitter = SentenceTextSplitter()
        text = "This is the first sentence. This is the second one! Is this the third?"
        chunks = splitter.split_text(text)
        assert len(chunks) == 3
        assert chunks[0]["content"] == "This is the first sentence."
        assert chunks[1]["content"] == "This is the second one!"
        assert chunks[2]["content"] == "Is this the third?"

    def test_metadata_with_sentences(self):
        splitter = SentenceTextSplitter()
        text = "Sentence one. Sentence two."
        metadata = {"author": "Manus"}
        chunks = splitter.split_text(text, metadata=metadata)
        assert len(chunks) == 2
        assert chunks[0]["metadata"]["author"] == "Manus"
        assert chunks[1]["metadata"]["sentence_index"] == 1

```

**Explanation:**

*   **Test Classes:** We use separate classes (`TestFixedSizeTextSplitter`, `TestSentenceTextSplitter`) to organize tests for different components.
*   **Test Cases:** Each method starting with `test_` is a test case. We cover:
    *   **Simple scenarios:** Basic functionality checks.
    *   **Edge cases:** Empty input, no overlap, etc.
    *   **Metadata:** Ensuring that metadata is correctly passed to the chunks.
*   **Assertions:** `assert` statements are used to verify that the output of a function matches the expected result. If an assertion fails, `pytest` will report a test failure.

Similar unit tests should be created for the `embedder` (checking embedding dimensions, cache functionality), `vector_store` (verifying CRUD operations, search accuracy), and `generator` (testing prompt formatting, context handling).

To run the tests, navigate to the root of your project (`rag-from-scratch/`) and simply run the `pytest` command in your terminal. `pytest` will automatically discover and run all test files.

### 3.2 Evaluation Notebook

While unit tests verify individual components, an end-to-end evaluation is necessary to assess the overall quality of the RAG system. A Jupyter notebook is an excellent tool for this purpose, allowing for interactive experimentation, visualization, and analysis.

**File:** `notebooks/evaluate_naive_rag.ipynb`

This notebook will guide you through the process of loading a test dataset, running the RAG pipeline, and evaluating its performance using standard RAG metrics.

**Key Steps in the Evaluation Notebook:**

1.  **Setup and Initialization:**
    *   Import necessary libraries.
    *   Load the RAG system by creating an instance of the `NaiveRAG` class.
    *   Load a test dataset. For RAG evaluation, a dataset with question-answer pairs and corresponding ground-truth documents is ideal. Examples include subsets of Natural Questions or MS MARCO. If such a dataset is not available, you can create a small, custom one.

2.  **Data Indexing:**
    *   Index the documents from your test dataset into the RAG system's vector store.

3.  **Run RAG Pipeline:**
    *   Iterate through the question-answer pairs in your test set.
    *   For each question, use the `rag.query()` method to get the generated answer.
    *   Store the retrieved documents and the generated answer for each question.

4.  **Evaluation Metrics:**
    *   **Retrieval Evaluation:** Assess the quality of the retrieval component.
        *   **Hit Rate:** The proportion of questions for which at least one of the ground-truth documents was retrieved.
        *   **Mean Reciprocal Rank (MRR):** The average of the reciprocal ranks of the first correct retrieved document. A higher MRR indicates that the correct document is ranked higher.
        *   **Recall@K:** The proportion of questions for which a ground-truth document was present in the top-K retrieved documents.
    *   **Generation Evaluation:** Assess the quality of the generated answers.
        *   **BLEU (Bilingual Evaluation Understudy):** Measures the n-gram overlap between the generated answer and a reference answer. Good for measuring fluency.
        *   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Measures the overlap of n-grams, word sequences, and word pairs between the generated and reference answers. Good for measuring informativeness.
        *   **BERTScore:** Uses contextual embeddings from BERT to compute a similarity score between the generated and reference answers, capturing semantic similarity better than n-gram-based metrics.
    *   **End-to-End Evaluation:**
        *   **Answer Correctness:** A binary metric (correct/incorrect) often assessed through human evaluation.
        *   **Answer Faithfulness:** Measures whether the generated answer is factually consistent with the retrieved context. This can be evaluated by a human or by using another LLM as a judge.
        *   **Answer Relevance:** Measures how well the generated answer addresses the user's question.

5.  **Visualization and Analysis:**
    *   Create plots and tables to visualize the performance metrics.
    *   Analyze cases where the system failed (e.g., incorrect retrieval, unfaithful generation) to identify areas for improvement.
    *   Compare different RAG configurations (e.g., different chunk sizes, embedding models, or LLMs) to understand their impact on performance.

**Example Code Snippet for Evaluation Notebook:**

```python
# In notebooks/evaluate_naive_rag.ipynb

import pandas as pd
from src.rag.naive_rag import NaiveRAG
from tqdm import tqdm

# 1. Initialize RAG system
rag = NaiveRAG(config_path=\'../config.yaml\')

# 2. Load a sample dataset (e.g., a CSV with 'question', 'answer', 'ground_truth_doc_id')
# For this example, we'll create a dummy one.
data = {
    'question': ["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?"],
    'answer': ["The capital of France is Paris.", "Harper Lee wrote 'To Kill a Mockingbird'."],
    'ground_truth_doc_id': ["doc1", "doc2"]
}
df_eval = pd.DataFrame(data)

# Assume documents 'doc1.txt' and 'doc2.txt' exist in your data/documents/ directory
# and have been indexed.

# 3. Run RAG pipeline and collect results
results = []
for index, row in tqdm(df_eval.iterrows(), total=df_eval.shape[0]):
    question = row['question']
    retrieved_docs = rag.retrieve(question, k=5)
    generated_answer = rag.generate_answer(question, retrieved_docs)
    results.append({
        'question': question,
        'retrieved_docs': retrieved_docs,
        'generated_answer': generated_answer,
        'ground_truth_answer': row['answer']
    })

# 4. Evaluate (simplified example)
def calculate_retrieval_hit_rate(results, ground_truth_doc_ids):
    hits = 0
    for i, res in enumerate(results):
        retrieved_ids = [doc['metadata'].get('source') for doc in res['retrieved_docs']]
        if ground_truth_doc_ids[i] in retrieved_ids:
            hits += 1
    return hits / len(results)

# In a real scenario, you would use libraries like 'rouge_score', 'bleu', 'bert_score'
# to calculate generation metrics.

# 5. Print results
retrieval_hit_rate = calculate_retrieval_hit_rate(results, df_eval['ground_truth_doc_id'].tolist())
print(f"Retrieval Hit Rate: {retrieval_hit_rate:.2f}")

for res in results:
    print(f"\nQuestion: {res['question']}")
    print(f"Generated Answer: {res['generated_answer']}")
    print(f"Ground Truth Answer: {res['ground_truth_answer']}")

```

This evaluation phase is not a one-time step. As you add more advanced features to your RAG system in the subsequent phases, you should continuously re-evaluate its performance to ensure that the changes are indeed improvements. A solid testing and evaluation framework is the cornerstone of building a reliable and effective RAG system.




## Phase 4: Advanced RAG Features

Building upon the Naive RAG implementation, this phase introduces advanced techniques to significantly enhance the retrieval and generation capabilities of the system. These features address common limitations of basic RAG, such as handling complex queries, improving retrieval accuracy, and refining the context provided to the LLM.

### 4.1 Query Optimization

Query optimization techniques aim to transform the user's raw query into a more effective form for retrieval. This can involve expanding the query, rephrasing it, or breaking it down into sub-queries to better capture the user's intent and improve the chances of finding relevant documents.

**File:** `src/rag/advanced_rag.py` (This file will extend or wrap `naive_rag.py`)

```python
from src.rag.naive_rag import NaiveRAG
from src.generation.generator import HuggingFaceGenerator # Assuming this is used for query rewriting
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

class AdvancedRAG(NaiveRAG):
    """
    Extends NaiveRAG with advanced query optimization techniques.
    """
    def __init__(self, config_path: str = \'config.yaml\'):
        super().__init__(config_path)
        # Initialize an LLM specifically for query rewriting if needed
        # self.query_rewriter_llm = HuggingFaceGenerator(model_name=\'distilgpt2\', device=\'cpu\') # Example

    def _rewrite_query_with_llm(self, query: str) -> str:
        """
        Rewrites the query using an LLM to improve retrieval effectiveness.
        This can involve rephrasing, adding context, or clarifying intent.
        """
        # Example prompt for query rewriting
        prompt = f"""
        Rewrite the following question to make it more effective for searching a document database. 
        Focus on extracting key entities and concepts. Only provide the rewritten question.

        Original question: {query}
        Rewritten question:
        """
        # For a real implementation, you'd use self.query_rewriter_llm.generate(prompt)
        # For now, a placeholder:
        logging.info(f"Rewriting query: {query}")
        return query # Placeholder: return original query

    def _expand_query_with_synonyms(self, query: str) -> List[str]:
        """
        Expands the query with synonyms or related terms.
        This could use a thesaurus, word embeddings, or an LLM.
        """
        logging.info(f"Expanding query: {query}")
        # Placeholder: In a real system, use a lexical database (e.g., WordNet),
        # pre-trained word embeddings (e.g., Word2Vec, GloVe), or an LLM.
        return [query, f"{query} synonym1", f"{query} synonym2"]

    def _generate_hyde_embedding(self, query: str) -> List[float]:
        """
        Generates a Hypothetical Document Embedding (HyDE) for the query.
        This involves generating a hypothetical answer to the query using an LLM,
        and then embedding that hypothetical answer.
        """
        logging.info(f"Generating HyDE embedding for query: {query}")
        # Step 1: Generate a hypothetical answer using the LLM
        hyde_prompt = f"""
        Please write a concise, hypothetical answer to the following question. 
        Do not state that it is hypothetical. Just provide the answer.

        Question: {query}
        Answer:
        """
        # For a real implementation, you'd use self.generator.generate(hyde_prompt)
        hypothetical_answer = f"Hypothetical answer for {query}" # Placeholder

        # Step 2: Embed the hypothetical answer
        hyde_embedding = self.embedder.embed([hypothetical_answer])[0]
        return hyde_embedding

    def _decompose_query(self, query: str) -> List[str]:
        """
        Decomposes a complex question into simpler sub-queries.
        """
        logging.info(f"Decomposing query: {query}")
        # Example: "What are the benefits of RAG and how does it compare to fine-tuning?"
        # -> ["What are the benefits of RAG?", "How does RAG compare to fine-tuning?"]
        # This typically requires an LLM to identify sub-questions.
        return [query] # Placeholder

    def retrieve_optimized(self, query: str, k: int = 5, filters: Dict[str, Any] = None, 
                           query_optimization_strategy: str = "none") -> List[Dict[str, Any]]:
        """
        Retrieves documents using various query optimization strategies.
        """
        optimized_queries = [query]
        if query_optimization_strategy == "llm_rewrite":
            optimized_queries = [self._rewrite_query_with_llm(query)]
        elif query_optimization_strategy == "expansion":
            optimized_queries = self._expand_query_with_synonyms(query)
        elif query_optimization_strategy == "hyde":
            # For HyDE, we generate an embedding from a hypothetical answer
            # and use it directly for retrieval.
            hyde_embedding = self._generate_hyde_embedding(query)
            retrieved_docs = self.vector_store.search(hyde_embedding, k=k, filters=filters)
            return retrieved_docs
        elif query_optimization_strategy == "decomposition":
            sub_queries = self._decompose_query(query)
            all_retrieved_docs = []
            for sq in sub_queries:
                sq_embedding = self.embedder.embed([sq])[0]
                all_retrieved_docs.extend(self.vector_store.search(sq_embedding, k=k, filters=filters))
            # Deduplicate and re-rank if necessary
            return list({doc["content"]: doc for doc in all_retrieved_docs}.values())

        # For strategies that produce multiple queries or a single rewritten query
        all_retrieved_docs = []
        for opt_query in optimized_queries:
            query_embedding = self.embedder.embed([opt_query])[0]
            all_retrieved_docs.extend(self.vector_store.search(query_embedding, k=k, filters=filters))
        
        # Deduplicate and return
        return list({doc["content"]: doc for doc in all_retrieved_docs}.values())

    def query_optimized(self, question: str, k: int = 5, filters: Dict[str, Any] = None,
                        query_optimization_strategy: str = "none") -> str:
        """
        End-to-end query method for Advanced RAG with query optimization.
        """
        retrieved_contexts = self.retrieve_optimized(question, k=k, filters=filters, 
                                                     query_optimization_strategy=query_optimization_strategy)
        if not retrieved_contexts:
            logging.warning("No relevant contexts found after query optimization. Cannot generate an answer.")
            return "I am sorry, but I could not find enough relevant information to answer your question."
        answer = self.generate_answer(question, retrieved_contexts)
        return answer

```

**Explanation:**

*   **`AdvancedRAG` Class:** Inherits from `NaiveRAG` to reuse its core functionalities while adding new query optimization methods. This promotes code reusability and modularity.
*   **Query Rewriting using LLM (`_rewrite_query_with_llm`):** An LLM can be used to rephrase the user's query into a more search-friendly format. For instance, a conversational query like 


 "Tell me about the capital of France" could be rewritten to "Paris, France capital city information" to better match document content. This method would typically involve a separate LLM call to generate the rewritten query.
*   **Query Expansion with Synonyms/Related Terms (`_expand_query_with_synonyms`):** This technique broadens the search by including synonyms or semantically related terms to the original query. This can be achieved using lexical databases (like WordNet), pre-trained word embeddings to find nearest neighbors, or even an LLM to suggest related terms. The expanded queries are then used to retrieve more documents, increasing recall.
*   **Hypothetical Document Embeddings (HyDE) (`_generate_hyde_embedding`):** HyDE is a powerful technique where an LLM first generates a hypothetical answer to the user's query. This hypothetical answer, which is often more descriptive and semantically rich than the original query, is then embedded. The embedding of this hypothetical document is then used to search the vector store. The intuition is that a hypothetical answer will be semantically closer to relevant documents than the original query itself, leading to more accurate retrieval.
*   **Sub-query Decomposition for Complex Questions (`_decompose_query`):** For complex, multi-faceted questions (e.g., "What are the benefits of RAG and how does it compare to fine-tuning?"), decomposing them into simpler, atomic sub-queries can significantly improve retrieval. Each sub-query is then executed independently, and the results are combined. An LLM can be employed to perform this decomposition.
*   **`retrieve_optimized` Method:** This method orchestrates the application of different query optimization strategies. It takes a `query_optimization_strategy` parameter to select which technique to apply. For strategies that generate multiple queries (like expansion or decomposition), it performs multiple retrievals and then deduplicates the results.
*   **`query_optimized` Method:** This is the public interface for querying the advanced RAG system, allowing users to specify the desired query optimization strategy.

These query optimization techniques aim to bridge the semantic gap between a user's natural language query and the content of the documents in the vector store, thereby improving the relevance and comprehensiveness of the retrieved information.

### 4.2 Retrieval Enhancements

Beyond optimizing the query, enhancing the retrieval process itself can significantly improve the quality of the documents passed to the LLM. This section covers reranking, hybrid search, and context compression.

**File:** `src/retrieval/reranker.py`, `src/retrieval/hybrid_search.py`, `src/retrieval/context_compressor.py`

#### 4.2.1 Reranking (`src/retrieval/reranker.py`)

Initial retrieval from a vector store often returns documents based solely on semantic similarity to the query embedding. Reranking refines these initial results by applying a more sophisticated model (often a cross-encoder) to re-score the relevance of each retrieved document, ensuring that the most pertinent information is presented to the LLM.

```python
import abc
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class Reranker(abc.ABC):
    """
    Abstract base class for reranking retrieved documents.
    """
    @abc.abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to the query.

        Args:
            query: The original user query.
            documents: A list of retrieved documents, each with at least a 'content' key.

        Returns:
            A list of documents, reranked by relevance score (highest first).
            Each document will have an additional 'relevance_score' key.
        """
        pass

class CrossEncoderReranker(Reranker):
    """
    Reranker using a Cross-Encoder model.
    Cross-encoders are typically more accurate than bi-encoders for relevance scoring
    because they process the query and document pair together.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        # Prepare sentence pairs for the cross-encoder
        sentence_pairs = [[query, doc["content"]] for doc in documents]

        # Get scores from the cross-encoder
        scores = self.model.predict(sentence_pairs)

        # Attach scores to documents and sort
        reranked_documents = []
        for i, doc in enumerate(documents):
            doc["relevance_score"] = float(scores[i])
            reranked_documents.append(doc)

        # Sort in descending order of relevance score
        reranked_documents.sort(key=lambda x: x["relevance_score"], reverse=True)
        return reranked_documents

# Example of integrating reranking into AdvancedRAG (in advanced_rag.py)
# class AdvancedRAG(NaiveRAG):
#     def __init__(self, config_path: str = \'config.yaml\'):
#         super().__init__(config_path)
#         self.reranker = CrossEncoderReranker() # Initialize reranker

#     def retrieve_and_rerank(self, query: str, k_initial: int = 10, k_final: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
#         # First, retrieve more documents than needed (e.g., k_initial = 2*k_final)
#         initial_retrieval = self.retrieve(query, k=k_initial, filters=filters)
#         
#         # Then, rerank these documents
#         reranked_docs = self.reranker.rerank(query, initial_retrieval)
#         
#         # Return the top k_final documents after reranking
#         return reranked_docs[:k_final]

```

**Explanation:**

*   **`Reranker` (Abstract Base Class):** Defines the `rerank` method, which takes a query and a list of documents and returns them reordered by relevance.
*   **`CrossEncoderReranker`:** Uses a `CrossEncoder` model from the `sentence-transformers` library. Unlike bi-encoders (used for embeddings), cross-encoders take both the query and the document as input simultaneously, allowing them to capture more nuanced interactions and provide more accurate relevance scores. The model `cross-encoder/ms-marco-MiniLM-L-6-v2` is a good general-purpose choice. The `rerank` method calculates a score for each query-document pair and then sorts the documents based on these scores.
*   **Integration:** In an `AdvancedRAG` class, you would typically retrieve a larger set of initial documents (e.g., 10-20) using the vector store, and then pass these to the reranker to select the top 5-10 most relevant ones. This two-stage approach (fast initial retrieval + accurate reranking) balances efficiency and precision.

#### 4.2.2 Hybrid Search (`src/retrieval/hybrid_search.py`)

Hybrid search combines the strengths of keyword-based retrieval (like BM25 or TF-IDF) with semantic (dense vector) retrieval. Keyword search excels at finding exact matches and rare terms, while semantic search is good at understanding the meaning and context. Combining them often leads to more robust and comprehensive retrieval.

```python
from typing import List, Dict, Any
from src.retrieval.vector_store import VectorStore # For dense retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HybridSearch:
    """
    Combines keyword-based (e.g., TF-IDF/BM25) and dense vector retrieval.
    """
    def __init__(self, dense_vector_store: VectorStore, documents_for_keyword_index: List[Dict[str, Any]]):
        self.dense_vector_store = dense_vector_store
        self.documents_for_keyword_index = documents_for_keyword_index
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform([doc["content"] for doc in documents_for_keyword_index])

    def search(self, query: str, k: int = 5, alpha: float = 0.5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search.

        Args:
            query: The user query.
            k: The number of top documents to retrieve.
            alpha: Weight for dense retrieval (0.0 to 1.0). 1.0 is purely dense, 0.0 is purely keyword.
            filters: Optional metadata filters.

        Returns:
            A list of documents, combined and scored.
        """
        # Dense retrieval
        query_embedding = self.dense_vector_store.embedder.embed([query])[0] # Assuming embedder is accessible
        dense_results = self.dense_vector_store.search(query_embedding, k=k*2, filters=filters) # Retrieve more initially

        # Keyword retrieval (TF-IDF example)
        query_tfidf = self.vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # Combine scores (simplified reciprocal rank fusion or weighted sum)
        combined_scores = {}
        for i, doc in enumerate(self.documents_for_keyword_index):
            doc_id = doc.get("id", i) # Use a unique ID for combination
            combined_scores[doc_id] = {"content": doc["content"], "metadata": doc["metadata"], "score": keyword_scores[i]}

        for doc in dense_results:
            doc_id = doc.get("id", doc["metadata"].get("id")) # Need consistent ID
            if doc_id in combined_scores:
                # Simple weighted sum for demonstration
                combined_scores[doc_id]["score"] = (1 - alpha) * combined_scores[doc_id]["score"] + alpha * (1 - doc["distance"] / max(doc["distance"] for doc in dense_results)) # Normalize distance
            else:
                combined_scores[doc_id] = {"content": doc["content"], "metadata": doc["metadata"], "score": alpha * (1 - doc["distance"] / max(doc["distance"] for doc in dense_results))} # Normalize distance

        sorted_results = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:k]

```

**Explanation:**

*   **`HybridSearch` Class:** Takes a `dense_vector_store` (e.g., `ChromaDBVectorStore`) and the original documents (or their content) to build a keyword index. Here, `TfidfVectorizer` is used for keyword indexing, but BM25 is a more common and often more effective choice for information retrieval.
*   **Dense Retrieval:** Performs a standard semantic search using the provided `dense_vector_store`.
*   **Keyword Retrieval:** Uses `TfidfVectorizer` to transform documents and queries into TF-IDF representations, then calculates cosine similarity to find keyword-relevant documents.
*   **Score Combination:** The core of hybrid search is combining the scores from both retrieval methods. A simple weighted sum (`alpha` parameter) is shown, but more advanced techniques like Reciprocal Rank Fusion (RRF) are often used to combine rankings from different sources without requiring score normalization.
*   **Integration:** The `HybridSearch` class would be integrated into the `AdvancedRAG` pipeline, allowing the system to leverage both keyword and semantic matching for retrieval.

#### 4.2.3 Context Compression (`src/retrieval/context_compressor.py`)

Context compression aims to reduce the amount of text passed to the LLM while retaining the most important information. This is crucial for managing the LLM's context window, reducing inference costs, and improving the LLM's ability to focus on relevant details. Techniques include summarization, extractive summarization, or using a smaller, more focused window around the retrieved passages.

```python
import abc
from typing import List, Dict, Any
from transformers import pipeline

class ContextCompressor(abc.ABC):
    """
    Abstract base class for compressing retrieved contexts.
    """
    @abc.abstractmethod
    def compress(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compresses a list of documents based on their relevance to the query.

        Args:
            query: The original user query.
            documents: A list of retrieved documents, each with at least a 'content' key.

        Returns:
            A list of compressed documents, potentially shorter but retaining key information.
        """
        pass

class ExtractiveContextCompressor(ContextCompressor):
    """
    Compressor that extracts the most relevant sentences/passages from documents.
    Uses a question-answering or summarization model to identify key information.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased-distilled-squad"):
        # Using a question-answering pipeline to extract relevant spans
        self.qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

    def compress(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compressed_documents = []
        for doc in documents:
            context = doc["content"]
            # Use QA model to find the most relevant span in the document for the query
            try:
                result = self.qa_pipeline(question=query, context=context)
                # Extract the answer span as the compressed content
                compressed_content = result["answer"]
                compressed_documents.append({"content": compressed_content, "metadata": doc["metadata"]})
            except Exception as e:
                # Fallback to original content if compression fails
                print(f"Warning: Could not compress document. Error: {e}. Using original content.")
                compressed_documents.append(doc)
        return compressed_documents

# Example of integrating context compression into AdvancedRAG (in advanced_rag.py)
# class AdvancedRAG(NaiveRAG):
#     def __init__(self, config_path: str = \'config.yaml\'):
#         super().__init__(config_path)
#         self.context_compressor = ExtractiveContextCompressor() # Initialize compressor

#     def generate_answer_with_compression(self, question: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
#         compressed_contexts = self.context_compressor.compress(question, retrieved_contexts)
#         return self.generate_answer(question, compressed_contexts)

```

**Explanation:**

*   **`ContextCompressor` (Abstract Base Class):** Defines the `compress` method for reducing the size of retrieved documents.
*   **`ExtractiveContextCompressor`:** This implementation uses a pre-trained question-answering model (like `distilbert-base-uncased-distilled-squad`) to extract the most relevant span of text from each retrieved document based on the user's query. This effectively acts as a form of extractive summarization, ensuring that only the most pertinent information is passed to the LLM.
*   **Integration:** The `ContextCompressor` would be applied after retrieval and reranking, just before the documents are formatted into the prompt for the LLM. This ensures that the LLM receives a concise and highly relevant context.

### 4.3 Iterative Retrieval

Iterative retrieval, also known as multi-hop retrieval or recursive retrieval, is designed to handle complex questions that require information from multiple documents or multiple steps of reasoning. Instead of a single retrieve-then-generate step, the system performs multiple retrieval steps, often refining the query or generating intermediate answers at each step.

**File:** `src/rag/iterative_rag.py`

```python
from src.rag.advanced_rag import AdvancedRAG
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

class IterativeRAG(AdvancedRAG):
    """
    Implements iterative retrieval strategies for complex questions.
    """
    def __init__(self, config_path: str = \'config.yaml\'):
        super().__init__(config_path)

    def multi_hop_query(self, initial_question: str, max_hops: int = 2) -> str:
        """
        Performs multi-hop retrieval to answer complex questions.
        At each hop, it generates an intermediate answer or refined query.
        """
        current_question = initial_question
        all_retrieved_contexts = []

        for hop in range(max_hops):
            logging.info(f"Multi-hop retrieval - Hop {hop + 1}: Question: {current_question}")
            
            # Step 1: Retrieve documents for the current question
            retrieved_docs = self.retrieve_optimized(current_question, k=5, query_optimization_strategy="none") # Can use optimization
            all_retrieved_contexts.extend(retrieved_docs)

            if not retrieved_docs:
                logging.warning(f"No documents retrieved in hop {hop + 1}. Ending multi-hop.")
                break

            # Step 2: Generate an intermediate answer or a refined follow-up question
            # This step typically requires an LLM to reason over retrieved docs
            # and decide if more information is needed or if a final answer can be formed.
            intermediate_prompt = self.prompt_template.format_prompt(
                question=f"Based on the following context, what is the key information related to 
                          \"{current_question}\" and what follow-up question should be asked if any?",
                context=[doc["content"] for doc in retrieved_docs]
            )
            intermediate_response = self.generator.generate(intermediate_prompt)
            logging.info(f"Intermediate response (Hop {hop + 1}): {intermediate_response}")

            # Heuristic to determine if a follow-up question is needed
            # In a real system, this would involve more sophisticated parsing of LLM output
            if "follow-up question:" in intermediate_response.lower():
                current_question = intermediate_response.split("follow-up question:")[-1].strip()
                if not current_question.endswith("?"):
                    current_question += "?"
            else:
                # If no follow-up question, assume we have enough info for final answer
                logging.info(f"No follow-up question detected in hop {hop + 1}. Proceeding to final answer.")
                break
        
        # Step 3: Generate final answer based on all accumulated contexts
        final_answer = self.generate_answer(initial_question, all_retrieved_contexts)
        return final_answer

    def adaptive_retrieval_query(self, question: str) -> str:
        """
        Implements adaptive retrieval, deciding whether to retrieve more information
        based on the LLM's confidence or initial answer quality.
        """
        logging.info(f"Adaptive retrieval for question: {question}")

        # First attempt: Naive RAG
        initial_retrieved_docs = self.retrieve_optimized(question, k=5)
        initial_answer = self.generate_answer(question, initial_retrieved_docs)
        logging.info(f"Initial answer: {initial_answer}")

        # Heuristic: Check if LLM expresses uncertainty or asks for more info
        # In a real system, this would involve parsing LLM output for specific phrases
        # or using a confidence score from the LLM.
        if "I don't know" in initial_answer.lower() or "cannot answer" in initial_answer.lower():
            logging.info("LLM expressed uncertainty. Attempting second retrieval.")
            # Second attempt: Try with more documents or a different query strategy
            more_retrieved_docs = self.retrieve_optimized(question, k=10, query_optimization_strategy="expansion")
            all_docs = list({doc["content"]: doc for doc in initial_retrieved_docs + more_retrieved_docs}.values())
            final_answer = self.generate_answer(question, all_docs)
            return final_answer
        else:
            logging.info("LLM seems confident. Returning initial answer.")
            return initial_answer

    def chain_of_thought_with_retrieval(self, question: str, max_steps: int = 3) -> str:
        """
        Combines Chain-of-Thought (CoT) prompting with retrieval.
        The LLM generates intermediate thoughts/steps, and retrieval is performed
        at each step to gather necessary information.
        """
        thought_process = []
        current_state = {"question": question, "answer_so_far": "", "retrieved_info": []}

        for step in range(max_steps):
            logging.info(f"CoT Retrieval - Step {step + 1}")
            # Step 1: LLM generates a thought or plan, potentially including a retrieval query
            cot_prompt = f"""
            You are an expert problem solver. Given the question: 
            \"{current_state[\"question\"]}\"
            and any information gathered so far: 
            \"{current_state[\"answer_so_far\"]}\"
            and retrieved contexts: 
            \"{current_state[\"retrieved_info\"]}\"
            
            Think step-by-step. What is your next thought or action? 
            If you need to retrieve more information, state a clear search query like: 
            


            \"SEARCH: [your search query here]\"
            If you have enough information to answer, state your final answer like: 
            \"FINAL ANSWER: [your answer here]\"
            
            Thought:
            """
            
            llm_thought = self.generator.generate(cot_prompt, max_new_tokens=200)
            thought_process.append(f"Step {step + 1} Thought: {llm_thought}")
            logging.info(f"LLM Thought: {llm_thought}")

            if llm_thought.startswith("SEARCH:"):
                search_query = llm_thought.replace("SEARCH:", "").strip()
                retrieved_docs = self.retrieve_optimized(search_query, k=5)
                retrieved_info_str = "\n".join([doc["content"] for doc in retrieved_docs])
                current_state["retrieved_info"].append(f"Retrieved for \'{search_query}\': {retrieved_info_str}")
                current_state["answer_so_far"] += f"\n\nRetrieved information for \'{search_query}\':\n{retrieved_info_str}"
            elif llm_thought.startswith("FINAL ANSWER:"):
                final_answer = llm_thought.replace("FINAL ANSWER:", "").strip()
                thought_process.append(f"Final Answer: {final_answer}")
                return final_answer
            else:
                # If LLM doesn't explicitly search or answer, assume it's an intermediate thought
                current_state["answer_so_far"] += f"\n\nIntermediate Thought: {llm_thought}"
        
        # If max steps reached without a final answer, try to generate one with accumulated info
        logging.warning("Max CoT steps reached without a final answer. Attempting to generate final answer.")
        final_prompt = self.prompt_template.format_prompt(
            question=question,
            context=[info for info in current_state["retrieved_info"]]
        )
        final_answer = self.generator.generate(final_prompt)
        thought_process.append(f"Final Answer (after max steps): {final_answer}")
        return final_answer

```

**Explanation:**

*   **`IterativeRAG` Class:** Extends `AdvancedRAG` to incorporate multi-step reasoning and retrieval patterns.
*   **Multi-hop Retrieval (`multi_hop_query`):** This method simulates a multi-hop question answering process. For complex questions that require synthesizing information from multiple sources or across different steps, a single retrieval might not suffice. In each 


hop, the system retrieves documents, and an LLM is used to either generate an intermediate answer or formulate a new, refined follow-up question based on the retrieved context. This process repeats until a final answer can be formed or a maximum number of hops is reached. The example uses a simple heuristic to detect a follow-up question, but in practice, this would involve more sophisticated parsing of the LLM's output or a structured output format.
*   **Adaptive Retrieval (`adaptive_retrieval_query`):** This strategy involves dynamically deciding whether to perform additional retrieval steps based on the LLM's initial response. If the LLM expresses uncertainty or indicates it needs more information, the system can trigger another retrieval with a modified query or by expanding the search space. This helps to improve the robustness of the system by allowing it to self-correct.
*   **Chain-of-Thought with Retrieval (`chain_of_thought_with_retrieval`):** This advanced technique combines Chain-of-Thought (CoT) prompting with retrieval. The LLM is prompted to think step-by-step, and at each step, it can decide whether it needs to retrieve more information. If it does, it formulates a search query, and the system performs a retrieval. The retrieved information is then incorporated into the LLM's ongoing thought process. This allows the LLM to gather information as needed, leading to more accurate and well-reasoned answers for complex questions. The example demonstrates a simple protocol where the LLM explicitly states 


 "SEARCH:" for retrieval or "FINAL ANSWER:" when it has a definitive answer. This structured interaction enables the system to guide the LLM through a multi-step reasoning process, leveraging external knowledge when necessary.

These advanced retrieval techniques move beyond a simple one-shot lookup, allowing the RAG system to handle more nuanced and complex information needs by iteratively refining its understanding and gathering additional context as required.




## Phase 5: Modular RAG Architecture

As RAG systems grow in complexity and integrate more advanced features, a modular architecture becomes paramount. This phase focuses on designing a system where components are loosely coupled, easily interchangeable, and can be configured dynamically. This approach enhances flexibility, maintainability, and scalability, allowing for rapid experimentation and adaptation to new requirements.

### 5.1 Modular Components

A truly modular RAG system treats each functional block (chunking, embedding, retrieval, generation, reranking, etc.) as a pluggable component. This allows developers to swap out different implementations (e.g., a new embedding model, a different vector store, or an alternative LLM) with minimal changes to the overall pipeline. This can be achieved through abstract interfaces and a configuration-driven approach.

**Refactoring Strategy:**

Instead of direct instantiation of concrete classes within the `NaiveRAG` or `AdvancedRAG` classes, we will introduce a factory pattern or a dependency injection mechanism. The `config.yaml` file will play a central role in defining which specific implementation of each component should be used at runtime.

**Conceptual Example (Refactoring `NaiveRAG` initialization):**

```python
# In src/rag/rag_factory.py

from src.chunking.text_splitter import FixedSizeTextSplitter, SentenceTextSplitter
from src.embedding.embedder import SentenceTransformerEmbedder # , OpenAIEmbedder
from src.retrieval.vector_store import ChromaDBVectorStore, FAISSVectorStore
from src.generation.generator import HuggingFaceGenerator # , OpenAIGenerator
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.context_compressor import ExtractiveContextCompressor

class RAGComponentFactory:
    @staticmethod
    def get_text_splitter(config: dict):
        splitter_type = config.get("type", "fixed_size")
        if splitter_type == "fixed_size":
            return FixedSizeTextSplitter(
                chunk_size=config.get("chunk_size", 500),
                chunk_overlap=config.get("chunk_overlap", 50)
            )
        elif splitter_type == "sentence":
            return SentenceTextSplitter()
        else:
            raise ValueError(f"Unknown text splitter type: {splitter_type}")

    @staticmethod
    def get_embedder(config: dict):
        embedder_type = config.get("type", "sentence_transformer")
        if embedder_type == "sentence_transformer":
            return SentenceTransformerEmbedder(
                model_name=config.get("model_name", "all-MiniLM-L6-v2"),
                cache_dir=config.get("cache_dir", "./embedding_cache")
            )
        # elif embedder_type == "openai":
        #     return OpenAIEmbedder(model_name=config.get("model_name", "text-embedding-ada-002"))
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")

    @staticmethod
    def get_vector_store(config: dict, embedder_instance):
        store_type = config.get("type", "chromadb")
        if store_type == "chromadb":
            return ChromaDBVectorStore(
                path=config.get("path", "./chroma_db"),
                collection_name=config.get("collection_name", "rag_collection"),
                # Pass embedder instance if ChromaDB needs it for internal use
                # or ensure its internal embedder is consistent
            )
        elif store_type == "faiss":
            embedding_dimension = config.get("embedding_dimension", 384) # Must be known
            return FAISSVectorStore(embedding_dimension=embedding_dimension)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")

    @staticmethod
    def get_generator(config: dict):
        generator_type = config.get("type", "huggingface")
        if generator_type == "huggingface":
            return HuggingFaceGenerator(
                model_name=config.get("model_name", "distilgpt2"),
                device=config.get("device", "cpu")
            )
        # elif generator_type == "openai":
        #     return OpenAIGenerator(model_name=config.get("model_name", "gpt-3.5-turbo"))
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

    @staticmethod
    def get_reranker(config: dict):
        reranker_type = config.get("type", "cross_encoder")
        if reranker_type == "cross_encoder":
            return CrossEncoderReranker(model_name=config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")

    @staticmethod
    def get_context_compressor(config: dict):
        compressor_type = config.get("type", "extractive")
        if compressor_type == "extractive":
            return ExtractiveContextCompressor(model_name=config.get("model_name", "distilbert-base-uncased-distilled-squad"))
        else:
            raise ValueError(f"Unknown compressor type: {compressor_type}")

# In src/rag/modular_rag.py (New main RAG class)

import yaml
import os
import logging
from typing import List, Dict, Any

# Assuming RAGComponentFactory is defined as above
# from src.rag.rag_factory import RAGComponentFactory

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

class ModularRAG:
    """
    A modular and configurable RAG pipeline.
    """
    def __init__(self, config_path: str = \'config.yaml\'):
        self.config = self._load_config(config_path)
        self._initialize_components()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, \'r\') as f:
            return yaml.safe_load(f)

    def _initialize_components(self):
        # Initialize core components
        self.text_splitter = RAGComponentFactory.get_text_splitter(self.config.get("text_splitter", {}))
        self.embedder = RAGComponentFactory.get_embedder(self.config.get("embedder", {}))
        # Vector store might need embedder instance, so initialize embedder first
        self.vector_store = RAGComponentFactory.get_vector_store(self.config.get("vector_store", {}), self.embedder)
        self.generator = RAGComponentFactory.get_generator(self.config.get("generator", {}))
        
        # Initialize optional components
        self.reranker = None
        if "reranker" in self.config:
            self.reranker = RAGComponentFactory.get_reranker(self.config["reranker"])

        self.context_compressor = None
        if "context_compressor" in self.config:
            self.context_compressor = RAGComponentFactory.get_context_compressor(self.config["context_compressor"])

        # Add other components like PromptTemplate, DocumentLoader etc.
        # self.prompt_template = PromptTemplate() # Assuming a default template or configurable
        # self.document_loader = DocumentLoader() # Assuming a default loader or configurable

    def query(self, question: str, k: int = 5, filters: Dict[str, Any] = None) -> str:
        # This method would now dynamically use the initialized components
        # based on the configuration and potentially implement routing logic.
        logging.info(f"Querying with Modular RAG: {question}")

        # 1. Retrieve
        query_embedding = self.embedder.embed([question])[0]
        retrieved_docs = self.vector_store.search(query_embedding, k=k*2, filters=filters) # Retrieve more for reranking

        # 2. Rerank (if configured)
        if self.reranker:
            retrieved_docs = self.reranker.rerank(question, retrieved_docs)
            retrieved_docs = retrieved_docs[:k] # Take top k after reranking

        # 3. Compress Context (if configured)
        if self.context_compressor:
            retrieved_docs = self.context_compressor.compress(question, retrieved_docs)

        if not retrieved_docs:
            logging.warning("No relevant contexts found.")
            return "I am sorry, but I could not find enough relevant information to answer your question."

        # 4. Generate Answer
        context_contents = [doc["content"] for doc in retrieved_docs]
        # Assuming prompt_template is initialized in _initialize_components
        formatted_prompt = self.prompt_template.format_prompt(question, context_contents)
        answer = self.generator.generate(formatted_prompt)
        return answer

```

**Explanation:**

*   **`RAGComponentFactory`:** This static factory class centralizes the creation of different RAG components. Instead of `ModularRAG` directly knowing how to instantiate `FixedSizeTextSplitter` or `SentenceTransformerEmbedder`, it delegates this responsibility to the factory. The factory reads the component-specific configuration and returns the appropriate instance.
*   **`ModularRAG` Class:** This new main RAG class is responsible for loading the overall configuration and then using the `RAGComponentFactory` to initialize all its sub-components. This makes the `ModularRAG` class much cleaner and easier to manage. It also dynamically initializes optional components like `reranker` and `context_compressor` only if they are specified in the configuration.
*   **Plugin System for Components:** The factory pattern effectively creates a plugin system. To add a new embedding model, you would simply create a new `Embedder` implementation, add a corresponding entry in the `RAGComponentFactory`, and update your `config.yaml`. No changes are needed in the core `ModularRAG` logic.
*   **Routing Module for Query Dispatch (Conceptual):** For more advanced modularity, especially in systems handling diverse query types or use cases, a 


routing module would be beneficial. This module would analyze the incoming query and determine the most appropriate RAG pipeline or combination of components to use. For example, a simple query might go through a basic RAG pipeline, while a complex, multi-hop question might trigger the `IterativeRAG` flow. This routing could be rule-based or even driven by an LLM that classifies the query type.

**Example of Routing Logic (Conceptual addition to `ModularRAG` or a new `Router` class):**

```python
# Inside ModularRAG or a separate Router class

# def route_query(self, question: str, k: int = 5, filters: Dict[str, Any] = None) -> str:
#     # Example: Simple rule-based routing
#     if "compare" in question.lower() or "difference" in question.lower():
#         logging.info("Routing to multi-hop retrieval for comparison query.")
#         # Assuming multi_hop_query is available and initialized
#         return self.multi_hop_query(question, max_hops=3)
#     elif len(question.split()) > 10 and "how" in question.lower():
#         logging.info("Routing to CoT with retrieval for complex 'how-to' query.")
#         # Assuming chain_of_thought_with_retrieval is available and initialized
#         return self.chain_of_thought_with_retrieval(question, max_steps=5)
#     else:
#         logging.info("Routing to standard RAG pipeline.")
#         return self.query(question, k=k, filters=filters)

```

*   **Memory Module for Conversation History:** For conversational RAG systems, maintaining a memory of past interactions is crucial. A dedicated memory module would store the conversation history (previous turns, retrieved documents, generated answers) and make it accessible to the RAG pipeline. This memory can then be used to contextualize new queries, resolve anaphora, or guide subsequent retrieval and generation steps. This module would typically involve a data structure to store turns and potentially an LLM to summarize or condense the history if it exceeds a certain length.

*   **Task Adapter for Different Use Cases:** A task adapter allows the RAG system to be easily configured and optimized for different downstream applications (e.g., question answering, summarization, content generation). This could involve: 
    *   **Different Prompt Templates:** Using specific prompt templates tailored to the task.
    *   **Component Selection:** Activating or deactivating certain components (e.g., reranking might be less critical for simple summarization).
    *   **Parameter Tuning:** Adjusting parameters like `k` (number of retrieved documents) or LLM generation parameters (`temperature`, `max_new_tokens`).
    *   **Post-processing:** Applying task-specific post-processing to the generated output.

By making all components hot-swappable via configuration, the RAG system becomes highly adaptable. This modularity is not just about code organization; it's about enabling rapid iteration, A/B testing of different strategies, and fine-tuning the system for specific performance goals.

### 5.2 Production Features

Transitioning a RAG system from a prototype to a production-ready application requires addressing concerns beyond core functionality, such as performance, scalability, reliability, and observability. This section outlines key production features.

#### 5.2.1 Async Processing for Better Performance

Many operations in a RAG pipeline, such as embedding generation, vector store lookups, and LLM inference, can be I/O-bound or computationally intensive. Implementing asynchronous processing allows the system to handle multiple requests concurrently without blocking, significantly improving throughput and responsiveness.

**Implementation:** Python's `asyncio` library, combined with `await` and `async` keywords, can be used. Libraries like `httpx` (for HTTP requests) and `faiss-async` (if available, or custom async wrappers for FAISS) would be beneficial. For LLM calls, most modern SDKs (e.g., OpenAI Python client) provide asynchronous interfaces.

```python
# Conceptual example for async embedding (requires async embedder)
# async def async_embed_texts(embedder, texts: List[str]) -> List[List[float]]:
#     # This would call an async version of the embedder.embed method
#     return await embedder.aembed(texts)

# Conceptual example for async retrieval
# async def async_retrieve(vector_store, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
#     return await vector_store.asearch(query_embedding, k)

# Conceptual example for async generation
# async def async_generate(generator, prompt: str) -> str:
#     return await generator.agenerate(prompt)

# In a FastAPI endpoint (example)
# from fastapi import FastAPI
# app = FastAPI()
# @app.post("/query")
# async def query_rag(request: QueryRequest):
#     # ... initialize components ...
#     query_embedding = await rag.embedder.aembed([request.question])[0]
#     retrieved_docs = await rag.vector_store.asearch(query_embedding, k=request.k)
#     answer = await rag.generator.agenerate(rag.prompt_template.format_prompt(request.question, retrieved_docs))
#     return {"answer": answer}

```

#### 5.2.2 Caching Layer for Embeddings and Results

Caching frequently accessed data can drastically reduce latency and computational costs. Two primary areas for caching in RAG are:

*   **Embedding Cache:** Store embeddings of previously processed documents or queries. This avoids recomputing embeddings for identical inputs. Our `Embedder` implementations already include a basic in-memory cache. For production, consider a persistent cache (e.g., Redis, Memcached) or a disk-based cache.
*   **Result Cache:** Store the full RAG pipeline results (query, retrieved documents, generated answer) for common or identical queries. If a query has been processed before, the cached answer can be returned immediately. This is particularly effective for high-volume, repetitive queries.

**Implementation:**

*   **In-memory:** Python dictionaries (as shown in `embedder.py`).
*   **Persistent:** Libraries like `redis-py` for Redis, or `sqlite3` for a simple disk-based cache. Decorators can be used to easily add caching to methods.

```python
# Example of a simple decorator for caching
import functools

def cache_results(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs) # Simple key generation
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

# Apply to methods in NaiveRAG or ModularRAG
# class ModularRAG:
#     @cache_results
#     def query(self, question: str, k: int = 5, filters: Dict[str, Any] = None) -> str:
#         # ... existing query logic ...
#         pass

```

#### 5.2.3 API Server using FastAPI

To expose the RAG system as a service, a robust and performant API server is essential. FastAPI is an excellent choice due to its high performance (built on Starlette and Pydantic), automatic API documentation (Swagger UI/ReDoc), and ease of use for building asynchronous APIs.

**File:** `main.py` (or `app.py`)

```python
# In main.py

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import logging

from src.rag.modular_rag import ModularRAG # Assuming ModularRAG is your main class

logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

app = FastAPI(title="RAG System API", version="1.0.0")

# Initialize your RAG system globally or as a dependency
# For simplicity, initializing here. In production, consider dependency injection.
rag_system = ModularRAG(config_path=\'config.yaml\')

class QueryRequest(BaseModel):
    question: str
    k: int = 5
    filters: Dict[str, Any] = None

class IndexRequest(BaseModel):
    document_paths: List[str]

@app.post("/query")
async def query_rag_endpoint(request: QueryRequest):
    """
    Query the RAG system with a question and retrieve an answer.
    """
    try:
        answer = rag_system.query(request.question, k=request.k, filters=request.filters)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        logging.error(f"Error during query: {e}")
        return {"error": str(e)}, 500

@app.post("/index")
async def index_documents_endpoint(request: IndexRequest):
    """
    Index new documents into the RAG system.
    """
    try:
        rag_system.index_documents(request.document_paths)
        return {"status": "success", "message": f"Successfully indexed {len(request.document_paths)} documents."}
    except Exception as e:
        logging.error(f"Error during indexing: {e}")
        return {"error": str(e)}, 500

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

```

To run this API server, save the code as `main.py` (or `app.py`) in your project root and execute:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

This will start the FastAPI server, accessible at `http://0.0.0.0:8000`. You can then interact with the API using tools like `curl` or Postman, or through the automatically generated Swagger UI at `http://0.0.0.0:8000/docs`.

#### 5.2.4 Monitoring and Logging with Structured Logs

For production systems, comprehensive monitoring and logging are critical for understanding system behavior, debugging issues, and tracking performance. Structured logging (e.g., JSON format) makes logs easier to parse and analyze with log management tools.

**Implementation:**

*   **Python `logging` module:** Configure it to output JSON. Libraries like `python-json-logger` can help.
*   **Metrics:** Use libraries like `Prometheus client` to expose application metrics (e.g., request latency, error rates, number of indexed documents, LLM token usage). These metrics can then be scraped by Prometheus and visualized in Grafana.
*   **Tracing:** For distributed systems, distributed tracing (e.g., OpenTelemetry) helps track requests across multiple services.

```python
# Example of structured logging setup (in a logging_config.py or similar)
import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "pathname": record.pathname,
            "lineno": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)
        return json.dumps(log_record)

# In your main application setup:
# handler = logging.StreamHandler()
# handler.setFormatter(JsonFormatter())
# logging.getLogger().addHandler(handler)
# logging.getLogger().setLevel(logging.INFO)

# Example usage:
# logging.info("Query processed successfully", extra={"query": question, "answer_length": len(answer)})
# logging.error("Failed to retrieve documents", extra={"error_details": str(e), "query": question})

```

#### 5.2.5 Configuration Management System

As the system grows, managing configurations across different environments (development, staging, production) becomes complex. A robust configuration management system allows for externalizing parameters, managing secrets, and dynamically loading settings.

**Implementation:**

*   **YAML/JSON files:** As used in our `config.yaml` example, simple and human-readable.
*   **Environment Variables:** Best for sensitive information (API keys) and environment-specific settings.
*   **Dedicated Libraries:** Libraries like `Dynaconf`, `ConfigObj`, or `Pydantic Settings` can provide more advanced features like layered configurations, validation, and secret management.
*   **Cloud-native solutions:** For cloud deployments, consider services like AWS Systems Manager Parameter Store, Azure Key Vault, or Google Secret Manager.

By implementing these production-grade features, the RAG system becomes more reliable, scalable, and easier to operate in a real-world environment.




## Phase 6: Advanced Applications

Beyond general-purpose question answering, RAG systems can be tailored and optimized for specific domains and use cases. This phase explores how to adapt the modular RAG architecture for specialized applications and discusses performance optimization techniques.

### 6.1 Domain-Specific RAG

Adapting a RAG system to a specific domain (e.g., legal, medical, code) often involves using domain-specific data, fine-tuning components, and sometimes incorporating multi-modal information.

#### 6.1.1 Code RAG for Programming Questions

Code RAG systems are designed to answer programming-related questions by retrieving relevant code snippets, documentation, or forum discussions. This is particularly useful for developers seeking help with APIs, debugging, or understanding complex codebases.

**Key Adaptations:**

*   **Specialized Chunking:** Code often has a hierarchical structure (functions, classes, methods). Chunking strategies should respect this structure, perhaps by splitting code into logical blocks rather than fixed-size text chunks. Abstract Syntax Tree (AST) parsing can be used for more intelligent code chunking.
*   **Code-Specific Embeddings:** While general-purpose embeddings can work, models pre-trained on large code corpora (e.g., CodeBERT, GraphCodeBERT, or models fine-tuned on code-related tasks) can generate more semantically rich representations of code. These models understand code syntax, structure, and common programming patterns.
*   **Vector Store for Code:** The vector store would index code snippets, function definitions, class implementations, and associated documentation. Metadata could include programming language, repository, file path, and function signatures.
*   **Prompt Engineering for Code:** Prompts for the LLM would guide it to generate code examples, explain errors, or provide best practices. For example:

    ```
    """
    Given the following Python code snippets and documentation, answer the question.
    If the question asks for code, provide a complete and runnable example.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    ```

*   **Retrieval of Code-Related Artifacts:** Beyond just code, the system might retrieve relevant error messages, stack traces, or links to official documentation.

#### 6.1.2 Medical RAG with Specialized Embeddings

Medical RAG systems provide information from clinical notes, research papers, or medical guidelines. Accuracy and trustworthiness are paramount in this domain.

**Key Adaptations:**

*   **Curated Data Sources:** Rely on authoritative medical databases, peer-reviewed journals, and clinical guidelines. Data preprocessing might involve anonymization of patient data.
*   **Medical-Specific Embeddings:** Use biomedical or clinical embedding models (e.g., BioBERT, ClinicalBERT, PubMedBERT) that are pre-trained on vast amounts of medical text. These models capture the nuances of medical terminology and concepts more effectively than general-purpose models.
*   **Strict Prompt Engineering:** Prompts must emphasize factual accuracy, citation of sources, and disclaimers about not providing medical advice. The LLM should be instructed to only use information present in the retrieved context.
*   **Evaluation:** Beyond standard RAG metrics, medical RAG systems require rigorous evaluation by medical professionals to ensure safety and accuracy.

#### 6.1.3 Multi-modal RAG for Images + Text

Multi-modal RAG extends the concept to include information from different modalities, such as images, videos, or audio, alongside text. For example, answering questions about an image by retrieving relevant textual descriptions or other images.

**Key Adaptations:**

*   **Multi-modal Embeddings:** Use models that can generate embeddings for multiple modalities (e.g., CLIP for image-text pairs, or specialized models for video/audio). These models project different modalities into a shared embedding space, allowing for cross-modal retrieval.
*   **Multi-modal Vector Store:** The vector store would store embeddings of images, text descriptions, and potentially other modalities. Queries could be text-based (e.g., "Describe this image") or image-based (e.g., "Find images similar to this one").
*   **Cross-modal Retrieval:** Given a text query, retrieve relevant images and their captions. Given an image query, retrieve relevant text descriptions. The LLM would then synthesize information from both modalities.
*   **Prompt Engineering:** Prompts would need to incorporate visual descriptions or references to images. For example, an LLM might be given an image caption and asked to generate a detailed description of the image, or answer questions about its content.

#### 6.1.4 Conversational RAG with Memory

Conversational RAG focuses on maintaining context and coherence over multiple turns in a dialogue. This requires integrating a memory component into the RAG pipeline.

**Key Adaptations:**

*   **Conversation History Management:** A memory module (as discussed in Phase 5) stores the turns of the conversation. This history can be summarized or condensed to fit within the LLM's context window.
*   **Query Rewriting with History:** The current user query is often ambiguous without the preceding conversation. An LLM can be used to rewrite the current query, incorporating relevant information from the conversation history to make it self-contained and unambiguous for retrieval.
*   **Adaptive Retrieval:** The system might decide to retrieve documents based on the entire conversation history, or only the most recent turns, depending on the query type.
*   **LLM for Dialogue Management:** The LLM not only generates answers but also manages the flow of the conversation, asks clarifying questions, and tracks user intent.

### 6.2 Performance Optimization

Optimizing the performance of a RAG system is crucial for real-time applications and cost efficiency. This involves reducing latency, improving throughput, and managing resource utilization.

#### 6.2.1 Embedding Quantization

Embedding quantization reduces the memory footprint and computational cost of storing and searching embeddings. This is achieved by representing embeddings with fewer bits (e.g., 8-bit integers instead of 32-bit floats) or by compressing them.

**Techniques:**

*   **Product Quantization (PQ):** Divides the embedding vector into sub-vectors and quantizes each sub-vector independently. This is commonly used in FAISS.
*   **Scalar Quantization (SQ):** Scales and quantizes each dimension of the embedding vector independently.
*   **Binary Quantization:** Represents each dimension as a single bit (0 or 1).

**Benefits:** Reduced memory usage, faster retrieval (especially for approximate nearest neighbor search), and lower storage costs.

**Trade-offs:** Can lead to a slight reduction in retrieval accuracy, depending on the quantization method and compression ratio.

#### 6.2.2 GPU Acceleration Where Applicable

Many components of a RAG system, particularly embedding generation and LLM inference, can be significantly accelerated using GPUs.

**Areas for GPU Acceleration:**

*   **Embedding Models:** Large embedding models (e.g., from `sentence-transformers` or `transformers`) can leverage GPUs for faster batch embedding generation.
*   **Vector Stores:** FAISS and other vector databases offer GPU-enabled indices (e.g., `faiss.IndexFlatL2GPU`) for extremely fast similarity search on large datasets.
*   **LLM Inference:** Running LLMs on GPUs is standard practice for production deployments, as it drastically reduces inference latency. Libraries like `transformers` and `PyTorch` allow easy deployment on CUDA-enabled GPUs.

**Implementation:** Ensure your environment has CUDA installed and that your libraries (PyTorch, TensorFlow, FAISS) are built with GPU support. Then, specify `device='cuda'` (or the appropriate device ID) when initializing models or indices.

#### 6.2.3 Optimize Chunk Size Dynamically

The optimal chunk size is highly dependent on the nature of the documents and the types of queries. A fixed chunk size may not be ideal for all scenarios.

**Dynamic Optimization Strategies:**

*   **Adaptive Chunking:** Instead of fixed sizes, chunk based on semantic boundaries (e.g., paragraphs, sections, topics). This can be achieved using NLP techniques or by leveraging document structure (e.g., Markdown headings).
*   **Experimentation:** Conduct A/B tests with different chunk sizes and overlaps on your specific dataset and query types to find the optimal configuration that balances retrieval recall and precision.
*   **LLM-Guided Chunking:** An LLM could potentially identify optimal chunk boundaries or summarize sections to create more effective chunks.

#### 6.2.4 Implement Selective Retrieval

Selective retrieval aims to reduce the number of documents that need to be processed by the LLM by intelligently filtering or prioritizing them.

**Techniques:**

*   **Pre-filtering:** Use metadata filters (as shown in `vector_store.py`) to narrow down the search space before performing vector similarity search. For example, if a query specifies a date range, only search documents within that range.
*   **Re-ranking:** As discussed in Phase 4, re-ranking helps prioritize the most relevant documents, allowing you to pass only the top `k` documents to the LLM.
*   **Document Summarization:** Instead of passing the full retrieved chunks, pass a summary of each chunk to the LLM, especially if the chunks are long. This reduces the token count and helps the LLM focus.
*   **Adaptive `k`:** Dynamically adjust the number of retrieved documents (`k`) based on the complexity of the query or the confidence of initial retrieval results.

#### 6.2.5 Add Result Caching Strategies

Beyond the basic caching discussed in Phase 5, more sophisticated caching strategies can be employed:

*   **Semantic Caching:** Instead of exact match caching, use semantic similarity to determine if a new query is sufficiently similar to a previously cached query. If so, return the cached answer. This requires embedding the incoming query and comparing it to embeddings of cached queries.
*   **Tiered Caching:** Implement multiple layers of caching (e.g., in-memory for hot data, Redis for frequently accessed data, disk for less frequent but persistent data).
*   **Cache Invalidation:** Develop strategies to invalidate cached results when underlying documents are updated or deleted to ensure freshness of information.

These advanced applications and performance optimizations demonstrate the versatility and potential of RAG systems. By carefully selecting and implementing these techniques, you can build highly specialized, efficient, and robust RAG solutions for a wide range of real-world problems.




## Best Practices for Using Claude in Cursor

This guide is designed to be highly compatible with an AI-powered coding assistant like Claude in Cursor. Leveraging such a tool effectively can significantly accelerate development, improve code quality, and streamline the debugging process. Here are some best practices for maximizing your productivity:

### 1. Iterative Development

Instead of attempting to build complex features in one go, break down tasks into smaller, manageable iterations. This approach is particularly effective when working with AI assistants.

*   **Start with Simple Implementations:** Begin with the most basic version of a component (e.g., a fixed-size text splitter without overlap, a simple embedder). Get this working and tested.
*   **Ask Claude to Add Features Incrementally:** Once the basic functionality is stable, prompt Claude to add features one by one. For example, "Now, add chunk overlap to the `FixedSizeTextSplitter`," or "Extend the `Embedder` to support caching." This allows you to review and integrate changes in small, digestible chunks.
*   **Test Each Component Before Moving On:** After each incremental addition, run your unit tests and perform quick sanity checks. This ensures that new features don't break existing functionality and helps isolate bugs to the most recent changes.

### 2. Code Review Prompts

Use Claude as an intelligent code reviewer. It can identify potential issues that might be missed during manual review, especially in large codebases.

*   **General Review:** Prompt Claude with a request like:
    ```
    "Review this implementation for:
    1. Potential bugs
    2. Performance bottlenecks
    3. Best practices violations
    4. Missing error handling"
    ```
    Be specific about the areas you want it to focus on.
*   **Security Review:** For sensitive components, ask for a security-focused review:
    ```
    "Review this code for common security vulnerabilities (e.g., injection flaws, insecure deserialization, improper error handling) and suggest mitigations."
    ```
*   **Readability and Maintainability:** Request feedback on code style and structure:
    ```
    "Assess the readability and maintainability of this module. Suggest improvements for clarity, modularity, and adherence to PEP 8."
    ```

### 3. Debugging Prompts

When encountering errors, Claude can be an invaluable debugging partner. Provide it with the error message, stack trace, and relevant code snippets.

*   **Error Analysis and Fix:** A common and effective prompt is:
    ```
    "This code produces [error]. Help me:
    1. Understand why this happens
    2. Fix the issue
    3. Add tests to prevent regression"
    ```
    This guides Claude to not only fix the immediate problem but also to provide a deeper understanding and preventative measures.
*   **Logical Error Identification:** If the code runs but produces incorrect results, describe the unexpected behavior:
    ```
    "The `retrieve` method is returning irrelevant documents for this query. I suspect a logical error in the similarity calculation or filtering. Can you help me trace the data flow and identify the problem?"
    ```
*   **Performance Debugging:** For performance issues, provide profiling results or describe the slow operation:
    ```
    "This function is a performance bottleneck. Based on the profiling data, it spends most of its time in X. Suggest ways to optimize its execution, focusing on algorithmic improvements or library usage."
    ```

### 4. Documentation Prompts

Automating documentation generation can save significant time and ensure consistency. Claude can help generate docstrings, comments, and even API documentation.

*   **Generate Docstrings:** For a given function or class, ask:
    ```
    "Add comprehensive docstrings and create API documentation for this module using Google style docstrings"
    ```
    Specify the desired docstring format (e.g., Google, NumPy, reStructuredText).
*   **Explain Complex Logic:** If a section of code is particularly intricate, ask Claude to explain it:
    ```
    "Explain the logic of this `multi_hop_query` function in simple terms, focusing on its step-by-step execution and purpose."
    ```
    Then, you can use this explanation to write inline comments or higher-level documentation.
*   **Generate README Content:** For project-level documentation, prompt Claude to draft sections of your `README.md`:
    ```
    "Draft a section for the README.md explaining how to set up the project and install dependencies, including commands for virtual environment activation."
    ```

By integrating Claude (or similar AI assistants) into your development workflow, you can significantly enhance your efficiency, improve code quality, and accelerate the development of complex RAG systems. Treat it as an intelligent pair programmer, providing clear instructions and iterative feedback.




## Testing Checklist

Thorough testing is paramount for building a reliable and high-performing RAG system. This checklist provides a structured approach to ensure comprehensive coverage across different aspects of your implementation.

*   **Unit Tests for Each Module:**
    *   **Text Splitter:** Verify correct chunking for various text lengths, overlaps, and edge cases (empty text, single word). Ensure metadata is preserved.
    *   **Embedder:** Confirm correct embedding dimensions. Test batch processing efficiency. Validate caching mechanism (e.g., ensure embeddings are not recomputed for identical inputs). Check for consistent similarity scores for semantically similar texts.
    *   **Vector Store:** Test CRUD (Create, Read, Update, Delete) operations. Verify similarity search accuracy (e.g., top-k results are indeed the most similar). Validate metadata filtering functionality. Assess performance for adding and searching large numbers of vectors.
    *   **Generator:** Check prompt formatting. Verify context handling (e.g., LLM uses provided context and does not hallucinate). Test different generation parameters (temperature, max tokens). Ensure streaming works as expected.

*   **Integration Tests for Full Pipeline:**
    *   **Ingestion Pipeline:** Test the entire document indexing process (loading, chunking, embedding, adding to vector store) with various document types and sizes.
    *   **Query Pipeline:** Validate the end-to-end query flow, from user question to generated answer, ensuring all components interact correctly.
    *   **Advanced Features:** Test query optimization (rewriting, expansion, HyDE, decomposition), retrieval enhancements (reranking, hybrid search, context compression), and iterative retrieval (multi-hop, adaptive, CoT) with relevant complex queries.

*   **Performance Benchmarks:**
    *   **Latency:** Measure the time taken for document indexing and query processing under various loads.
    *   **Throughput:** Determine the number of queries or documents processed per unit of time.
    *   **Scalability:** Evaluate how the system performs as the number of documents and concurrent users increases.
    *   **Component-level profiling:** Identify bottlenecks in specific modules (e.g., embedding generation, vector search, LLM inference).

*   **Memory Usage Profiling:**
    *   Monitor memory consumption during indexing and querying, especially for large models and datasets, to prevent out-of-memory errors.
    *   Identify memory leaks or inefficient data structures.

*   **Error Handling Validation:**
    *   Test how the system responds to invalid inputs (e.g., non-existent file paths, malformed queries).
    *   Verify graceful degradation and informative error messages when external services (LLM API, vector database) are unavailable or return errors.

*   **Edge Case Coverage:**
    *   Test with very short or very long documents/queries.
    *   Test with documents containing unusual characters, formatting, or languages.
    *   Test scenarios where no relevant documents are found.
    *   Test with empty contexts or very small contexts.

*   **Human Evaluation:**
    *   **Answer Quality:** Have human evaluators assess the correctness, faithfulness, relevance, and fluency of generated answers.
    *   **Retrieval Quality:** Manually inspect retrieved documents for relevance to the query.

*   **A/B Testing:**
    *   Compare different RAG configurations (e.g., different chunk sizes, embedding models, rerankers) to quantitatively assess their impact on performance metrics.

By systematically working through this checklist, you can build confidence in the robustness and effectiveness of your RAG implementation.




## Resources & References

This section provides a curated list of datasets, pre-trained models, and evaluation metrics that are essential for building, testing, and evaluating RAG systems. These resources will help you acquire the necessary data and tools to implement and benchmark your RAG pipeline effectively.

### 1. Datasets for Testing:

*   **Natural Questions (simplified subset):** A dataset of real user questions posed to the Google search engine, along with Wikipedia articles that contain the answers. A simplified subset is often used for RAG evaluation, where the task is to retrieve the relevant passage and generate an answer. [Link to dataset (e.g., Hugging Face Datasets)](https://huggingface.co/datasets/natural_questions)
*   **MS MARCO (Microsoft Machine Reading Comprehension):** A large-scale dataset for machine reading comprehension, question answering, and passage ranking. It contains queries and passages, with human-annotated relevance judgments. [Link to dataset (e.g., MS MARCO official site)](https://microsoft.github.io/MSMARCO/)
*   **Your own domain-specific documents:** For building a RAG system tailored to a particular domain (e.g., legal, medical, internal company documents), collecting and curating a dataset of relevant documents is crucial. This often involves web scraping, document parsing, and manual annotation.

### 2. Pre-trained Models:

*   **Embeddings:**
    *   `sentence-transformers/all-MiniLM-L6-v2`: A highly efficient and effective sentence embedding model. It maps sentences and paragraphs to a 384-dimensional dense vector space and is suitable for a wide range of semantic similarity tasks. [Link to Hugging Face Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    *   Other options include larger Sentence-BERT models, OpenAI embeddings (e.g., `text-embedding-ada-002`), or domain-specific embeddings (e.g., BioBERT for medical text).
*   **Reranking:**
    *   `cross-encoder/ms-marco-MiniLM-L-6-v2`: A cross-encoder model fine-tuned on the MS MARCO dataset for passage re-ranking. It takes a query and a passage pair and outputs a relevance score. [Link to Hugging Face Model Card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
*   **Generation (LLMs):**
    *   `mistralai/Mistral-7B-Instruct-v0.1`: A powerful 7-billion parameter instruction-tuned language model from Mistral AI. It offers a good balance of performance and efficiency for various generation tasks. [Link to Hugging Face Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
    *   Other options include larger open-source models (e.g., Llama 2, Falcon) or proprietary models (e.g., OpenAI GPT series, Anthropic Claude).

### 3. Evaluation Metrics:

Evaluating a RAG system involves assessing both the quality of retrieval and the quality of generation. A combination of automatic and human evaluation metrics is recommended.

*   **Retrieval Metrics:**
    *   **Mean Reciprocal Rank (MRR):** Measures the effectiveness of a search engine or recommender system. It is the average of the reciprocal ranks of the first relevant item.
    *   **Normalized Discounted Cumulative Gain (NDCG):** Measures the usefulness, or gain, of a document based on its position in the result list. The gain is accumulated from the top of the result list to the bottom, with the gain of each result discounted at lower ranks.
    *   **Recall@K:** The proportion of queries for which at least one relevant document is found within the top K retrieved documents.
    *   **Precision@K:** The proportion of retrieved documents within the top K that are relevant.
    *   **Hit Rate:** The percentage of queries for which the correct answer (or a document containing it) is found among the retrieved documents.

*   **Generation Metrics:**
    *   **BLEU (Bilingual Evaluation Understudy):** A metric for evaluating the quality of text which has been machine-translated from one natural language to another. It measures the n-gram overlap between the generated text and reference texts.
    *   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** A set of metrics for evaluating automatic summarization and machine translation. It measures the overlap of n-grams, word sequences, and word pairs between the generated text and reference texts.
    *   **BERTScore:** A more advanced metric that leverages pre-trained BERT embeddings to compute a similarity score between the generated text and reference text, capturing semantic similarity more effectively than traditional n-gram based metrics.
    *   **LLM-as-a-Judge:** Using a powerful LLM to evaluate the quality of generated answers based on criteria like correctness, faithfulness, and relevance. This can automate parts of human evaluation.

*   **End-to-End Metrics (often human-evaluated):**
    *   **Faithfulness/Factuality:** Does the generated answer only contain information supported by the retrieved documents?
    *   **Relevance:** Does the generated answer directly address the user's question?
    *   **Coherence/Fluency:** Is the generated answer grammatically correct and easy to read?
    *   **Completeness:** Does the generated answer provide all necessary information to answer the question?

By utilizing these resources and metrics, you can systematically develop, evaluate, and improve your RAG system, ensuring it meets the desired performance and quality standards.




## Common Pitfalls to Avoid

Building a RAG system can be complex, and several common pitfalls can hinder development and performance. Being aware of these can help you navigate the process more smoothly.

1.  **Don't start too complex:** It's tempting to jump straight into advanced RAG features like query rewriting, reranking, or multi-hop retrieval. However, this can quickly lead to an unmanageable system. Start with a basic Naive RAG implementation, get it working, and establish a baseline. Then, incrementally add complexity and evaluate the impact of each new feature.

2.  **Test with small data first:** Before scaling to millions of documents, develop and test your RAG components and pipeline with a small, representative dataset (e.g., 100-1000 documents). This allows for faster iteration, easier debugging, and quicker identification of fundamental issues without waiting for long indexing or retrieval times.

3.  **Monitor memory usage:** Embedding models and vector stores, especially with large datasets, can be memory-intensive. Unchecked memory consumption can lead to crashes or slow performance. Regularly profile your application's memory usage during indexing and querying to identify and address potential memory leaks or inefficiencies. Consider techniques like embedding quantization or using disk-based vector stores for very large datasets.

4.  **Version your experiments:** RAG development often involves experimenting with different chunking strategies, embedding models, rerankers, and LLM prompts. Without proper versioning, it's easy to lose track of what worked and why. Use version control (e.g., Git) to manage your codebase, and consider logging experiment configurations and results (e.g., using MLflow, Weights & Biases) to ensure reproducibility and track progress.

5.  **Profile before optimizing:** Don't optimize prematurely. Identify actual performance bottlenecks through profiling before investing time in optimization efforts. A component you assume is slow might not be the real culprit. Use profiling tools to pinpoint where your system spends most of its time (e.g., embedding generation, vector search, LLM inference) and focus your optimization efforts there.

6.  **Ignoring the LLM's context window:** LLMs have a finite context window. If the retrieved documents, combined with the question and prompt, exceed this limit, the LLM will truncate the input, leading to loss of information and potentially poor answers. Implement strategies for context window management, such as intelligent chunking, context compression, or dynamic truncation, to ensure that only the most relevant information fits within the LLM's capacity.

7.  **Over-reliance on automatic metrics:** While automatic evaluation metrics (BLEU, ROUGE, MRR, NDCG) are useful for quick iteration and large-scale testing, they don't always perfectly correlate with human judgment. Supplement automatic metrics with qualitative human evaluation, especially for answer quality and faithfulness. LLM-as-a-judge can also be a valuable tool for scaling human-like evaluation.

8.  **Lack of robust error handling:** Production RAG systems interact with multiple external services (LLM APIs, vector databases, document storage). Implement comprehensive error handling, retry mechanisms, and fallback strategies to ensure the system remains resilient to transient failures or unexpected responses from these services.

By keeping these common pitfalls in mind, you can build a more robust, efficient, and effective RAG system.



