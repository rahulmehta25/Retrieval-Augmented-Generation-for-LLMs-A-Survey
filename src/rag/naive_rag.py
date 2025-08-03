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
            from src.chunking.text_splitter import SentenceTextSplitter
            return SentenceTextSplitter()
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
        elif embedder_type == 'openai':
            from src.embedding.embedder import OpenAIEmbedder
            return OpenAIEmbedder(model_name=embedder_config.get('model_name', 'text-embedding-ada-002'))
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
        elif generator_type == 'openai':
            from src.generation.generator import OpenAIGenerator
            return OpenAIGenerator(model_name=generator_config.get('model_name', 'gpt-3.5-turbo'))
        elif generator_type == 'ollama':
            from src.generation.generator import OllamaGenerator
            return OllamaGenerator(
                model_name=generator_config.get('model_name', 'gemma:2b'),
                host=generator_config.get('host', 'localhost'),
                port=generator_config.get('port', 11434)
            )
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
        
        # Check if the retrieved contexts are actually relevant
        # If the best match has a very high distance, it might not be relevant
        if retrieved_contexts and retrieved_contexts[0].get('distance', 0) > 0.8:
            logging.warning("Retrieved contexts have low relevance scores.")
            return "I am sorry, but I could not find enough relevant information to answer your question."
            
        answer = self.generate_answer(question, retrieved_contexts)
        return answer 