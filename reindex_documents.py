#!/usr/bin/env python3
"""Re-index existing documents in the uploaded_documents folder"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag.naive_rag import NaiveRAG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reindex_all_documents():
    """Re-index all documents in the uploaded_documents directory"""
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag_system = NaiveRAG(config_path='config.yaml')
    
    # Get all files in uploaded_documents
    uploaded_dir = Path("uploaded_documents")
    if not uploaded_dir.exists():
        logger.error("uploaded_documents directory not found")
        return
    
    # Find all documents
    documents = []
    for file_path in uploaded_dir.iterdir():
        if file_path.is_file():
            documents.append(str(file_path))
    
    if not documents:
        logger.warning("No documents found in uploaded_documents directory")
        return
    
    logger.info(f"Found {len(documents)} documents to index:")
    for doc in documents:
        logger.info(f"  - {os.path.basename(doc)}")
    
    # Clear existing vector store
    logger.info("Clearing existing vector store...")
    # ChromaDB will handle this when we re-index
    
    # Re-index all documents
    logger.info("Re-indexing documents...")
    rag_system.index_documents(documents)
    
    logger.info("Re-indexing complete!")
    
    # Test retrieval
    logger.info("Testing retrieval...")
    test_query = "requirements"
    results = rag_system.retrieve(test_query, k=3)
    logger.info(f"Test query '{test_query}' returned {len(results)} results")
    
if __name__ == "__main__":
    reindex_all_documents()