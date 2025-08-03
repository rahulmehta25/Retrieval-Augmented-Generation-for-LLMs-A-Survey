#!/usr/bin/env python3
"""
Simple test script to verify the RAG system implementation.
"""

import os
import tempfile
from src.rag.naive_rag import NaiveRAG

def create_test_document():
    """Create a simple test document."""
    content = """
    Paris is the capital and most populous city of France. 
    It is known for its iconic Eiffel Tower, which was completed in 1889. 
    The city is also famous for the Louvre Museum, which houses the Mona Lisa. 
    Paris is located in northern France and is a major center for art, fashion, and culture.
    
    The city has a rich history dating back to the 3rd century BC. 
    It became the capital of France in the 12th century and has remained so ever since. 
    Paris is divided into 20 arrondissements and is known for its beautiful architecture, 
    wide boulevards, and numerous parks and gardens.
    """
    return content

def test_rag_system():
    """Test the complete RAG pipeline."""
    print("Testing RAG system...")
    
    # Create a temporary test document
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(create_test_document())
        test_file = f.name
    
    try:
        # Initialize RAG system
        print("Initializing RAG system...")
        rag = NaiveRAG(config_path='config.yaml')
        
        # Index the test document
        print("Indexing test document...")
        rag.index_documents([test_file])
        
        # Test query
        print("Testing query...")
        question = "What is the capital of France?"
        answer = rag.query(question)
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
        
        # Test another query
        question2 = "When was the Eiffel Tower completed?"
        answer2 = rag.query(question2)
        
        print(f"\nQuestion: {question2}")
        print(f"Answer: {answer2}")
        
        print("\nRAG system test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == "__main__":
    test_rag_system() 