#!/usr/bin/env python3
"""
Demo script to showcase the RAG system capabilities.
"""

import os
import tempfile
from src.rag.naive_rag import NaiveRAG

def create_sample_documents():
    """Create sample documents for demonstration."""
    documents = {
        "paris.txt": """
        Paris is the capital and most populous city of France. 
        It is known for its iconic Eiffel Tower, which was completed in 1889. 
        The city is also famous for the Louvre Museum, which houses the Mona Lisa. 
        Paris is located in northern France and is a major center for art, fashion, and culture.
        
        The city has a rich history dating back to the 3rd century BC. 
        It became the capital of France in the 12th century and has remained so ever since. 
        Paris is divided into 20 arrondissements and is known for its beautiful architecture, 
        wide boulevards, and numerous parks and gardens.
        """,
        
        "ai.txt": """
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines that can perform tasks that typically require human intelligence. 
        These tasks include learning, reasoning, problem-solving, perception, and language understanding.
        
        Machine Learning is a subset of AI that focuses on algorithms and statistical models 
        that enable computers to improve their performance on a specific task through experience. 
        Deep Learning, a subset of machine learning, uses neural networks with multiple layers 
        to model and understand complex patterns in data.
        
        AI has applications in various fields including healthcare, finance, transportation, 
        entertainment, and education. Recent advances in AI have led to the development of 
        large language models like GPT, BERT, and others that can understand and generate human language.
        """,
        
        "rag.txt": """
        Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval 
        with text generation to create more accurate and informative responses. RAG systems work 
        by first retrieving relevant documents or passages from a knowledge base, then using 
        a language model to generate a response based on the retrieved information.
        
        The key components of a RAG system include:
        1. Document indexing and chunking
        2. Embedding generation and vector storage
        3. Query processing and retrieval
        4. Context-aware text generation
        
        RAG systems are particularly useful for question-answering tasks where the answer 
        requires information from external knowledge sources. They help reduce hallucination 
        in language models by grounding responses in retrieved facts and evidence.
        """
    }
    return documents

def demo_rag_system():
    """Demonstrate the RAG system with sample documents and queries."""
    print("🚀 RAG for LLMs Demo")
    print("=" * 50)
    
    # Create sample documents
    documents = create_sample_documents()
    temp_files = []
    
    try:
        # Create temporary files for the documents
        for filename, content in documents.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_files.append(f.name)
        
        print("📚 Initializing RAG system...")
        rag = NaiveRAG(config_path='config.yaml')
        
        print("📖 Indexing documents...")
        rag.index_documents(temp_files)
        print(f"✅ Indexed {len(temp_files)} documents successfully!")
        
        # Demo queries
        demo_queries = [
            "What is the capital of France?",
            "When was the Eiffel Tower completed?",
            "What is Artificial Intelligence?",
            "How does RAG work?",
            "What are the key components of a RAG system?"
        ]
        
        print("\n🔍 Testing RAG System with Sample Queries")
        print("-" * 50)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{i}. Question: {query}")
            print("-" * 30)
            
            # Get answer
            answer = rag.query(query)
            print(f"Answer: {answer}")
            
            # Get retrieved documents for context
            retrieved_docs = rag.retrieve(query, k=2)
            print(f"\nRetrieved {len(retrieved_docs)} relevant documents:")
            for j, doc in enumerate(retrieved_docs, 1):
                print(f"  {j}. Source: {doc['metadata'].get('source', 'Unknown')}")
                print(f"     Content: {doc['content'][:100]}...")
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Document indexing and chunking")
        print("✅ Semantic search and retrieval")
        print("✅ Context-aware answer generation")
        print("✅ Multi-document knowledge base")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        raise
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

if __name__ == "__main__":
    demo_rag_system() 