#!/usr/bin/env python3
"""
Advanced RAG Demo - Showcasing Query Optimization and Reranking Features
"""

import os
import tempfile
from src.rag.advanced_rag import AdvancedRAG
from src.retrieval.reranker import CrossEncoderReranker, SimpleReranker

def create_sample_documents():
    """Create sample documents for demonstration."""
    documents = {
        "france.txt": """
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

def demo_query_optimization():
    """Demonstrate different query optimization strategies."""
    print("🚀 Advanced RAG Demo - Query Optimization")
    print("=" * 60)
    
    # Create sample documents
    documents = create_sample_documents()
    temp_files = []
    
    try:
        # Create temporary files for the documents
        for filename, content in documents.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_files.append(f.name)
        
        print("📚 Initializing Advanced RAG system...")
        advanced_rag = AdvancedRAG(config_path='config.yaml')
        
        print("📖 Indexing documents...")
        advanced_rag.index_documents(temp_files)
        print(f"✅ Indexed {len(temp_files)} documents successfully!")
        
        # Test queries
        test_queries = [
            "What is the capital of France?",
            "When was the Eiffel Tower built?",
            "What is AI and how does it work?",
            "How does RAG work and what are its components?"
        ]
        
        optimization_strategies = [
            "none",
            "expansion", 
            "decomposition",
            "hyde"
        ]
        
        print("\n🔍 Testing Query Optimization Strategies")
        print("-" * 60)
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 40)
            
            for strategy in optimization_strategies:
                print(f"\nStrategy: {strategy.upper()}")
                
                # Test optimized retrieval
                retrieved_docs = advanced_rag.retrieve_optimized(
                    query, k=3, query_optimization_strategy=strategy
                )
                
                print(f"  Retrieved {len(retrieved_docs)} documents")
                for i, doc in enumerate(retrieved_docs[:2]):  # Show top 2
                    print(f"  {i+1}. {doc['content'][:80]}...")
                
                # Test optimized query
                answer = advanced_rag.query_optimized(
                    query, k=3, query_optimization_strategy=strategy
                )
                print(f"  Answer: {answer[:100]}...")
        
        print("\n🎉 Query Optimization Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        raise
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def demo_reranking():
    """Demonstrate reranking functionality."""
    print("\n\n🔄 Reranking Demo")
    print("=" * 60)
    
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
        rag = AdvancedRAG(config_path='config.yaml')
        
        print("📖 Indexing documents...")
        rag.index_documents(temp_files)
        
        # Initialize rerankers
        print("🔄 Initializing rerankers...")
        try:
            cross_encoder_reranker = CrossEncoderReranker()
            print("✅ Cross-Encoder reranker initialized")
        except Exception as e:
            print(f"⚠️ Cross-Encoder reranker failed: {e}")
            cross_encoder_reranker = None
            
        simple_reranker = SimpleReranker()
        print("✅ Simple reranker initialized")
        
        # Test queries
        test_queries = [
            "What is the capital of France?",
            "How does artificial intelligence work?",
            "What are the components of RAG?"
        ]
        
        print("\n🔍 Testing Reranking")
        print("-" * 60)
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 40)
            
            # Get initial retrieval results
            initial_docs = rag.retrieve(query, k=5)
            print(f"Initial retrieval: {len(initial_docs)} documents")
            
            # Show original order
            print("\nOriginal order:")
            for i, doc in enumerate(initial_docs[:3]):
                print(f"  {i+1}. {doc['content'][:60]}...")
            
            # Test simple reranking
            if simple_reranker:
                simple_reranked = simple_reranker.rerank(query, initial_docs)
                print("\nSimple reranking:")
                for i, doc in enumerate(simple_reranked[:3]):
                    score = doc.get('relevance_score', 0)
                    print(f"  {i+1}. (Score: {score:.3f}) {doc['content'][:60]}...")
            
            # Test cross-encoder reranking
            if cross_encoder_reranker:
                cross_reranked = cross_encoder_reranker.rerank(query, initial_docs)
                print("\nCross-Encoder reranking:")
                for i, doc in enumerate(cross_reranked[:3]):
                    score = doc.get('relevance_score', 0)
                    print(f"  {i+1}. (Score: {score:.3f}) {doc['content'][:60]}...")
        
        print("\n🎉 Reranking Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        raise
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def main():
    """Run the complete advanced demo."""
    print("🚀 Advanced RAG System Demo")
    print("=" * 80)
    print("This demo showcases:")
    print("✅ Query optimization strategies")
    print("✅ Reranking with cross-encoders")
    print("✅ Advanced retrieval techniques")
    print("=" * 80)
    
    # Run query optimization demo
    demo_query_optimization()
    
    # Run reranking demo
    demo_reranking()
    
    print("\n🎉 Advanced RAG Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Query expansion with synonyms")
    print("✅ Query decomposition for complex questions")
    print("✅ HyDE (Hypothetical Document Embeddings)")
    print("✅ Cross-encoder reranking")
    print("✅ Simple heuristic reranking")

if __name__ == "__main__":
    main() 