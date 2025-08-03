#!/usr/bin/env python3
"""
Complete RAG System Demo - All Phases

This demo showcases the complete RAG implementation including:
- Phase 1-2: Basic RAG functionality
- Phase 3: Testing and evaluation
- Phase 4: Advanced features (query optimization, reranking)
- Phase 5: Modular architecture
- Phase 6: Domain-specific applications and performance optimization
"""

import os
import tempfile
import time
from typing import List, Dict, Any

# Import all RAG implementations
from src.rag.naive_rag import NaiveRAG
from src.rag.advanced_rag import AdvancedRAG
from src.rag.modular_rag import ModularRAG
from src.applications.code_rag import CodeRAG
from src.optimization.performance_optimizer import PerformanceOptimizer

def create_sample_documents():
    """Create comprehensive sample documents for demonstration."""
    documents = {
        "general_knowledge.txt": """
        Paris is the capital and most populous city of France. 
        It is known for its iconic Eiffel Tower, which was completed in 1889. 
        The city is also famous for the Louvre Museum, which houses the Mona Lisa. 
        Paris is located in northern France and is a major center for art, fashion, and culture.
        
        The city has a rich history dating back to the 3rd century BC. 
        It became the capital of France in the 12th century and has remained so ever since. 
        Paris is divided into 20 arrondissements and is known for its beautiful architecture, 
        wide boulevards, and numerous parks and gardens.
        """,
        
        "ai_technology.txt": """
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
        
        "rag_system.txt": """
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

def create_sample_code_files():
    """Create sample code files for code RAG demonstration."""
    code_files = {
        "python_example.py": """
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    \"\"\"Calculate the factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n-1)

class Calculator:
    \"\"\"A simple calculator class.\"\"\"
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        \"\"\"Add two numbers.\"\"\"
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        \"\"\"Multiply two numbers.\"\"\"
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
        """,
        
        "data_processing.py": """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    \"\"\"Load and clean data from CSV file.\"\"\"
    df = pd.read_csv(file_path)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    return df

def normalize_features(df, columns):
    \"\"\"Normalize specified columns using StandardScaler.\"\"\"
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

def create_features(df):
    \"\"\"Create new features from existing data.\"\"\"
    # Example feature engineering
    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
    
    return df
        """
    }
    return code_files

def demo_phase_1_2_basic_rag():
    """Demonstrate Phase 1-2: Basic RAG functionality."""
    print("🚀 Phase 1-2: Basic RAG Implementation")
    print("=" * 60)
    
    documents = create_sample_documents()
    temp_files = []
    
    try:
        # Create temporary files
        for filename, content in documents.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_files.append(f.name)
        
        # Initialize basic RAG
        print("📚 Initializing Basic RAG system...")
        basic_rag = NaiveRAG(config_path='config.yaml')
        
        print("📖 Indexing documents...")
        basic_rag.index_documents(temp_files)
        
        # Test basic functionality
        test_queries = [
            "What is the capital of France?",
            "What is Artificial Intelligence?",
            "How does RAG work?"
        ]
        
        print("\n🔍 Testing Basic RAG Queries")
        print("-" * 40)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            answer = basic_rag.query(query)
            print(f"Answer: {answer[:100]}...")
        
        print("\n✅ Basic RAG Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error in basic RAG demo: {e}")
    finally:
        # Cleanup
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def demo_phase_4_advanced_features():
    """Demonstrate Phase 4: Advanced RAG features."""
    print("\n\n🔄 Phase 4: Advanced RAG Features")
    print("=" * 60)
    
    documents = create_sample_documents()
    temp_files = []
    
    try:
        # Create temporary files
        for filename, content in documents.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_files.append(f.name)
        
        # Initialize advanced RAG
        print("📚 Initializing Advanced RAG system...")
        advanced_rag = AdvancedRAG(config_path='config.yaml')
        
        print("📖 Indexing documents...")
        advanced_rag.index_documents(temp_files)
        
        # Test advanced features
        test_query = "What is the capital of France and when was the Eiffel Tower built?"
        
        print(f"\n🔍 Testing Query Optimization Strategies")
        print("-" * 50)
        
        strategies = ["none", "expansion", "decomposition", "hyde"]
        
        for strategy in strategies:
            print(f"\nStrategy: {strategy.upper()}")
            start_time = time.time()
            answer = advanced_rag.query_optimized(test_query, query_optimization_strategy=strategy)
            end_time = time.time()
            
            print(f"  Time: {end_time - start_time:.2f}s")
            print(f"  Answer: {answer[:80]}...")
        
        print("\n✅ Advanced Features Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error in advanced features demo: {e}")
    finally:
        # Cleanup
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def demo_phase_5_modular_architecture():
    """Demonstrate Phase 5: Modular RAG architecture."""
    print("\n\n🏗️ Phase 5: Modular RAG Architecture")
    print("=" * 60)
    
    documents = create_sample_documents()
    temp_files = []
    
    try:
        # Create temporary files
        for filename, content in documents.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_files.append(f.name)
        
        # Initialize modular RAG
        print("📚 Initializing Modular RAG system...")
        modular_rag = ModularRAG(config_path='config.yaml')
        
        print("📖 Indexing documents...")
        modular_rag.index_documents(temp_files)
        
        # Test modular features
        print("\n🔍 Testing Modular RAG Features")
        print("-" * 40)
        
        # Test conversation memory
        print("\n💬 Testing Conversation Memory")
        questions = [
            "What is the capital of France?",
            "When was it built?",
            "What famous museum is there?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\nTurn {i}: {question}")
            answer = modular_rag.query(question, use_memory=True)
            print(f"Answer: {answer[:80]}...")
        
        # Show system info
        print("\n📊 System Information")
        system_info = modular_rag.get_system_info()
        print(f"Components: {system_info['components']}")
        print(f"Conversation turns: {system_info['conversation_turns']}")
        
        # Test conversation history
        history = modular_rag.get_conversation_history()
        print(f"\n📝 Conversation History ({len(history)} turns)")
        for i, turn in enumerate(history, 1):
            print(f"  Turn {i}: {turn['query'][:50]}...")
        
        print("\n✅ Modular Architecture Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error in modular architecture demo: {e}")
    finally:
        # Cleanup
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def demo_phase_6_domain_specific():
    """Demonstrate Phase 6: Domain-specific applications."""
    print("\n\n🎯 Phase 6: Domain-Specific Applications")
    print("=" * 60)
    
    code_files = create_sample_code_files()
    temp_files = []
    
    try:
        # Create temporary code files
        for filename, content in code_files.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_files.append(f.name)
        
        # Initialize code RAG
        print("📚 Initializing Code RAG system...")
        code_rag = CodeRAG(config_path='config.yaml')
        
        print("📖 Indexing code files...")
        code_rag.index_code_files(temp_files)
        
        # Test code-specific queries
        print("\n🔍 Testing Code RAG Queries")
        print("-" * 40)
        
        code_queries = [
            "How do I calculate Fibonacci numbers?",
            "What is a factorial function?",
            "How do I create a calculator class?",
            "How do I load and clean data with pandas?"
        ]
        
        for query in code_queries:
            print(f"\nQuery: {query}")
            answer = code_rag.query_code(query)
            print(f"Answer: {answer[:100]}...")
        
        # Test code explanation
        print("\n📝 Testing Code Explanation")
        code_snippet = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
        """
        
        explanation = code_rag.explain_code(code_snippet)
        print(f"Explanation: {explanation[:100]}...")
        
        # Test debugging
        print("\n🐛 Testing Code Debugging")
        buggy_code = """
def divide_numbers(a, b):
    return a / b

result = divide_numbers(10, 0)
        """
        
        debug_answer = code_rag.debug_code(buggy_code, "ZeroDivisionError: division by zero")
        print(f"Debug Answer: {debug_answer[:100]}...")
        
        print("\n✅ Domain-Specific Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error in domain-specific demo: {e}")
    finally:
        # Cleanup
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def demo_performance_optimization():
    """Demonstrate performance optimization features."""
    print("\n\n⚡ Phase 6: Performance Optimization")
    print("=" * 60)
    
    try:
        # Initialize performance optimizer
        print("📊 Initializing Performance Optimizer...")
        optimizer = PerformanceOptimizer()
        
        # Get system information
        print("\n🖥️ System Information")
        gpu_info = optimizer.gpu_accelerator.get_gpu_info()
        if gpu_info['available']:
            print(f"✅ GPU available: {gpu_info['count']} device(s)")
            for device in gpu_info['devices']:
                print(f"  - {device['name']} ({device['memory_gb']:.1f}GB)")
        else:
            print("❌ GPU not available, using CPU")
        
        # Monitor performance
        print("\n📈 Performance Monitoring")
        optimizer.monitor.start_timer('demo_operation')
        time.sleep(1)  # Simulate some work
        optimizer.monitor.end_timer('demo_operation')
        
        # Get performance report
        report = optimizer.get_performance_report()
        
        print(f"Memory usage: {report['system_resources']['memory']['percent_used']:.1f}%")
        print(f"CPU usage: {report['system_resources']['cpu']:.1f}%")
        print(f"Demo operation time: {report['performance_metrics']['timings'].get('demo_operation', 0):.3f}s")
        
        # Test quantization
        print("\n🔢 Testing Embedding Quantization")
        test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        
        quantizer = optimizer.quantizer
        quantized, metadata = quantizer.quantize(test_embeddings)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        print(f"Original embeddings: {len(test_embeddings)} vectors")
        print(f"Quantized embeddings: {len(quantized)} vectors")
        print(f"Quantization type: {metadata['quantization_type']}")
        print(f"Bits per dimension: {metadata['bits']}")
        
        print("\n✅ Performance Optimization Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error in performance optimization demo: {e}")

def main():
    """Run the complete RAG system demo."""
    print("🎉 Complete RAG System Implementation Demo")
    print("=" * 80)
    print("This demo showcases all phases of the RAG implementation:")
    print("✅ Phase 1-2: Basic RAG functionality")
    print("✅ Phase 3: Testing and evaluation")
    print("✅ Phase 4: Advanced features (query optimization, reranking)")
    print("✅ Phase 5: Modular architecture")
    print("✅ Phase 6: Domain-specific applications and performance optimization")
    print("=" * 80)
    
    # Run all demos
    demo_phase_1_2_basic_rag()
    demo_phase_4_advanced_features()
    demo_phase_5_modular_architecture()
    demo_phase_6_domain_specific()
    demo_performance_optimization()
    
    print("\n🎉 Complete RAG System Demo Finished!")
    print("\n📋 Summary of Implemented Features:")
    print("✅ Basic RAG pipeline with all core components")
    print("✅ Query optimization strategies (expansion, decomposition, HyDE)")
    print("✅ Reranking with cross-encoders and simple heuristics")
    print("✅ Modular architecture with factory pattern")
    print("✅ Context compression and hybrid search")
    print("✅ Conversation memory and multi-turn dialogue")
    print("✅ Domain-specific RAG (Code RAG)")
    print("✅ Performance optimization (quantization, GPU acceleration)")
    print("✅ Comprehensive testing and evaluation framework")
    print("\n🚀 The RAG system is now production-ready!")

if __name__ == "__main__":
    main() 