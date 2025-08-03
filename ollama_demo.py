#!/usr/bin/env python3
"""
Ollama RAG Demo

This script demonstrates how to use the RAG system with Ollama for local LLM inference.
Ollama provides a simple way to run large language models locally via HTTP API.

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Start Ollama server: ollama serve
3. Download a model: ollama pull gemma:2b

Usage:
    python ollama_demo.py
"""

import yaml
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag.naive_rag import NaiveRAG


def create_ollama_config():
    """Create configuration for Ollama-based RAG system."""
    return {
        "text_splitter": {
            "type": "fixed_size",
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "embedder": {
            "type": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2",
            "cache_dir": "./embedding_cache"
        },
        "vector_store": {
            "type": "chromadb",
            "path": "./ollama_chroma_db",
            "collection_name": "ollama_rag_demo"
        },
        "generator": {
            "type": "ollama",
            "model_name": "gemma:7b",  # Using your installed Gemma 7B model
            "host": "localhost",
            "port": 11434
        }
    }


def load_sample_documents():
    """Load sample documents for demonstration."""
    return [
        """
        Artificial Intelligence (AI) is a broad field of computer science focused on building 
        smart machines capable of performing tasks that typically require human intelligence. 
        AI systems can learn, reason, perceive, and make decisions. The field encompasses 
        machine learning, deep learning, natural language processing, computer vision, 
        and robotics.
        """,
        """
        Machine Learning (ML) is a subset of artificial intelligence that enables computers 
        to learn and improve from experience without being explicitly programmed. ML algorithms 
        build mathematical models based on training data to make predictions or decisions. 
        Common types include supervised learning, unsupervised learning, and reinforcement learning.
        """,
        """
        Deep Learning is a specialized subset of machine learning that uses artificial neural 
        networks with multiple layers (hence "deep") to model and understand complex patterns 
        in data. It's particularly effective for tasks like image recognition, natural language 
        processing, and speech recognition. Popular frameworks include TensorFlow and PyTorch.
        """,
        """
        Natural Language Processing (NLP) is a branch of AI that helps computers understand, 
        interpret, and manipulate human language. NLP combines computational linguistics with 
        statistical machine learning and deep learning models. Applications include chatbots, 
        language translation, sentiment analysis, and text summarization.
        """,
        """
        Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval 
        with text generation. RAG systems first retrieve relevant documents from a knowledge base, 
        then use this information to generate more accurate and contextual responses. This approach 
        helps language models provide more factual and up-to-date information.
        """,
        """
        Ollama is a tool that allows you to run large language models locally on your machine. 
        It provides a simple API for interacting with various open-source models like Llama, 
        Mistral, and Gemma. Ollama handles model management, serving, and provides both 
        command-line and HTTP API interfaces for easy integration.
        """
    ]


def main():
    """Main demo function."""
    print("🦙 Ollama RAG Demo")
    print("=" * 50)
    
    try:
        # Create configuration
        print("⚙️  Creating Ollama-based RAG configuration...")
        config = create_ollama_config()
        
        # Save config to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        # Initialize RAG system
        print("🚀 Initializing RAG system with Ollama...")
        rag = NaiveRAG(config_path)
        
        # Load and index documents
        print("📚 Loading and indexing sample documents...")
        documents = load_sample_documents()
        
        # Create temporary files for documents
        import os
        doc_paths = []
        for i, doc in enumerate(documents):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(doc.strip())
                doc_paths.append(f.name)
        
        rag.index_documents(doc_paths)
        
        # Clean up temporary document files
        for path in doc_paths:
            os.unlink(path)
        print(f"✅ Indexed {len(documents)} documents")
        
        # Interactive query loop
        print("\n💬 RAG System Ready! You can now ask questions.")
        print("💡 Try questions like:")
        print("   - What is machine learning?")
        print("   - How does RAG work?")
        print("   - What is Ollama?")
        print("   - Explain deep learning")
        print("\n💡 Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                question = input("\n❓ Your question: ").strip()
                
                # Check for exit
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                # Query the RAG system
                print("🤖 Thinking...")
                answer = rag.query(question)
                
                print(f"\n📝 Answer: {answer}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing question: {str(e)}")
                continue
    
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Ollama is installed and running:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        print("   ollama serve")
        print("\n2. Download the Gemma model:")
        print("   ollama pull gemma:2b")
        print("\n3. Alternative models you can try:")
        print("   ollama pull llama2:7b")
        print("   ollama pull mistral:7b")
        print("   ollama pull codellama:7b")
        print("\n4. Update the config in this script to use a different model")
        sys.exit(1)


def show_available_models():
    """Show available Ollama models and configuration options."""
    print("\n📋 Popular Ollama Models:")
    print("┌─────────────────┬─────────────┬─────────────────────────────────┐")
    print("│ Model           │ Size        │ Description                     │")
    print("├─────────────────┼─────────────┼─────────────────────────────────┤")
    print("│ gemma:2b        │ ~1.7GB      │ Lightweight, fast              │")
    print("│ gemma:7b        │ ~5.0GB      │ Better quality                  │")
    print("│ llama2:7b       │ ~3.8GB      │ Meta's Llama 2                  │")
    print("│ llama2:13b      │ ~7.3GB      │ Larger Llama 2                  │")
    print("│ mistral:7b      │ ~4.1GB      │ High quality, efficient         │")
    print("│ codellama:7b    │ ~3.8GB      │ Code-focused model              │")
    print("│ neural-chat:7b  │ ~4.1GB      │ Optimized for conversations     │")
    print("└─────────────────┴─────────────┴─────────────────────────────────┘")
    
    print("\n⚙️  To use a different model:")
    print("1. Download: ollama pull <model_name>")
    print("2. Update the model_name in create_ollama_config()")
    print("3. Restart the demo")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--models":
        show_available_models()
    else:
        main()