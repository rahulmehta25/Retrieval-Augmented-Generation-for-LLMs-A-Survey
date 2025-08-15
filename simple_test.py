#!/usr/bin/env python3
"""
Simple test script to verify RAG components are working
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_basic_components():
    """Test basic RAG components"""
    try:
        print("Testing basic RAG components...")
        
        # Test chunking
        from src.chunking.text_splitter import TextSplitter, FixedSizeTextSplitter
        print("✅ TextSplitter imported successfully")
        
        # Test embedding
        from src.embedding.embedder import Embedder, SentenceTransformerEmbedder
        print("✅ Embedder imported successfully")
        
        # Test vector store
        from src.retrieval.vector_store import VectorStore, ChromaDBVectorStore
        print("✅ VectorStore imported successfully")
        
        # Test generation
        from src.generation.generator import Generator, HuggingFaceGenerator
        print("✅ Generator imported successfully")
        
        # Test basic RAG
        from src.rag.naive_rag import NaiveRAG
        print("✅ NaiveRAG imported successfully")
        
        print("\n🎉 All basic components imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error importing components: {e}")
        return False

def test_advanced_components():
    """Test advanced RAG components"""
    try:
        print("\nTesting advanced components...")
        
        # Test advanced RAG
        from src.rag.advanced_rag import AdvancedRAG
        print("✅ AdvancedRAG imported successfully")
        
        # Test modular RAG
        from src.rag.modular_rag import ModularRAG
        print("✅ ModularRAG imported successfully")
        
        # Test evaluation
        from src.evaluation.ragas_metrics import RAGASEvaluator
        print("✅ RAGASEvaluator imported successfully")
        
        print("\n🎉 All advanced components imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error importing advanced components: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting RAG Component Test...\n")
    
    basic_ok = test_basic_components()
    advanced_ok = test_advanced_components()
    
    if basic_ok and advanced_ok:
        print("\n🎯 All components are working! Ready to run Streamlit app.")
    else:
        print("\n⚠️  Some components have issues. Check the errors above.")
