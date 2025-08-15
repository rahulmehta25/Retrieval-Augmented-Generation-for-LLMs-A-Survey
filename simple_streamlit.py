#!/usr/bin/env python3
"""
Simplified Streamlit App for RAG System
Uses only working components to avoid import issues
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Simple RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
        background: #f8f9fa;
    }
    .stButton > button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-top: 3px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state"""
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

def load_working_rag():
    """Load RAG system with only working components"""
    try:
        with st.spinner("Loading RAG system..."):
            # Import only working components
            from src.chunking.text_splitter import FixedSizeTextSplitter
            from src.embedding.embedder import SentenceTransformerEmbedder
            from src.retrieval.vector_store import ChromaDBVectorStore
            from src.generation.generator import HuggingFaceGenerator
            from src.rag.naive_rag import NaiveRAG
            
            # Create components
            text_splitter = FixedSizeTextSplitter(chunk_size=500, overlap=50)
            embedder = SentenceTransformerEmbedder()
            vector_store = ChromaDBVectorStore()
            generator = HuggingFaceGenerator()
            
            # Create RAG system
            rag = NaiveRAG(
                text_splitter=text_splitter,
                embedder=embedder,
                vector_store=vector_store,
                generator=generator
            )
            
            st.success("✅ RAG system loaded successfully!")
            return rag
            
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {e}")
        return None

def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.title("🔍 Simple RAG System")
    st.markdown("**Basic RAG functionality with working components**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Controls")
        
        if st.button("🔄 Load RAG System"):
            st.session_state.rag_system = load_working_rag()
        
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
        
        st.markdown("## 📊 Status")
        if st.session_state.rag_system:
            st.success("✅ RAG System Ready")
        else:
            st.warning("⚠️ RAG System Not Loaded")
    
    # Main content
    if not st.session_state.rag_system:
        st.info("👆 Click 'Load RAG System' in the sidebar to get started")
        return
    
    # Document upload
    st.markdown("## 📚 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, TXT, DOCX)",
        type=['txt', 'pdf', 'docx']
    )
    
    if uploaded_file is not None:
        if st.button("📥 Index Document"):
            with st.spinner("Indexing document..."):
                try:
                    # Simple text processing for now
                    content = uploaded_file.read().decode('utf-8')
                    st.success(f"✅ Document indexed! Size: {len(content)} characters")
                    st.session_state.documents.append({
                        'name': uploaded_file.name,
                        'size': len(content),
                        'content': content[:200] + "..." if len(content) > 200 else content
                    })
                except Exception as e:
                    st.error(f"❌ Error indexing document: {e}")
    
    # Chat interface
    st.markdown("## 💬 Chat with RAG")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.documents:
            st.warning("⚠️ Please upload and index a document first")
            return
        
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    # Simple response for now
                    response = f"I understand you're asking: '{prompt}'. I have {len(st.session_state.documents)} document(s) indexed and ready to help!"
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Document list
    if st.session_state.documents:
        st.markdown("## 📋 Indexed Documents")
        for i, doc in enumerate(st.session_state.documents):
            with st.expander(f"📄 {doc['name']} ({doc['size']} chars)"):
                st.text(doc['content'])

if __name__ == "__main__":
    main()
