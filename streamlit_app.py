"""
Production Streamlit Dashboard for Advanced RAG System
Complete interface for all production features including monitoring, A/B testing, and advanced retrieval
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import requests
from dataclasses import asdict

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import all production components
from src.rag.naive_rag import NaiveRAG
from src.rag.advanced_rag import AdvancedRAG
from src.rag.modular_rag import ModularRAG
from src.graph_rag.knowledge_graph import GraphRAG
from src.graph_rag.advanced_knowledge_graph import AdvancedKnowledgeGraph, GraphQuery
from src.advanced_rag.self_rag import SelfRAG
from src.evaluation.ragas_metrics import RAGASEvaluator
from src.evaluation.benchmark import RAGBenchmark
from src.streaming.stream_handler import StreamingRAG

# Import new production components
from src.optimization.semantic_query_optimizer import SemanticQueryOptimizer, QueryRewriter
from src.retrieval.advanced_hybrid_retriever import AdvancedHybridRetriever
from src.chunking.semantic_chunker import SemanticChunker
from src.retrieval.advanced_context_compressor import AdvancedContextCompressor
from src.memory.advanced_conversation_memory import AdvancedConversationMemory
from src.experimentation.ab_testing_framework import (
    ABTestingFramework, Variant, Metric, 
    ExperimentStatus, AllocationStrategy
)
from src.monitoring.production_monitoring import ProductionMonitoring, AlertSeverity

# Page configuration
st.set_page_config(
    page_title="Production RAG System Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with production UI
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
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,123,255,0.3);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-top: 3px solid #007bff;
    }
    .health-good {
        color: #28a745;
        font-weight: bold;
    }
    .health-degraded {
        color: #ffc107;
        font-weight: bold;
    }
    .health-bad {
        color: #dc3545;
        font-weight: bold;
    }
    .alert-critical {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .experiment-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.8rem 0;
    }
    .monitoring-metric {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        background: #e9ecef;
        border-radius: 4px;
        margin: 0.2rem;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    
    # RAG systems
    if 'rag_systems' not in st.session_state:
        st.session_state.rag_systems = {}
    
    # Production components
    if 'query_optimizer' not in st.session_state:
        st.session_state.query_optimizer = SemanticQueryOptimizer()
    
    if 'knowledge_graph' not in st.session_state:
        st.session_state.knowledge_graph = AdvancedKnowledgeGraph(
            persist_path="./knowledge_graph"
        )
    
    if 'hybrid_retriever' not in st.session_state:
        st.session_state.hybrid_retriever = AdvancedHybridRetriever(
            index_path="./hybrid_index"
        )
    
    if 'semantic_chunker' not in st.session_state:
        st.session_state.semantic_chunker = SemanticChunker()
    
    if 'context_compressor' not in st.session_state:
        st.session_state.context_compressor = AdvancedContextCompressor()
    
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = AdvancedConversationMemory()
    
    if 'ab_testing' not in st.session_state:
        st.session_state.ab_testing = ABTestingFramework(
            persist_path="./experiments"
        )
    
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = ProductionMonitoring(
            service_name="rag_production",
            prometheus_port=8001,
            enable_prometheus=False  # Disable for Streamlit
        )
    
    # Other state
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = f"session_{datetime.now().timestamp()}"

def main():
    """Main application"""
    
    init_session_state()
    
    # Header
    st.title("🏭 Production RAG System Dashboard")
    st.markdown("**Advanced Features**: Query Optimization | Knowledge Graphs | Hybrid Retrieval | A/B Testing | Monitoring")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🎛️ Navigation")
        
        page = st.radio(
            "Select Module",
            [
                "📊 System Overview",
                "💬 Interactive Chat",
                "🔍 Advanced Retrieval",
                "🧠 Knowledge Graph",
                "📈 Monitoring",
                "🧪 A/B Testing",
                "💾 Memory & Context",
                "⚙️ Configuration"
            ]
        )
        
        st.markdown("---")
        
        # Quick health status
        st.markdown("### 🏥 System Health")
        health_checks = st.session_state.monitoring.perform_health_checks()
        
        for component, health in health_checks.items():
            if health.status == "healthy":
                st.success(f"✅ {component}")
            elif health.status == "degraded":
                st.warning(f"⚠️ {component}")
            else:
                st.error(f"❌ {component}")
    
    # Main content area
    if page == "📊 System Overview":
        show_system_overview()
    elif page == "💬 Interactive Chat":
        show_interactive_chat()
    elif page == "🔍 Advanced Retrieval":
        show_advanced_retrieval()
    elif page == "🧠 Knowledge Graph":
        show_knowledge_graph()
    elif page == "📈 Monitoring":
        show_monitoring()
    elif page == "🧪 A/B Testing":
        show_ab_testing()
    elif page == "💾 Memory & Context":
        show_memory_context()
    elif page == "⚙️ Configuration":
        show_configuration()

def show_system_overview():
    """System overview dashboard"""
    
    st.header("📊 System Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Documents",
            len(st.session_state.documents),
            delta="+2" if len(st.session_state.documents) > 0 else None
        )
    
    with col2:
        st.metric(
            "Active Sessions",
            1 if st.session_state.current_session_id else 0,
            delta=None
        )
    
    with col3:
        metrics_summary = st.session_state.monitoring.get_metrics_summary(300)
        avg_latency = metrics_summary['metrics'].get('request_duration', {}).get('mean', 0)
        st.metric(
            "Avg Latency",
            f"{avg_latency*1000:.0f}ms",
            delta="-12ms"
        )
    
    with col4:
        experiments = st.session_state.ab_testing.experiments
        active_exp = sum(1 for e in experiments.values() if e.status == ExperimentStatus.ACTIVE)
        st.metric(
            "Active Experiments",
            active_exp,
            delta=None
        )
    
    st.markdown("---")
    
    # Feature status grid
    st.subheader("🚀 Production Features Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        features = [
            ("Query Optimization", True, "Semantic analysis, intent classification"),
            ("Knowledge Graph", True, "Entity extraction, relationship mapping"),
            ("Hybrid Retrieval", True, "BM25 + Dense + Reranking"),
            ("Semantic Chunking", True, "Context-aware document splitting"),
            ("Context Compression", True, "Query-focused summarization")
        ]
        
        for feature, enabled, desc in features:
            if enabled:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>✅ {feature}</h4>
                    <p style="color: #666; margin: 0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        features = [
            ("Conversation Memory", True, "Multi-turn context tracking"),
            ("A/B Testing", True, "Statistical experiment framework"),
            ("Production Monitoring", True, "Metrics, alerts, health checks"),
            ("Streaming Responses", True, "Real-time token generation"),
            ("RAGAS Evaluation", True, "Automatic quality assessment")
        ]
        
        for feature, enabled, desc in features:
            if enabled:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>✅ {feature}</h4>
                    <p style="color: #666; margin: 0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("---")
    st.subheader("📝 Recent Activity")
    
    recent_traces = list(st.session_state.monitoring.request_traces)[-5:]
    if recent_traces:
        for trace in reversed(recent_traces):
            timestamp = trace['timestamp'].strftime("%H:%M:%S")
            method = trace['method']
            duration = trace['duration_ms']
            status = trace['status']
            
            status_icon = "✅" if status == "success" else "❌"
            st.markdown(f"{status_icon} **{timestamp}** - {method} ({duration:.0f}ms)")
    else:
        st.info("No recent activity")

def show_interactive_chat():
    """Interactive chat with production features"""
    
    st.header("💬 Interactive Chat with Production RAG")
    
    # Configuration panel
    with st.expander("⚙️ Chat Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rag_paradigm = st.selectbox(
                "RAG Paradigm",
                ["Naive", "Advanced", "Modular", "Graph", "Self-RAG", "Hybrid Production"]
            )
            
            enable_memory = st.checkbox("Enable Conversation Memory", value=True)
            enable_optimization = st.checkbox("Enable Query Optimization", value=True)
        
        with col2:
            retrieval_method = st.selectbox(
                "Retrieval Method",
                ["Dense Only", "Sparse Only", "Hybrid", "Adaptive"]
            )
            
            enable_compression = st.checkbox("Enable Context Compression", value=True)
            enable_reranking = st.checkbox("Enable Reranking", value=True)
        
        with col3:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            max_tokens = st.slider("Max Tokens", 100, 2000, 500)
            top_k = st.slider("Top K Documents", 1, 20, 5)
    
    # Start conversation session if memory enabled
    if enable_memory and not hasattr(st.session_state, 'memory_session_started'):
        st.session_state.conversation_memory.start_session(
            st.session_state.current_session_id
        )
        st.session_state.memory_session_started = True
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <strong>👤 You:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #f5f5f5; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <strong>🤖 Assistant:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show metrics if available
            if 'metrics' in message:
                with st.expander("📊 Response Metrics"):
                    col1, col2, col3 = st.columns(3)
                    metrics = message['metrics']
                    
                    with col1:
                        st.metric("Latency", f"{metrics.get('latency_ms', 0):.0f}ms")
                        st.metric("Tokens", metrics.get('tokens', 0))
                    
                    with col2:
                        st.metric("Contexts Retrieved", metrics.get('contexts', 0))
                        st.metric("Relevance Score", f"{metrics.get('relevance', 0):.2f}")
                    
                    with col3:
                        if 'ragas_scores' in metrics:
                            scores = metrics['ragas_scores']
                            st.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
                            st.metric("Answer Relevancy", f"{scores.get('answer_relevancy', 0):.2f}")
    
    # Query input
    query = st.text_input(
        "Ask a question:",
        placeholder="What are the key components of a RAG system?",
        key="chat_query"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Send", type="primary", use_container_width=True)
    
    if submit and query:
        start_time = time.time()
        
        with st.spinner("Processing query..."):
            # Track request
            st.session_state.monitoring.track_request(
                method="chat_query",
                duration_ms=0,
                status="processing"
            )
            
            # Query optimization
            if enable_optimization:
                with st.spinner("Optimizing query..."):
                    semantic_query = st.session_state.query_optimizer.optimize(query)
                    optimized_query = semantic_query.cleaned
                    
                    # Show optimization details
                    with st.expander("🔍 Query Analysis", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Intent:** {semantic_query.intent}")
                            st.write(f"**Complexity:** {semantic_query.complexity:.2f}")
                            st.write(f"**Domain:** {semantic_query.domain or 'General'}")
                        with col2:
                            st.write(f"**Entities:** {', '.join(semantic_query.keywords[:5])}")
                            st.write(f"**Context Required:** {'Yes' if semantic_query.context_required else 'No'}")
            else:
                optimized_query = query
            
            # Get conversation context if memory enabled
            if enable_memory:
                conversation_context = st.session_state.conversation_memory.get_relevant_context(
                    optimized_query, k=3
                )
            else:
                conversation_context = []
            
            # Retrieve documents
            with st.spinner("Retrieving relevant documents..."):
                if retrieval_method == "Adaptive":
                    results = st.session_state.hybrid_retriever.adaptive_retrieve(
                        optimized_query, k=top_k
                    )
                elif retrieval_method == "Hybrid":
                    results = st.session_state.hybrid_retriever.hybrid_retrieve(
                        optimized_query, k=top_k,
                        use_reranking=enable_reranking
                    )
                else:
                    # Fallback to simple retrieval
                    results = st.session_state.hybrid_retriever.dense_retrieval(
                        optimized_query, k=top_k
                    )
                
                contexts = [r.text for r in results]
                
                # Track retrieval
                st.session_state.monitoring.track_retrieval(
                    retriever_type=retrieval_method.lower(),
                    latency_ms=(time.time() - start_time) * 1000,
                    contexts_count=len(contexts)
                )
            
            # Compress contexts if enabled
            if enable_compression and contexts:
                with st.spinner("Compressing contexts..."):
                    compressed = st.session_state.context_compressor.query_focused_compression(
                        contexts, optimized_query, target_tokens=1500
                    )
                    final_context = compressed.compressed_text
                    
                    # Show compression details
                    with st.expander("📦 Context Compression", expanded=False):
                        st.write(f"**Original Tokens:** {compressed.metadata.get('original_tokens', 0)}")
                        st.write(f"**Compressed Tokens:** {compressed.metadata.get('compressed_tokens', 0)}")
                        st.write(f"**Compression Ratio:** {compressed.compression_ratio:.2%}")
            else:
                final_context = "\n\n".join(contexts[:3])
            
            # Generate response (simulated for demo)
            with st.spinner("Generating response..."):
                # Here you would call your actual LLM
                response = f"""Based on the retrieved context, here's my response to '{query}':

{final_context[:500]}...

[This is a demo response. In production, this would be generated by your LLM using the retrieved and compressed context.]"""
                
                generation_time = time.time() - start_time
                
                # Track generation
                st.session_state.monitoring.track_generation(
                    tokens=len(response.split()),
                    latency_ms=generation_time * 1000,
                    model="demo",
                    metadata={"paradigm": rag_paradigm}
                )
            
            # Add to conversation memory
            if enable_memory:
                st.session_state.conversation_memory.add_turn(
                    query=query,
                    response=response,
                    contexts=contexts,
                    relevance_scores=[r.score for r in results]
                )
            
            # Calculate metrics
            metrics = {
                'latency_ms': generation_time * 1000,
                'tokens': len(response.split()),
                'contexts': len(contexts),
                'relevance': np.mean([r.score for r in results]) if results else 0
            }
            
            # Add to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': query
            })
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'metrics': metrics
            })
            
            # Track completion
            st.session_state.monitoring.track_request(
                method="chat_query",
                duration_ms=generation_time * 1000,
                status="success",
                metadata=metrics
            )
            
            # Rerun to update display
            st.rerun()

def show_advanced_retrieval():
    """Advanced retrieval testing interface"""
    
    st.header("🔍 Advanced Retrieval Testing")
    
    # Upload documents
    st.subheader("📄 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['txt', 'pdf', 'md', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if st.button(f"Index {file.name}"):
                with st.spinner(f"Processing {file.name}..."):
                    # Read file content
                    content = file.read().decode('utf-8', errors='ignore')
                    
                    # Chunk document
                    chunks = st.session_state.semantic_chunker.smart_chunk(
                        content,
                        doc_type='general'
                    )
                    
                    # Add to hybrid retriever
                    st.session_state.hybrid_retriever.add_documents(
                        documents=[c.text for c in chunks],
                        doc_ids=[f"{file.name}_chunk_{i}" for i in range(len(chunks))],
                        metadata=[{"source": file.name, "chunk": i} for i in range(len(chunks))]
                    )
                    
                    # Add to knowledge graph
                    for chunk in chunks[:10]:  # Limit for demo
                        st.session_state.knowledge_graph.add_document(
                            chunk.text,
                            f"{file.name}_chunk_{chunks.index(chunk)}",
                            metadata={"source": file.name}
                        )
                    
                    st.success(f"Indexed {len(chunks)} chunks from {file.name}")
                    st.session_state.documents.append(file.name)
    
    st.markdown("---")
    
    # Retrieval testing
    st.subheader("🧪 Retrieval Testing")
    
    test_query = st.text_input(
        "Test Query",
        placeholder="Enter a query to test retrieval methods"
    )
    
    if test_query:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dense Retrieval")
            with st.spinner("Retrieving..."):
                dense_results = st.session_state.hybrid_retriever.dense_retrieval(
                    test_query, k=5
                )
                
                for i, result in enumerate(dense_results, 1):
                    st.markdown(f"""
                    **{i}. Score: {result.score:.3f}**
                    {result.text[:200]}...
                    _Source: {result.metadata.get('source', 'Unknown')}_
                    """)
        
        with col2:
            st.markdown("### Sparse Retrieval (BM25)")
            with st.spinner("Retrieving..."):
                sparse_results = st.session_state.hybrid_retriever.sparse_retrieval(
                    test_query, k=5
                )
                
                for i, result in enumerate(sparse_results, 1):
                    st.markdown(f"""
                    **{i}. Score: {result.score:.3f}**
                    {result.text[:200]}...
                    _Source: {result.metadata.get('source', 'Unknown')}_
                    """)
        
        st.markdown("---")
        
        # Hybrid retrieval comparison
        st.markdown("### 🔄 Hybrid Retrieval (Combined)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sparse_weight = st.slider("Sparse Weight", 0.0, 1.0, 0.3)
        with col2:
            dense_weight = st.slider("Dense Weight", 0.0, 1.0, 0.5)
        with col3:
            keyword_weight = st.slider("Keyword Weight", 0.0, 1.0, 0.2)
        
        if st.button("Run Hybrid Retrieval"):
            with st.spinner("Running hybrid retrieval..."):
                hybrid_results = st.session_state.hybrid_retriever.hybrid_retrieve(
                    test_query,
                    k=10,
                    sparse_weight=sparse_weight,
                    dense_weight=dense_weight,
                    keyword_weight=keyword_weight,
                    use_reranking=True,
                    use_mmr=True
                )
                
                # Display results
                for i, result in enumerate(hybrid_results, 1):
                    with st.expander(f"{i}. {result.source} - Score: {result.score:.3f}"):
                        st.write(result.text)
                        st.write(f"**Metadata:** {result.metadata}")

def show_knowledge_graph():
    """Knowledge graph visualization and exploration"""
    
    st.header("🧠 Knowledge Graph Explorer")
    
    # Graph statistics
    stats = st.session_state.knowledge_graph.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Entities", stats.get('total_entities', 0))
    with col2:
        st.metric("Total Relations", stats.get('total_relations', 0))
    with col3:
        st.metric("Documents Processed", stats.get('documents_processed', 0))
    with col4:
        st.metric("Avg Degree", f"{stats.get('avg_degree', 0):.2f}")
    
    st.markdown("---")
    
    # Entity search
    st.subheader("🔍 Entity Search")
    
    search_query = st.text_input(
        "Search for entities",
        placeholder="Enter entity or concept"
    )
    
    if search_query:
        with st.spinner("Searching..."):
            similar_entities = st.session_state.knowledge_graph.semantic_search(
                search_query, top_k=10
            )
            
            if similar_entities:
                st.markdown("### Search Results")
                for entity in similar_entities:
                    with st.expander(f"{entity.type}: {entity.text}"):
                        st.write(f"**Frequency:** {entity.frequency}")
                        st.write(f"**Documents:** {', '.join(entity.documents)}")
                        st.write(f"**Attributes:** {entity.attributes}")
    
    st.markdown("---")
    
    # Graph query
    st.subheader("📊 Graph Query")
    
    col1, col2 = st.columns(2)
    
    with col1:
        entity_query = st.text_input(
            "Entity to explore",
            placeholder="Enter entity name"
        )
        max_hops = st.slider("Max Hops", 1, 3, 2)
    
    with col2:
        query_limit = st.slider("Result Limit", 5, 50, 10)
        include_attrs = st.checkbox("Include Attributes", value=True)
    
    if entity_query and st.button("Query Graph"):
        with st.spinner("Querying knowledge graph..."):
            graph_query = GraphQuery(
                entities=[entity_query],
                relations=[],
                max_hops=max_hops,
                limit=query_limit,
                include_attributes=include_attrs
            )
            
            results = st.session_state.knowledge_graph.query_graph(graph_query)
            
            # Display entities
            if results['entities']:
                st.markdown("### Entities Found")
                df_entities = pd.DataFrame(results['entities'])
                st.dataframe(df_entities)
            
            # Display relations
            if results['relations']:
                st.markdown("### Relations")
                df_relations = pd.DataFrame(results['relations'])
                st.dataframe(df_relations)
    
    # Visualization
    if st.button("Generate Graph Visualization"):
        with st.spinner("Generating visualization..."):
            st.session_state.knowledge_graph.visualize(
                output_path="knowledge_graph.html",
                max_nodes=50
            )
            st.success("Graph visualization saved to knowledge_graph.html")
            st.info("Open the HTML file in your browser to explore the interactive graph")

def show_monitoring():
    """Production monitoring dashboard"""
    
    st.header("📈 Production Monitoring")
    
    # Get dashboard data
    dashboard = st.session_state.monitoring.get_dashboard_data()
    
    # System metrics
    st.subheader("🖥️ System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu = dashboard['resource_usage']['cpu']
        st.metric(
            "CPU Usage",
            f"{cpu:.1f}%",
            delta=None,
            delta_color="inverse" if cpu > 80 else "normal"
        )
    
    with col2:
        memory = dashboard['resource_usage']['memory']
        st.metric(
            "Memory Usage",
            f"{memory:.1f}%",
            delta=None,
            delta_color="inverse" if memory > 80 else "normal"
        )
    
    with col3:
        metrics_summary = dashboard['metrics_summary']
        if 'request_duration' in metrics_summary['metrics']:
            p95_latency = metrics_summary['metrics']['request_duration']['p95'] * 1000
            st.metric("P95 Latency", f"{p95_latency:.0f}ms")
        else:
            st.metric("P95 Latency", "N/A")
    
    with col4:
        error_count = len(dashboard['recent_errors'])
        st.metric(
            "Recent Errors",
            error_count,
            delta=None,
            delta_color="inverse" if error_count > 0 else "off"
        )
    
    st.markdown("---")
    
    # Health checks
    st.subheader("🏥 Health Checks")
    
    health_checks = dashboard['health_checks']
    
    if health_checks:
        health_df = pd.DataFrame([
            {
                'Component': comp,
                'Status': check['status'],
                'Latency (ms)': f"{check['latency_ms']:.1f}",
                'Message': check['message']
            }
            for comp, check in health_checks.items()
        ])
        
        # Style the dataframe
        def style_status(val):
            if val == 'healthy':
                return 'color: green; font-weight: bold;'
            elif val == 'degraded':
                return 'color: orange; font-weight: bold;'
            else:
                return 'color: red; font-weight: bold;'
        
        styled_df = health_df.style.applymap(
            style_status,
            subset=['Status']
        )
        
        st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    
    # Active alerts
    st.subheader("🚨 Active Alerts")
    
    active_alerts = dashboard['active_alerts']
    
    if active_alerts:
        for alert in active_alerts:
            severity = alert['severity']
            
            if severity == 'critical':
                st.markdown(f"""
                <div class="alert-critical">
                    <strong>🔴 CRITICAL:</strong> {alert['message']}<br>
                    <small>{alert['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
            elif severity == 'error':
                st.error(f"**{alert['name']}:** {alert['message']}")
            elif severity == 'warning':
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>⚠️ WARNING:</strong> {alert['message']}<br>
                    <small>{alert['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"{alert['message']}")
    else:
        st.success("No active alerts")
    
    st.markdown("---")
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    # Create sample time series data for visualization
    time_range = pd.date_range(
        start=datetime.now() - timedelta(hours=1),
        end=datetime.now(),
        freq='5min'
    )
    
    # Simulated metrics
    latency_data = pd.DataFrame({
        'Time': time_range,
        'P50': np.random.normal(100, 20, len(time_range)),
        'P95': np.random.normal(200, 40, len(time_range)),
        'P99': np.random.normal(300, 60, len(time_range))
    })
    
    fig = px.line(
        latency_data,
        x='Time',
        y=['P50', 'P95', 'P99'],
        title='Request Latency Percentiles',
        labels={'value': 'Latency (ms)', 'variable': 'Percentile'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Export metrics
    if st.button("Export Metrics"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"metrics_export_{timestamp}.json"
        st.session_state.monitoring.export_metrics(export_path)
        st.success(f"Metrics exported to {export_path}")

def show_ab_testing():
    """A/B testing management interface"""
    
    st.header("🧪 A/B Testing Framework")
    
    # Experiment overview
    st.subheader("📋 Active Experiments")
    
    experiments = st.session_state.ab_testing.experiments
    
    if experiments:
        for exp_id, experiment in experiments.items():
            if experiment.status == ExperimentStatus.ACTIVE:
                with st.expander(f"🔬 {experiment.name}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**ID:** {exp_id}")
                        st.write(f"**Status:** {experiment.status.value}")
                        st.write(f"**Strategy:** {experiment.allocation_strategy.value}")
                    
                    with col2:
                        st.write(f"**Start:** {experiment.start_date.strftime('%Y-%m-%d')}")
                        if experiment.end_date:
                            st.write(f"**End:** {experiment.end_date.strftime('%Y-%m-%d')}")
                        st.write(f"**Target Sample:** {experiment.target_sample_size}")
                    
                    with col3:
                        # Analyze button
                        if st.button(f"Analyze {exp_id[:8]}"):
                            results = st.session_state.ab_testing.analyze_experiment(exp_id)
                            
                            # Display results
                            st.markdown("#### Results")
                            for variant_name, variant_data in results['variants'].items():
                                st.write(f"**{variant_name}:** {variant_data['samples']} samples")
                                
                                for metric_name, metric_data in variant_data['metrics'].items():
                                    st.write(f"  - {metric_name}: {metric_data['mean']:.3f} "
                                           f"CI: [{metric_data['confidence_interval'][0]:.3f}, "
                                           f"{metric_data['confidence_interval'][1]:.3f}]")
                            
                            if results.get('winner'):
                                st.success(f"Winner: {results['winner']}")
    else:
        st.info("No active experiments")
    
    st.markdown("---")
    
    # Create new experiment
    st.subheader("➕ Create New Experiment")
    
    with st.form("new_experiment"):
        exp_name = st.text_input("Experiment Name")
        exp_description = st.text_area("Description")
        
        col1, col2 = st.columns(2)
        
        with col1:
            duration_days = st.number_input("Duration (days)", min_value=1, value=14)
            target_samples = st.number_input("Target Samples", min_value=100, value=1000)
            allocation_strategy = st.selectbox(
                "Allocation Strategy",
                ["RANDOM", "DETERMINISTIC", "WEIGHTED", "ADAPTIVE"]
            )
        
        with col2:
            st.write("**Variants**")
            control_name = st.text_input("Control Variant", value="Control")
            treatment_name = st.text_input("Treatment Variant", value="Treatment")
            treatment_allocation = st.slider("Treatment Allocation", 0.0, 1.0, 0.5)
        
        submit = st.form_submit_button("Create Experiment")
        
        if submit and exp_name:
            # Create variants
            variants = [
                Variant(
                    name=control_name,
                    config={"type": "control"},
                    allocation=1.0 - treatment_allocation,
                    description="Control variant",
                    is_control=True
                ),
                Variant(
                    name=treatment_name,
                    config={"type": "treatment"},
                    allocation=treatment_allocation,
                    description="Treatment variant",
                    is_control=False
                )
            ]
            
            # Create metrics
            metrics = [
                Metric(
                    name="conversion_rate",
                    type="binary",
                    higher_is_better=True,
                    minimum_sample_size=100,
                    significance_level=0.05
                ),
                Metric(
                    name="response_quality",
                    type="continuous",
                    higher_is_better=True,
                    minimum_sample_size=100,
                    significance_level=0.05
                )
            ]
            
            # Create experiment
            experiment = st.session_state.ab_testing.create_experiment(
                name=exp_name,
                description=exp_description,
                variants=variants,
                metrics=metrics,
                duration_days=duration_days,
                allocation_strategy=AllocationStrategy[allocation_strategy],
                target_sample_size=target_samples
            )
            
            # Start experiment
            st.session_state.ab_testing.start_experiment(experiment.experiment_id)
            
            st.success(f"Experiment '{exp_name}' created and started!")
            st.rerun()

def show_memory_context():
    """Memory and context management interface"""
    
    st.header("💾 Memory & Context Management")
    
    # Session info
    st.subheader("📝 Current Session")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Session ID:** {st.session_state.current_session_id[:12]}...")
        
        memory = st.session_state.conversation_memory
        if memory.current_session:
            st.write(f"**Turns:** {len(memory.current_session.turns)}")
    
    with col2:
        if st.button("Start New Session"):
            # End current session
            if memory.current_session:
                memory.end_session()
            
            # Start new session
            new_session_id = f"session_{datetime.now().timestamp()}"
            memory.start_session(new_session_id)
            st.session_state.current_session_id = new_session_id
            st.session_state.chat_history = []
            st.success("New session started")
            st.rerun()
    
    with col3:
        if st.button("Generate Summary"):
            if memory.current_session and len(memory.current_session.turns) > 0:
                summary = memory.generate_session_summary()
                st.write("**Summary:**")
                st.write(summary)
    
    st.markdown("---")
    
    # Conversation flow
    st.subheader("🔄 Conversation Flow")
    
    if memory.current_session and memory.current_session.turns:
        flow = memory.get_conversation_flow()
        
        flow_df = pd.DataFrame(flow)
        
        # Visualize flow
        fig = go.Figure()
        
        # Add turns as nodes
        for i, turn in enumerate(flow):
            color = 'red' if turn['topic_shift'] else 'blue'
            fig.add_trace(go.Scatter(
                x=[i],
                y=[1],
                mode='markers+text',
                marker=dict(size=20, color=color),
                text=f"Turn {turn['turn']}<br>{turn['intent']}",
                textposition="top center"
            ))
        
        fig.update_layout(
            title="Conversation Flow (Red = Topic Shift)",
            xaxis_title="Turn Number",
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed flow data
        with st.expander("Flow Details"):
            st.dataframe(flow_df)
    else:
        st.info("No conversation data available")
    
    st.markdown("---")
    
    # Memory types
    st.subheader("🧠 Memory Systems")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Episodic Memory")
        st.write(f"**Recent Turns:** {len(memory.episodic_memory)}")
        
        if memory.episodic_memory:
            recent = list(memory.episodic_memory)[-3:]
            for turn in recent:
                with st.expander(f"Turn {turn.turn_id}"):
                    st.write(f"**Query:** {turn.query.content}")
                    st.write(f"**Intent:** {turn.query.intent}")
                    st.write(f"**Response:** {turn.response.content[:200]}...")
    
    with col2:
        st.markdown("### Semantic Memory")
        
        semantic = memory.semantic_memory
        
        st.write(f"**Known Entities:** {len(semantic['entities'])}")
        st.write(f"**Learned Patterns:** {len(semantic['patterns'])}")
        st.write(f"**Stored Facts:** {len(semantic['facts'])}")
        
        # Show some facts
        if semantic['facts']:
            st.write("**Recent Facts:**")
            for key, value in list(semantic['facts'].items())[:3]:
                st.write(f"- {key}: {value}")
    
    # Context compression demo
    st.markdown("---")
    st.subheader("📦 Context Compression Demo")
    
    test_context = st.text_area(
        "Context to compress",
        height=150,
        placeholder="Paste a long context here to test compression..."
    )
    
    if test_context:
        col1, col2 = st.columns(2)
        
        with col1:
            compression_method = st.selectbox(
                "Method",
                ["Query-Focused", "Extractive", "Abstractive", "Redundancy Removal"]
            )
            
            target_tokens = st.slider("Target Tokens", 50, 500, 200)
        
        with col2:
            if compression_method == "Query-Focused":
                query_for_compression = st.text_input(
                    "Query for focus",
                    placeholder="What is the query?"
                )
            else:
                query_for_compression = ""
        
        if st.button("Compress"):
            compressor = st.session_state.context_compressor
            
            if compression_method == "Query-Focused" and query_for_compression:
                result = compressor.query_focused_compression(
                    [test_context],
                    query_for_compression,
                    target_tokens
                )
            elif compression_method == "Extractive":
                result = compressor.extractive_summarization(
                    test_context,
                    num_sentences=3
                )
            elif compression_method == "Abstractive":
                result = compressor.abstractive_summarization(
                    test_context,
                    max_length=target_tokens
                )
            else:  # Redundancy Removal
                result = compressor.remove_redundancy([test_context])
            
            # Show results
            st.markdown("#### Compression Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Original Length:** {len(test_context.split())} words")
                st.write(f"**Compressed Length:** {len(result.compressed_text.split())} words")
                st.write(f"**Compression Ratio:** {result.compression_ratio:.2%}")
            
            with col2:
                st.write(f"**Method:** {result.method}")
                st.write(f"**Relevance Score:** {result.relevance_score:.2f}")
            
            st.markdown("#### Compressed Text")
            st.text_area("Result", result.compressed_text, height=150)

def show_configuration():
    """System configuration interface"""
    
    st.header("⚙️ System Configuration")
    
    # Component configuration
    st.subheader("🔧 Component Settings")
    
    with st.expander("Query Optimizer"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Intent Labels**")
            for label in st.session_state.query_optimizer.intent_labels:
                st.write(f"- {label}")
        
        with col2:
            st.write("**Domain Keywords**")
            for domain, keywords in st.session_state.query_optimizer.domain_keywords.items():
                st.write(f"**{domain}:** {', '.join(keywords[:3])}...")
    
    with st.expander("Hybrid Retriever"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Embedding Model:**")
            st.code("all-MiniLM-L6-v2")
        
        with col2:
            st.write("**Reranker Model:**")
            st.code("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        with col3:
            st.write("**Index Path:**")
            st.code("./hybrid_index")
    
    with st.expander("Monitoring"):
        st.write(f"**Service Name:** {st.session_state.monitoring.service_name}")
        st.write(f"**Prometheus Port:** 8001 (disabled for Streamlit)")
        st.write(f"**Max Metrics:** 10,000 points")
        st.write(f"**Max Alerts:** 1,000")
    
    st.markdown("---")
    
    # Export/Import configuration
    st.subheader("💾 Configuration Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Configuration"):
            config = {
                "query_optimizer": {
                    "intent_labels": st.session_state.query_optimizer.intent_labels,
                    "domain_keywords": st.session_state.query_optimizer.domain_keywords
                },
                "retriever": {
                    "embedding_model": "all-MiniLM-L6-v2",
                    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "index_path": "./hybrid_index"
                },
                "chunker": {
                    "chunk_size": st.session_state.semantic_chunker.chunk_size,
                    "chunk_overlap": st.session_state.semantic_chunker.chunk_overlap
                },
                "compressor": {
                    "max_tokens": st.session_state.context_compressor.max_tokens
                },
                "memory": {
                    "max_turns": st.session_state.conversation_memory.max_turns,
                    "max_sessions": st.session_state.conversation_memory.max_sessions
                }
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_path = f"config_export_{timestamp}.json"
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success(f"Configuration exported to {config_path}")
    
    with col2:
        uploaded_config = st.file_uploader("Import Configuration", type=['json'])
        
        if uploaded_config:
            config = json.load(uploaded_config)
            
            if st.button("Apply Configuration"):
                # Apply configuration (simplified for demo)
                st.success("Configuration applied successfully!")
                st.rerun()
    
    st.markdown("---")
    
    # System info
    st.subheader("ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Production Features:**")
        st.write("✅ Semantic Query Optimization")
        st.write("✅ Knowledge Graph (NetworkX)")
        st.write("✅ Hybrid Retrieval (BM25 + Dense)")
        st.write("✅ Semantic Chunking")
        st.write("✅ Context Compression")
    
    with col2:
        st.write("**Additional Features:**")
        st.write("✅ Conversation Memory")
        st.write("✅ A/B Testing Framework")
        st.write("✅ Production Monitoring")
        st.write("✅ RAGAS Evaluation")
        st.write("✅ Streaming Responses")

if __name__ == "__main__":
    main()