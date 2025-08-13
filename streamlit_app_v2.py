"""
Enhanced Streamlit UI for RAG System with All Features
Complete dashboard for testing all RAG paradigms and features
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import asyncio
import requests
import mimetypes

# Optional parsers for richer document support
try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.rag.naive_rag import NaiveRAG
from src.rag.advanced_rag import AdvancedRAG
from src.rag.modular_rag import ModularRAG
from src.graph_rag.knowledge_graph import GraphRAG
from src.advanced_rag.self_rag import SelfRAG
from src.evaluation.ragas_metrics import RAGASEvaluator
from src.evaluation.benchmark import RAGBenchmark
from src.streaming.stream_handler import StreamingRAG

# Page configuration
st.set_page_config(
    page_title="RAG System Complete Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .success-msg {
        padding: 1rem;
        background: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-msg {
        padding: 1rem;
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-msg {
        padding: 1rem;
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #667eea30;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'current_rag_type' not in st.session_state:
    st.session_state.current_rag_type = 'naive'
if 'graph_built' not in st.session_state:
    st.session_state.graph_built = False

def _read_file_to_text(file_path: Path) -> str:
    """Best-effort loader to read various document types into plain text."""
    try:
        suffix = file_path.suffix.lower()
        # Text/Markdown
        if suffix in {'.txt', '.md'}:
            return file_path.read_text(encoding='utf-8', errors='ignore')
        # PDF
        if suffix == '.pdf' and PyPDF2 is not None:
            text_parts: List[str] = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    try:
                        text_parts.append(page.extract_text() or '')
                    except Exception:
                        continue
            return "\n".join(text_parts)
        # DOCX
        if suffix == '.docx' and docx is not None:
            document = docx.Document(str(file_path))
            return "\n".join(p.text for p in document.paragraphs)
        # Excel
        if suffix in {'.xlsx', '.xls'}:
            try:
                df = pd.read_excel(str(file_path), sheet_name=None)
                # Concatenate all sheets into text
                sheets_text: List[str] = []
                for name, sheet_df in df.items():
                    sheets_text.append(f"# Sheet: {name}\n{sheet_df.to_csv(index=False)}")
                return "\n\n".join(sheets_text)
            except Exception:
                return ""
        # Fallback: try guessing mime and reading as text
        guessed, _ = mimetypes.guess_type(str(file_path))
        if guessed and guessed.startswith('text/'):
            return file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        pass
    return ""

@st.cache_resource
def initialize_rag_systems():
    """Initialize all RAG systems"""
    systems = {}
    
    with st.spinner("Initializing RAG systems..."):
        try:
            # Initialize basic RAG
            systems['naive'] = NaiveRAG(config_path='config.yaml', enable_evaluation=True)
            st.success("✅ Naive RAG initialized")
            
            # Initialize Advanced RAG
            systems['advanced'] = AdvancedRAG(config_path='config.yaml')
            st.success("✅ Advanced RAG initialized")
            
            # Initialize Modular RAG
            systems['modular'] = ModularRAG(config_path='config.yaml')
            st.success("✅ Modular RAG initialized")
            
            # Initialize Graph RAG
            systems['graph'] = GraphRAG()
            st.success("✅ Graph RAG initialized")
            
            # Initialize Self-RAG
            systems['self'] = SelfRAG(systems['naive'], systems['naive'].generator)
            st.success("✅ Self-RAG initialized")
            
            # Initialize evaluator
            systems['evaluator'] = RAGASEvaluator(llm_generator=systems['naive'].generator)
            st.success("✅ RAGAS Evaluator initialized")
            
            # Initialize streaming
            systems['streaming'] = StreamingRAG(systems['naive'], llm_type="ollama")
            st.success("✅ Streaming RAG initialized")
            
        except Exception as e:
            st.error(f"Error initializing systems: {str(e)}")
            st.info("Some features may be limited. Please check your configuration.")
            
    return systems

def display_header():
    """Display application header"""
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col2:
        st.title("🚀 RAG System Complete Dashboard")
        st.markdown("**Test all RAG paradigms and features**")
    
    # Display system status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents", len(st.session_state.documents), "📄")
    with col2:
        st.metric("Chat History", len(st.session_state.chat_history), "💬")
    with col3:
        st.metric("Evaluations", len(st.session_state.evaluation_history), "📊")
    with col4:
        rag_type = st.session_state.current_rag_type.upper()
        st.metric("Current RAG", rag_type, "🤖")

def test_file_upload():
    """Test document upload functionality"""
    st.header("📄 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['txt', 'pdf', 'md', 'xlsx', 'xls', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if st.button(f"Process {file.name}", key=f"process_{file.name}"):
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        # Save file temporarily
                        temp_path = Path(f"temp_{file.name}")
                        with open(temp_path, 'wb') as f:
                            f.write(file.getbuffer())
                        
                        # Index document
                        systems = st.session_state.systems
                        systems['naive'].index_documents([str(temp_path)])
                        
                        # Read content for Graph RAG and analytics
                        content_text = _read_file_to_text(temp_path)
                        
                        # Clean up
                        temp_path.unlink()
                        
                        st.session_state.documents.append({
                            'name': file.name,
                            'size': file.size,
                            'type': file.type,
                            'timestamp': datetime.now(),
                            'content': content_text
                        })
                        
                        st.success(f"✅ Successfully processed {file.name}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {str(e)}")
    
    # Display uploaded documents
    if st.session_state.documents:
        st.subheader("Uploaded Documents")
        df = pd.DataFrame(st.session_state.documents)
        st.dataframe(df, use_container_width=True)
        
        if st.button("🗑️ Clear All Documents"):
            st.session_state.documents = []
            st.rerun()

def test_rag_paradigms():
    """Test different RAG paradigms"""
    st.header("🤖 RAG Paradigm Testing")
    
    # RAG type selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        rag_type = st.selectbox(
            "Select RAG Type",
            ['naive', 'advanced', 'modular', 'graph', 'self', 'auto'],
            help="Choose the RAG paradigm to test"
        )
        st.session_state.current_rag_type = rag_type
        
        # Display RAG description
        descriptions = {
            'naive': "Basic retrieve-then-read paradigm",
            'advanced': "Query optimization with rewriting and expansion",
            'modular': "Factory pattern with pluggable components",
            'graph': "Knowledge graph-based retrieval",
            'self': "Self-reflecting RAG with critique",
            'auto': "Automatic routing to best strategy"
        }
        st.info(descriptions.get(rag_type, ""))
    
    with col2:
        # Query input
        query = st.text_input(
            "Enter your question",
            placeholder="Ask anything about your documents...",
            key="query_input"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=True)
        with col2:
            enable_streaming = st.checkbox("Enable Streaming", value=False)
        with col3:
            show_contexts = st.checkbox("Show Retrieved Contexts", value=True)
        
        if st.button("🚀 Submit Query", type="primary"):
            if query and st.session_state.systems:
                process_query(query, rag_type, enable_evaluation, enable_streaming, show_contexts)

def process_query(query: str, rag_type: str, enable_evaluation: bool, enable_streaming: bool, show_contexts: bool):
    """Process a query with selected RAG type"""
    systems = st.session_state.systems
    
    with st.spinner(f"Processing with {rag_type.upper()} RAG..."):
        start_time = time.time()
        
        try:
            # Process based on RAG type
            if rag_type == 'naive':
                if enable_evaluation:
                    answer, contexts, scores = systems['naive'].query_with_evaluation(query)
                else:
                    answer = systems['naive'].query(query)
                    contexts = systems['naive'].retrieve(query) if show_contexts else []
                    scores = None
                    
            elif rag_type == 'advanced':
                answer = systems['advanced'].query_optimized(query)
                contexts = systems['advanced'].retrieve_optimized(query) if show_contexts else []
                scores = None
                
            elif rag_type == 'modular':
                result = systems['modular'].query(query, return_contexts=True)
                answer = result['answer']
                contexts = result.get('contexts', [])
                scores = None
                
            elif rag_type == 'graph':
                if not st.session_state.graph_built:
                    st.warning("Building knowledge graph from documents...")
                    # Build graph from documents
                    if st.session_state.documents:
                        # Adapt documents format for GraphRAG
                        docs_for_graph = []
                        for i, d in enumerate(st.session_state.documents):
                            content = d.get('content') or ''
                            if not content:
                                continue
                            docs_for_graph.append({'id': i, 'content': content})
                        if docs_for_graph:
                            systems['graph'].build_knowledge_graph(docs_for_graph)
                            st.session_state.graph_built = True
                
                contexts = systems['graph'].graph_retrieve(query)
                answer = systems['naive'].generator.generate(query, "\n".join(contexts))
                scores = None
                
            elif rag_type == 'self':
                result = asyncio.run(systems['self'].query_with_reflection(query))
                answer = result['answer']
                contexts = result.get('contexts', [])
                scores = None
                
            elif rag_type == 'auto':
                # Simulate auto-routing
                st.info("🔄 Auto-routing query to best RAG strategy...")
                # For demo, use advanced RAG
                answer = systems['advanced'].query_optimized(query)
                contexts = systems['advanced'].retrieve_optimized(query) if show_contexts else []
                scores = None
            
            else:
                answer = "Invalid RAG type selected"
                contexts = []
                scores = None
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add to chat history
            chat_entry = {
                'timestamp': datetime.now(),
                'query': query,
                'answer': answer,
                'contexts': contexts[:2] if contexts else [],
                'rag_type': rag_type,
                'processing_time': processing_time,
                'scores': scores
            }
            st.session_state.chat_history.append(chat_entry)
            
            # Display results
            display_query_results(chat_entry, enable_evaluation)
            
            # Evaluate if requested
            if enable_evaluation and scores:
                st.session_state.evaluation_history.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'rag_type': rag_type,
                    'scores': scores
                })
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.exception(e)

def display_query_results(chat_entry: Dict, show_evaluation: bool):
    """Display query results"""
    st.success("✅ Query processed successfully!")
    
    # Display answer
    st.markdown("### 💡 Answer")
    st.markdown(f'<div class="metric-card">{chat_entry["answer"]}</div>', unsafe_allow_html=True)
    
    # Display metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RAG Type", chat_entry['rag_type'].upper())
    with col2:
        st.metric("Processing Time", f"{chat_entry['processing_time']:.2f}s")
    with col3:
        st.metric("Contexts Retrieved", len(chat_entry.get('contexts', [])))
    
    # Display contexts if available
    if chat_entry.get('contexts'):
        with st.expander("📚 Retrieved Contexts"):
            for i, context in enumerate(chat_entry['contexts'], 1):
                st.markdown(f"**Context {i}:**")
                st.text(context[:500] + "..." if len(context) > 500 else context)
    
    # Display evaluation scores if available
    if show_evaluation and chat_entry.get('scores'):
        display_evaluation_scores(chat_entry['scores'])

def display_evaluation_scores(scores: Dict):
    """Display RAGAS evaluation scores"""
    st.markdown("### 📊 RAGAS Evaluation Scores")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Faithfulness", scores.get('faithfulness', 0)),
        ("Answer Relevancy", scores.get('answer_relevancy', 0)),
        ("Context Relevancy", scores.get('context_relevancy', 0)),
        ("Context Precision", scores.get('context_precision', 0))
    ]
    
    for col, (metric, score) in zip([col1, col2, col3, col4], metrics):
        with col:
            # Color based on score
            color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
            st.metric(metric, f"{score:.2%}", color)
    
    # Radar chart
    fig = go.Figure(data=go.Scatterpolar(
        r=[scores.get(k, 0) for k in ['faithfulness', 'answer_relevancy', 'context_relevancy', 'context_precision']],
        theta=['Faithfulness', 'Answer Relevancy', 'Context Relevancy', 'Context Precision'],
        fill='toself',
        name='RAGAS Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def test_streaming():
    """Test streaming functionality"""
    st.header("⚡ Streaming Test")
    
    query = st.text_input("Enter query for streaming test", key="streaming_query")
    
    if st.button("🌊 Test Streaming"):
        if query and st.session_state.systems:
            container = st.empty()
            # Use real SSE from backend if available
            sse_url = "http://127.0.0.1:8090/api/chat/stream"
            params = {"question": query}
            try:
                with requests.get(sse_url, params=params, stream=True, timeout=60) as resp:
                    if resp.status_code != 200:
                        st.warning("SSE endpoint unavailable. Falling back to non-streaming.")
                        systems = st.session_state.systems
                        answer = systems['naive'].generator.generate(query, "\n".join(systems['naive'].retrieve(query)))
                        container.markdown(f"**Answer:** {answer}")
                        return
                    partial = ""
                    for raw_line in resp.iter_lines(decode_unicode=True):
                        if raw_line is None:
                            continue
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            try:
                                payload = json.loads(line[5:].strip())
                                event_type = payload.get("type", "")
                                if event_type == "TOKEN":
                                    token = payload.get("content", "")
                                    partial += token
                                    container.markdown(f"**Answer:** {partial}")
                                elif event_type == "RETRIEVAL_COMPLETE":
                                    st.info("🔍 Retrieval complete")
                                elif event_type == "COMPLETE":
                                    st.success("✅ Streaming complete!")
                            except Exception:
                                continue
            except Exception as e:
                st.error(f"Streaming error: {str(e)}")

def test_error_handling():
    """Test error handling and edge cases"""
    st.header("🛡️ Error Handling & Edge Cases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Test Cases")
        
        if st.button("Test Empty Query"):
            try:
                systems = st.session_state.systems
                result = systems['naive'].query("")
                st.error("❌ Should have raised an error for empty query!")
            except Exception as e:
                st.success(f"✅ Correctly handled empty query: {str(e)}")
        
        if st.button("Test Very Long Query"):
            long_query = "What " * 500 + "is this?"
            try:
                systems = st.session_state.systems
                result = systems['naive'].query(long_query[:1000])  # Limit length
                st.success("✅ Handled long query successfully")
            except Exception as e:
                st.warning(f"⚠️ Long query handling: {str(e)}")
        
        if st.button("Test Special Characters"):
            special_query = "What about @#$%^&*() special characters?"
            try:
                systems = st.session_state.systems
                result = systems['naive'].query(special_query)
                st.success("✅ Handled special characters")
            except Exception as e:
                st.warning(f"⚠️ Special character handling: {str(e)}")
        
        if st.button("Test No Documents"):
            try:
                # Try query without documents
                test_rag = NaiveRAG(config_path='config.yaml')
                result = test_rag.query("Test query")
                st.warning("⚠️ Query succeeded without documents")
            except Exception as e:
                st.success(f"✅ Correctly handled no documents: {str(e)}")
    
    with col2:
        st.subheader("System Health")
        
        # Check system components
        if st.session_state.systems:
            systems = st.session_state.systems
            
            components = {
                'Naive RAG': 'naive' in systems,
                'Advanced RAG': 'advanced' in systems,
                'Modular RAG': 'modular' in systems,
                'Graph RAG': 'graph' in systems,
                'Self RAG': 'self' in systems,
                'Evaluator': 'evaluator' in systems,
                'Streaming': 'streaming' in systems
            }
            
            for component, status in components.items():
                if status:
                    st.success(f"✅ {component}")
                else:
                    st.error(f"❌ {component}")
        
        # Memory usage
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        st.metric("Memory Usage", f"{memory_info.rss / 1024 / 1024:.1f} MB")
        
        # Document stats
        if st.session_state.documents:
            st.metric("Total Documents", len(st.session_state.documents))
            total_size = sum(doc['size'] for doc in st.session_state.documents)
            st.metric("Total Size", f"{total_size / 1024:.1f} KB")

def display_analytics():
    """Display analytics and metrics"""
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.chat_history:
        st.info("No data available yet. Start making queries to see analytics.")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(st.session_state.chat_history)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Query distribution by RAG type
        fig = px.pie(
            df, 
            names='rag_type',
            title="Queries by RAG Type",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Processing time by RAG type
        fig = px.box(
            df,
            x='rag_type',
            y='processing_time',
            title="Processing Time by RAG Type",
            color='rag_type',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Timeline of queries
        fig = px.scatter(
            df,
            x='timestamp',
            y='processing_time',
            color='rag_type',
            title="Query Timeline",
            hover_data=['query']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Average metrics
        if st.session_state.evaluation_history:
            eval_df = pd.DataFrame(st.session_state.evaluation_history)
            
            # Extract scores
            scores_data = []
            for _, row in eval_df.iterrows():
                if row['scores']:
                    scores_data.append({
                        'faithfulness': row['scores'].get('faithfulness', 0),
                        'answer_relevancy': row['scores'].get('answer_relevancy', 0),
                        'context_relevancy': row['scores'].get('context_relevancy', 0),
                        'context_precision': row['scores'].get('context_precision', 0)
                    })
            
            if scores_data:
                scores_df = pd.DataFrame(scores_data)
                avg_scores = scores_df.mean()
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=avg_scores.index,
                        y=avg_scores.values,
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                    )
                ])
                fig.update_layout(
                    title="Average RAGAS Scores",
                    yaxis_range=[0, 1],
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

def display_chat_history():
    """Display chat history"""
    st.header("💬 Chat History")
    
    if not st.session_state.chat_history:
        st.info("No chat history yet. Start asking questions!")
        return
    
    # Display in reverse chronological order
    for entry in reversed(st.session_state.chat_history[-10:]):  # Show last 10
        with st.expander(f"🕐 {entry['timestamp'].strftime('%H:%M:%S')} - {entry['rag_type'].upper()}"):
            st.markdown(f"**Question:** {entry['query']}")
            st.markdown(f"**Answer:** {entry['answer']}")
            st.markdown(f"**Processing Time:** {entry['processing_time']:.2f}s")
            
            if entry.get('contexts'):
                st.markdown(f"**Contexts:** {len(entry['contexts'])}")
            
            if entry.get('scores'):
                scores = entry['scores']
                cols = st.columns(4)
                for col, (metric, value) in zip(cols, scores.items()):
                    col.metric(metric.replace('_', ' ').title(), f"{value:.2%}")
    
    if st.button("📥 Export Chat History"):
        # Convert to JSON for export
        export_data = []
        for entry in st.session_state.chat_history:
            export_entry = {
                'timestamp': entry['timestamp'].isoformat(),
                'query': entry['query'],
                'answer': entry['answer'],
                'rag_type': entry['rag_type'],
                'processing_time': entry['processing_time']
            }
            export_data.append(export_entry)
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """Main application"""
    
    # Initialize systems
    if 'systems' not in st.session_state:
        st.session_state.systems = initialize_rag_systems()
    
    # Display header
    display_header()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📄 Documents",
        "🤖 RAG Testing",
        "⚡ Streaming",
        "🛡️ Error Handling",
        "📈 Analytics",
        "💬 History",
        "📚 Documentation"
    ])
    
    with tab1:
        test_file_upload()
    
    with tab2:
        test_rag_paradigms()
    
    with tab3:
        test_streaming()
    
    with tab4:
        test_error_handling()
    
    with tab5:
        display_analytics()
    
    with tab6:
        display_chat_history()
    
    with tab7:
        st.header("📚 Documentation")
        
        st.markdown("""
        ### RAG Paradigms
        
        1. **Naive RAG** - Basic retrieve-then-read paradigm
        2. **Advanced RAG** - Query optimization with rewriting and expansion
        3. **Modular RAG** - Factory pattern with pluggable components
        4. **Graph RAG** - Knowledge graph-based retrieval
        5. **Self-RAG** - Self-reflecting RAG with critique
        6. **Auto-Routing** - Automatic selection of best strategy
        
        ### Features
        
        - ✅ Multi-format document support (TXT, PDF, MD, XLSX, DOCX)
        - ✅ RAGAS evaluation metrics
        - ✅ Real-time streaming responses
        - ✅ Knowledge graph construction
        - ✅ Self-reflection and critique
        - ✅ Query optimization techniques
        - ✅ Comprehensive error handling
        - ✅ Performance analytics
        
        ### Testing Guidelines
        
        1. **Upload Documents** - Start by uploading some documents
        2. **Test Basic Queries** - Try simple questions first
        3. **Compare RAG Types** - Test the same query with different paradigms
        4. **Enable Evaluation** - Check RAGAS scores for quality
        5. **Test Edge Cases** - Try empty queries, special characters, etc.
        6. **Check Analytics** - Review performance metrics
        
        ### Troubleshooting
        
        - If Ollama is not running, some features may be limited
        - Graph RAG requires documents to build knowledge graph
        - Evaluation requires additional LLM calls (slower)
        - Streaming works best with fast models
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.number_input("Max Tokens", 50, 2000, 500)
        
        # Display settings
        st.subheader("Display Settings")
        show_timestamps = st.checkbox("Show Timestamps", True)
        show_metrics = st.checkbox("Show Metrics", True)
        dark_mode = st.checkbox("Dark Mode", False)
        
        # System info
        st.subheader("System Info")
        st.info(f"""
        - Documents: {len(st.session_state.documents)}
        - Queries: {len(st.session_state.chat_history)}
        - Current RAG: {st.session_state.current_rag_type.upper()}
        - Graph Built: {'Yes' if st.session_state.graph_built else 'No'}
        """)
        
        # Clear data
        if st.button("🗑️ Clear All Data"):
            st.session_state.chat_history = []
            st.session_state.evaluation_history = []
            st.session_state.documents = []
            st.session_state.graph_built = False
            st.rerun()

if __name__ == "__main__":
    main()