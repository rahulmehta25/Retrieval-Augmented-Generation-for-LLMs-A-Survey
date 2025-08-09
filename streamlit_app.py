"""
Streamlit UI for RAG System

A user-friendly interface for interacting with the RAG system,
uploading documents, and viewing evaluation metrics.
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.rag.naive_rag import NaiveRAG
from src.rag.advanced_rag import AdvancedRAG
from src.rag.modular_rag import ModularRAG
from src.evaluation.ragas_metrics import RAGASEvaluator
from src.evaluation.benchmark import RAGBenchmark

# Page configuration
st.set_page_config(
    page_title="RAG System Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 20px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 5px;
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.rag_type = "naive"
    st.session_state.evaluator = None
    st.session_state.chat_history = []
    st.session_state.evaluation_results = []
    st.session_state.documents_loaded = False
    st.session_state.uploaded_files = []

def initialize_rag_system(rag_type: str = "naive", enable_evaluation: bool = True):
    """Initialize the selected RAG system"""
    try:
        with st.spinner(f"Initializing {rag_type.upper()} RAG system..."):
            if rag_type == "naive":
                rag = NaiveRAG(config_path='config.yaml', enable_evaluation=enable_evaluation)
            elif rag_type == "advanced":
                rag = AdvancedRAG(config_path='config.yaml')
            elif rag_type == "modular":
                rag = ModularRAG(config_path='config.yaml')
            else:
                raise ValueError(f"Unknown RAG type: {rag_type}")
            
            st.session_state.rag_system = rag
            st.session_state.rag_type = rag_type
            
            if enable_evaluation:
                st.session_state.evaluator = RAGASEvaluator(llm_generator=rag.generator)
            
            return True
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return False

def upload_and_index_documents(files):
    """Upload and index documents"""
    if not st.session_state.rag_system:
        st.error("Please initialize RAG system first")
        return
    
    temp_files = []
    try:
        # Save uploaded files temporarily
        for file in files:
            temp_path = Path(f"./temp_{file.name}")
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            temp_files.append(str(temp_path))
            st.session_state.uploaded_files.append(file.name)
        
        # Index documents
        with st.spinner(f"Indexing {len(files)} documents..."):
            st.session_state.rag_system.index_documents(temp_files)
            st.session_state.documents_loaded = True
        
        st.success(f"Successfully indexed {len(files)} documents!")
        
    except Exception as e:
        st.error(f"Error indexing documents: {e}")
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink()
            except:
                pass

def query_rag_system(question: str, use_evaluation: bool = False):
    """Query the RAG system and optionally evaluate"""
    if not st.session_state.rag_system:
        st.error("Please initialize RAG system first")
        return None, None
    
    try:
        with st.spinner("Generating answer..."):
            start_time = time.time()
            
            # Get answer with contexts
            if hasattr(st.session_state.rag_system, 'query_with_contexts'):
                result = st.session_state.rag_system.query_with_contexts(question)
                answer = result['answer']
                contexts = result['contexts']
            else:
                answer = st.session_state.rag_system.query(question)
                contexts = []
            
            query_time = time.time() - start_time
            
            # Evaluate if requested
            ragas_scores = None
            if use_evaluation and st.session_state.evaluator and contexts:
                with st.spinner("Evaluating response..."):
                    ragas_scores = st.session_state.evaluator.evaluate(
                        question=question,
                        answer=answer,
                        contexts=contexts
                    )
            
            # Store in chat history
            st.session_state.chat_history.append({
                'question': question,
                'answer': answer,
                'contexts': contexts,
                'ragas_scores': ragas_scores.to_dict() if ragas_scores else None,
                'query_time': query_time
            })
            
            if ragas_scores:
                st.session_state.evaluation_results.append(ragas_scores.to_dict())
            
            return answer, contexts, ragas_scores, query_time
    
    except Exception as e:
        st.error(f"Error querying RAG system: {e}")
        return None, None, None, None

def display_ragas_metrics(scores: Dict[str, float]):
    """Display RAGAS metrics in a nice format"""
    if not scores:
        return
    
    cols = st.columns(5)
    
    metrics = [
        ("Faithfulness", scores.get('faithfulness', 0)),
        ("Answer Relevancy", scores.get('answer_relevancy', 0)),
        ("Context Relevancy", scores.get('context_relevancy', 0)),
        ("Context Precision", scores.get('context_precision', 0)),
        ("Overall", scores.get('overall', 0))
    ]
    
    for col, (name, value) in zip(cols, metrics):
        with col:
            # Color based on score
            if value >= 0.8:
                color = "🟢"
            elif value >= 0.6:
                color = "🟡"
            else:
                color = "🔴"
            
            st.metric(
                label=name,
                value=f"{value:.3f}",
                delta=f"{color}"
            )

def display_evaluation_dashboard():
    """Display evaluation metrics dashboard"""
    if not st.session_state.evaluation_results:
        st.info("No evaluation results yet. Enable evaluation and make some queries.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.evaluation_results)
    
    # Average metrics
    st.subheader("📊 Average Metrics")
    avg_metrics = df.mean()
    display_ragas_metrics(avg_metrics.to_dict())
    
    # Metrics over time
    st.subheader("📈 Metrics Over Time")
    
    # Create line chart
    fig = go.Figure()
    
    for metric in ['faithfulness', 'answer_relevancy', 'context_relevancy', 'overall']:
        if metric in df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="RAGAS Metrics Trend",
        xaxis_title="Query Number",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution of scores
    st.subheader("📊 Score Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot
        fig_box = go.Figure()
        for metric in ['faithfulness', 'answer_relevancy', 'context_relevancy']:
            if metric in df.columns:
                fig_box.add_trace(go.Box(
                    y=df[metric],
                    name=metric.replace('_', ' ').title(),
                ))
        
        fig_box.update_layout(
            title="Score Distribution by Metric",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Radar chart for latest evaluation
        if len(df) > 0:
            latest = df.iloc[-1]
            
            categories = ['Faithfulness', 'Answer\nRelevancy', 'Context\nRelevancy', 'Context\nPrecision']
            values = [
                latest.get('faithfulness', 0),
                latest.get('answer_relevancy', 0),
                latest.get('context_relevancy', 0),
                latest.get('context_precision', 0)
            ]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=categories + [categories[0]],
                fill='toself',
                name='Latest Query',
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='rgb(102, 126, 234)', width=2)
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Latest Query Performance",
                height=350
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Format DataFrame for display
    display_df = df.copy()
    display_df.index = [f"Query {i+1}" for i in range(len(df))]
    display_df = display_df.round(3)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=300
    )
    
    # Export results
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as JSON
        json_str = json.dumps(st.session_state.evaluation_results, indent=2)
        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name=f"ragas_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"ragas_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Main app
def main():
    st.title("🔍 RAG System Interface")
    st.markdown("Interactive interface for Retrieval-Augmented Generation with RAGAS evaluation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # RAG System Selection
        st.subheader("RAG System Type")
        rag_type = st.selectbox(
            "Select RAG implementation:",
            ["naive", "advanced", "modular"],
            index=0,
            help="Choose between different RAG implementations"
        )
        
        # Evaluation settings
        st.subheader("Evaluation Settings")
        enable_evaluation = st.checkbox(
            "Enable RAGAS Evaluation",
            value=True,
            help="Calculate RAGAS metrics for each query"
        )
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            if initialize_rag_system(rag_type, enable_evaluation):
                st.success(f"✅ {rag_type.upper()} RAG system initialized!")
                st.balloons()
        
        # System status
        st.subheader("System Status")
        if st.session_state.rag_system:
            st.success(f"✅ RAG: {st.session_state.rag_type.upper()}")
            if st.session_state.evaluator:
                st.success("✅ Evaluator: Ready")
            if st.session_state.documents_loaded:
                st.success(f"✅ Documents: {len(st.session_state.uploaded_files)}")
        else:
            st.warning("⚠️ System not initialized")
        
        # Document Management
        st.subheader("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf', 'md', 'docx', 'xlsx'],
            accept_multiple_files=True,
            help="Upload documents to index in the RAG system"
        )
        
        if uploaded_files:
            if st.button("📥 Index Documents", use_container_width=True):
                upload_and_index_documents(uploaded_files)
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("Uploaded Files")
            for file in st.session_state.uploaded_files:
                st.text(f"📄 {file}")
        
        # Clear data
        st.subheader("🗑️ Data Management")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        if st.button("Clear Evaluation Results", use_container_width=True):
            st.session_state.evaluation_results = []
            st.success("Evaluation results cleared!")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Evaluation", "📈 Benchmark", "📚 Documentation"])
    
    with tab1:
        st.header("💬 RAG Chat Interface")
        
        # Check if system is ready
        if not st.session_state.rag_system:
            st.warning("⚠️ Please initialize the RAG system in the sidebar first.")
            st.stop()
        
        if not st.session_state.documents_loaded:
            st.info("💡 Upload and index some documents to get better answers.")
        
        # Query input
        col1, col2 = st.columns([5, 1])
        
        with col1:
            question = st.text_input(
                "Ask a question:",
                placeholder="What is Retrieval-Augmented Generation?",
                key="question_input"
            )
        
        with col2:
            query_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if query_button and question:
            answer, contexts, ragas_scores, query_time = query_rag_system(
                question, 
                use_evaluation=enable_evaluation
            )
            
            if answer:
                # Display answer
                st.subheader("Answer")
                st.success(answer)
                
                # Display metrics
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric("Query Time", f"{query_time:.2f}s")
                with col2:
                    if ragas_scores:
                        display_ragas_metrics(ragas_scores.to_dict())
                
                # Display contexts
                with st.expander("📚 Retrieved Contexts", expanded=False):
                    for i, ctx in enumerate(contexts, 1):
                        st.markdown(f"**Context {i}:**")
                        st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                        st.divider()
        
        # Chat history
        if st.session_state.chat_history:
            st.subheader("📜 Chat History")
            
            for i, item in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {item['question'][:100]}...", expanded=False):
                    st.markdown(f"**Question:** {item['question']}")
                    st.markdown(f"**Answer:** {item['answer']}")
                    st.markdown(f"**Query Time:** {item['query_time']:.2f}s")
                    
                    if item.get('ragas_scores'):
                        st.markdown("**RAGAS Scores:**")
                        scores_df = pd.DataFrame([item['ragas_scores']])
                        st.dataframe(scores_df.round(3))
    
    with tab2:
        st.header("📊 Evaluation Dashboard")
        display_evaluation_dashboard()
    
    with tab3:
        st.header("📈 Benchmark Testing")
        
        if not st.session_state.rag_system:
            st.warning("⚠️ Please initialize the RAG system first.")
            st.stop()
        
        # Benchmark settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dataset = st.selectbox(
                "Dataset",
                ["custom", "squad", "ms_marco"],
                help="Select evaluation dataset"
            )
        
        with col2:
            max_examples = st.number_input(
                "Max Examples",
                min_value=1,
                max_value=100,
                value=10,
                help="Number of examples to evaluate"
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_benchmark = st.button("🏃 Run Benchmark", type="primary")
        
        if run_benchmark:
            with st.spinner(f"Running benchmark on {dataset} dataset..."):
                try:
                    benchmark = RAGBenchmark(
                        st.session_state.rag_system,
                        st.session_state.evaluator
                    )
                    
                    result = benchmark.run_benchmark(
                        dataset=dataset,
                        max_examples=max_examples,
                        save_results=True
                    )
                    
                    # Display results
                    st.success("✅ Benchmark completed!")
                    
                    # Metrics summary
                    st.subheader("📊 Benchmark Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Dataset", result.dataset_name)
                        st.metric("Examples Evaluated", result.num_examples)
                    
                    with col2:
                        st.metric("Token Efficiency", f"{result.token_efficiency:.3f}")
                        st.metric("Avg Latency", f"{result.latency_stats['mean']:.2f}s")
                    
                    # Detailed metrics
                    st.subheader("Detailed Metrics")
                    metrics_df = pd.DataFrame([result.metrics])
                    st.dataframe(metrics_df.round(3), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Benchmark failed: {e}")
    
    with tab4:
        st.header("📚 Documentation")
        
        st.markdown("""
        ## Quick Start Guide
        
        ### 1. Initialize the System
        - Select a RAG type (Naive, Advanced, or Modular) in the sidebar
        - Enable/disable RAGAS evaluation
        - Click "Initialize System"
        
        ### 2. Upload Documents
        - Upload your documents (TXT, PDF, MD, DOCX, XLSX)
        - Click "Index Documents" to process them
        
        ### 3. Ask Questions
        - Type your question in the chat interface
        - Click "Search" to get answers
        - View RAGAS metrics if evaluation is enabled
        
        ### 4. View Evaluation
        - Check the Evaluation tab for metrics dashboard
        - Export results as JSON or CSV
        
        ## RAG Types
        
        ### Naive RAG
        - Basic retrieve-then-read approach
        - Simple and fast
        - Good for general queries
        
        ### Advanced RAG
        - Query optimization (rewriting, expansion)
        - HyDE (Hypothetical Document Embeddings)
        - Better for complex queries
        
        ### Modular RAG
        - Factory pattern architecture
        - Context compression
        - Conversation memory
        - Best for production use
        
        ## RAGAS Metrics
        
        - **Faithfulness**: How grounded the answer is in contexts (0-1)
        - **Answer Relevancy**: How relevant the answer is to the question (0-1)
        - **Context Relevancy**: How relevant retrieved contexts are (0-1)
        - **Context Precision**: Ranking quality of contexts (0-1)
        - **Overall**: Average of all metrics (0-1)
        
        ## Tips for Better Results
        
        1. **Upload relevant documents**: The system can only answer based on indexed documents
        2. **Be specific**: Clear, specific questions get better answers
        3. **Check metrics**: Use RAGAS scores to understand answer quality
        4. **Try different RAG types**: Each has strengths for different queries
        5. **Review contexts**: Check retrieved contexts to understand the answer source
        
        ## Troubleshooting
        
        - **No answer returned**: Check if documents are indexed
        - **Low RAGAS scores**: Try rephrasing the question or using Advanced RAG
        - **Slow performance**: Reduce the number of retrieved contexts
        - **Error messages**: Check the system status in the sidebar
        """)

if __name__ == "__main__":
    main()