"""
Shared fixtures and configuration for RAG system tests
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.services.interfaces import (
    QueryOptimizationResult,
    RetrievalResult,
    GenerationResult,
    MemoryContext
)


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_query():
    """Sample user query for testing"""
    return "What is machine learning and how does it work?"


@pytest.fixture
def sample_queries():
    """Multiple sample queries for testing"""
    return [
        "What is machine learning?",
        "How does neural networks work?",
        "Explain deep learning concepts",
        "What are the benefits of AI?",
        "Tell me about natural language processing"
    ]


@pytest.fixture
def sample_contexts():
    """Sample document contexts for testing"""
    return [
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.",
        "Deep learning is a machine learning technique that uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing (NLP) is a branch of AI that helps computers understand and interpret human language."
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return [
        {"source": "doc1.txt", "page": 1, "section": "introduction"},
        {"source": "doc2.txt", "page": 2, "section": "concepts"},
        {"source": "doc3.txt", "page": 1, "section": "theory"},
        {"source": "doc4.txt", "page": 3, "section": "applications"}
    ]


@pytest.fixture
def sample_scores():
    """Sample relevance scores for testing"""
    return [0.95, 0.87, 0.82, 0.76]


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing"""
    return [
        {"query": "What is AI?", "response": "AI is artificial intelligence."},
        {"query": "How does it work?", "response": "AI works by processing data through algorithms."},
        {"query": "What are the applications?", "response": "AI has many applications including ML, NLP, and computer vision."}
    ]


# Service Interface Mocks

@pytest.fixture
def mock_query_optimizer():
    """Mock semantic query optimizer"""
    mock = Mock()
    
    # Mock optimized query result
    mock_result = Mock()
    mock_result.cleaned = "optimized query"
    mock_result.intent = "explanation"
    mock_result.complexity = 0.8
    mock_result.keywords = ["machine", "learning", "AI"]
    mock_result.sub_queries = ["What is machine learning?", "How does ML work?"]
    
    mock.optimize.return_value = mock_result
    return mock


@pytest.fixture
def mock_query_rewriter():
    """Mock query rewriter for HyDE"""
    mock = Mock()
    mock.generate_hypothetical_document.return_value = "Machine learning is a method of data analysis that automates analytical model building."
    return mock


@pytest.fixture
def mock_hybrid_retriever():
    """Mock hybrid retriever"""
    mock = Mock()
    
    # Mock retrieval results
    @dataclass
    class MockRetrievalItem:
        text: str
        score: float
        metadata: Dict[str, Any] = None
    
    mock_results = [
        MockRetrievalItem("First context", 0.95, {"source": "doc1"}),
        MockRetrievalItem("Second context", 0.87, {"source": "doc2"}),
        MockRetrievalItem("Third context", 0.82, {"source": "doc3"})
    ]
    
    mock.adaptive_retrieve.return_value = mock_results
    mock.hybrid_retrieve.return_value = mock_results
    mock.sparse_retrieval.return_value = mock_results
    mock.dense_retrieval.return_value = mock_results
    mock.add_documents.return_value = None
    
    return mock


@pytest.fixture
def mock_context_compressor():
    """Mock context compressor"""
    mock = Mock()
    
    # Mock compression result
    mock_result = Mock()
    mock_result.compressed_text = "Compressed context text"
    mock_result.compression_ratio = 0.65
    
    mock.adaptive_compression.return_value = mock_result
    return mock


@pytest.fixture
def mock_semantic_chunker():
    """Mock semantic chunker"""
    mock = Mock()
    
    # Mock chunk result
    @dataclass
    class MockChunk:
        text: str
        metadata: Dict[str, Any] = None
    
    mock_chunks = [
        MockChunk("First chunk of text"),
        MockChunk("Second chunk of text"),
        MockChunk("Third chunk of text")
    ]
    
    mock.smart_chunk.return_value = mock_chunks
    return mock


@pytest.fixture
def mock_knowledge_graph():
    """Mock knowledge graph"""
    mock = Mock()
    
    # Mock graph query results
    mock_graph_results = {
        'entities': [
            {'type': 'concept', 'text': 'machine learning concept'},
            {'type': 'definition', 'text': 'ML definition'}
        ],
        'relations': [
            {'source': 'AI', 'target': 'ML', 'relation': 'includes'}
        ]
    }
    
    mock.query_graph.return_value = mock_graph_results
    mock.add_document.return_value = None
    mock.get_statistics.return_value = {'total_entities': 100, 'total_relations': 50}
    
    return mock


@pytest.fixture
def mock_llm_generator():
    """Mock LLM generator"""
    mock = Mock()
    mock.generate.return_value = "This is a generated response based on the provided context."
    return mock


@pytest.fixture
def mock_conversation_memory():
    """Mock conversation memory"""
    mock = Mock()
    
    # Mock memory contexts
    mock_contexts = [
        {
            'content': 'Previous conversation context',
            'relevance_score': 0.8,
            'timestamp': '2024-01-01T10:00:00',
            'turn_id': 1,
            'session_id': 'test_session'
        }
    ]
    
    mock.get_relevant_context.return_value = mock_contexts
    mock.resolve_references.return_value = "resolved query"
    mock.add_turn.return_value = None
    mock.start_session.return_value = None
    mock.end_session.return_value = None
    mock.save_memory.return_value = None
    
    # Mock current session
    mock_session = Mock()
    mock_session.session_id = "test_session"
    mock_session.start_time = "2024-01-01T10:00:00"
    mock_session.turns = []
    mock.current_session = mock_session
    
    return mock


@pytest.fixture
def mock_production_monitoring():
    """Mock production monitoring"""
    mock = Mock()
    
    mock.track_request.return_value = None
    mock.track_retrieval.return_value = None
    mock.track_generation.return_value = None
    mock.track_error.return_value = None
    mock.create_alert.return_value = None
    mock.register_health_check.return_value = None
    mock.export_metrics.return_value = None
    
    # Mock health checks
    mock.health_checks = {
        'database': lambda: (True, "Database is healthy"),
        'memory': lambda: (True, "Memory usage is normal")
    }
    
    # Mock metrics summary
    mock.get_metrics_summary.return_value = {
        'requests_total': 100,
        'avg_latency': 250.5,
        'error_rate': 0.02,
        'uptime_seconds': 3600
    }
    
    return mock


@pytest.fixture
def mock_ab_testing():
    """Mock A/B testing framework"""
    mock = Mock()
    
    # Mock variant
    mock_variant = Mock()
    mock_variant.name = "variant_a"
    mock_variant.config = {"temperature": 0.1}
    
    mock.get_variant.return_value = mock_variant
    mock.record_event.return_value = None
    
    return mock


@pytest.fixture
def mock_ragas_evaluator():
    """Mock RAGAS evaluator"""
    mock = Mock()
    
    # Mock evaluation result
    mock_result = Mock()
    mock_result.faithfulness = 0.85
    mock_result.answer_relevancy = 0.90
    mock_result.context_relevancy = 0.88
    mock_result.context_precision = 0.82
    
    mock.evaluate.return_value = mock_result
    return mock


# Fixture for sample results

@pytest.fixture
def sample_query_optimization_result():
    """Sample QueryOptimizationResult for testing"""
    return QueryOptimizationResult(
        original_query="What is machine learning?",
        optimized_query="machine learning definition concepts",
        hyde_query="Machine learning is a subset of AI that enables computers to learn patterns from data.",
        sub_queries=["What is machine learning?", "How does machine learning work?"],
        intent="definition",
        complexity=0.6,
        entities=["machine learning", "AI", "algorithms"]
    )


@pytest.fixture
def sample_retrieval_result():
    """Sample RetrievalResult for testing"""
    return RetrievalResult(
        contexts=[
            "Machine learning is a subset of artificial intelligence.",
            "ML algorithms learn patterns from data without explicit programming.",
            "Common ML techniques include supervised and unsupervised learning."
        ],
        scores=[0.95, 0.87, 0.82],
        metadata=[
            {"source": "ml_intro.txt", "chunk_id": 0},
            {"source": "ml_concepts.txt", "chunk_id": 1},
            {"source": "ml_techniques.txt", "chunk_id": 2}
        ],
        graph_entities_found=5,
        graph_relations_used=3,
        compression_ratio=0.7,
        contexts_compressed=True
    )


@pytest.fixture
def sample_generation_result():
    """Sample GenerationResult for testing"""
    return GenerationResult(
        answer="Machine learning is a powerful subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for each task.",
        tokens_generated=25,
        generation_time_ms=150.5
    )


@pytest.fixture
def sample_memory_contexts():
    """Sample MemoryContext objects for testing"""
    return [
        MemoryContext(
            content="Previous discussion about AI fundamentals",
            relevance_score=0.85,
            metadata={
                "timestamp": "2024-01-01T10:00:00",
                "turn_id": 1,
                "session_id": "test_session"
            }
        ),
        MemoryContext(
            content="Earlier conversation about machine learning applications",
            relevance_score=0.78,
            metadata={
                "timestamp": "2024-01-01T10:05:00",
                "turn_id": 2,
                "session_id": "test_session"
            }
        )
    ]


# Parametrized fixtures for testing different scenarios

@pytest.fixture(params=[
    "simple_query",
    "complex_multi_part_query",
    "question_with_entities",
    "follow_up_query"
])
def query_scenarios(request):
    """Different query scenarios for comprehensive testing"""
    scenarios = {
        "simple_query": "What is AI?",
        "complex_multi_part_query": "What is machine learning, how does it differ from traditional programming, and what are its main applications in healthcare?",
        "question_with_entities": "How does Tesla use artificial intelligence in their autonomous vehicles?",
        "follow_up_query": "Can you explain more about that?"
    }
    return scenarios[request.param]


@pytest.fixture(params=[
    "success",
    "partial_failure", 
    "complete_failure"
])
def error_scenarios(request):
    """Different error scenarios for testing error handling"""
    return request.param


# Performance testing fixtures

@pytest.fixture
def performance_config():
    """Configuration for performance testing"""
    return {
        "max_latency_ms": 5000,
        "min_throughput_qps": 10,
        "memory_limit_mb": 1024,
        "cpu_limit_percent": 80
    }


# Integration test fixtures

@pytest.fixture
def integration_config():
    """Configuration for integration testing"""
    return {
        "enable_query_optimization": True,
        "enable_query_decomposition": True,
        "enable_hyde": True,
        "retrieval_method": "adaptive",
        "top_k": 5,
        "enable_compression": True,
        "max_context_tokens": 1000,
        "use_reranking": True,
        "use_mmr": True,
        "mmr_lambda": 0.5,
        "max_graph_hops": 2
    }


# Test data validation

@pytest.fixture
def validate_test_data():
    """Fixture to validate test data consistency"""
    def _validate(contexts: List[str], scores: List[float], metadata: List[Dict]):
        assert len(contexts) == len(scores) == len(metadata), "Data length mismatch"
        assert all(isinstance(ctx, str) for ctx in contexts), "Contexts must be strings"
        assert all(0 <= score <= 1 for score in scores), "Scores must be between 0 and 1"
        assert all(isinstance(meta, dict) for meta in metadata), "Metadata must be dictionaries"
        return True
    
    return _validate


# Cleanup fixtures

@pytest.fixture(autouse=True)
def cleanup_logs():
    """Automatically cleanup test logs after each test"""
    yield
    # Cleanup logic could go here if needed
    pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging configuration for tests"""
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress noisy loggers during testing
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)