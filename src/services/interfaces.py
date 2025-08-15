"""
Service interfaces for Production RAG System using Python Protocol
"""

from typing import Protocol, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import abstractmethod

@dataclass 
class QueryOptimizationResult:
    """Result of query optimization"""
    original_query: str
    optimized_query: str
    hyde_query: Optional[str]
    sub_queries: List[str]
    intent: Optional[str]
    complexity: Optional[float]
    entities: List[str]

@dataclass
class RetrievalResult:
    """Result of document retrieval"""
    contexts: List[str]
    scores: List[float]
    metadata: List[Dict[str, Any]]
    graph_entities_found: int
    graph_relations_used: int
    compression_ratio: Optional[float]
    contexts_compressed: bool

@dataclass
class GenerationResult:
    """Result of response generation"""
    answer: str
    tokens_generated: int
    generation_time_ms: float

@dataclass
class MemoryContext:
    """Context from conversation memory"""
    content: str
    relevance_score: float
    metadata: Dict[str, Any]

class QueryServiceInterface(Protocol):
    """Interface for query optimization service"""
    
    @abstractmethod
    def optimize_query(
        self,
        query: str,
        enable_decomposition: bool = True,
        enable_hyde: bool = True
    ) -> QueryOptimizationResult:
        """Optimize query with various techniques"""
        ...

class RetrievalServiceInterface(Protocol):
    """Interface for document retrieval service"""
    
    @abstractmethod
    def retrieve_contexts(
        self,
        query: str,
        sub_queries: List[str],
        entities: List[str],
        conversation_contexts: List[MemoryContext],
        retrieval_method: str = "adaptive",
        top_k: int = 10,
        enable_compression: bool = True,
        max_context_tokens: int = 2000
    ) -> RetrievalResult:
        """Retrieve and process contexts from multiple sources"""
        ...
    
    @abstractmethod
    def index_document(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
        doc_type: Optional[str] = None
    ) -> None:
        """Index a document"""
        ...

class GenerationServiceInterface(Protocol):
    """Interface for response generation service"""
    
    @abstractmethod
    def generate_response(
        self,
        query: str,
        contexts: List[str],
        conversation_history: Optional[List[Dict]] = None
    ) -> GenerationResult:
        """Generate response based on contexts"""
        ...

class MemoryServiceInterface(Protocol):
    """Interface for conversation memory service"""
    
    @abstractmethod
    def get_relevant_context(
        self,
        query: str,
        k: int = 3
    ) -> List[MemoryContext]:
        """Get relevant context from conversation memory"""
        ...
    
    @abstractmethod
    def resolve_references(self, query: str) -> str:
        """Resolve references in query"""
        ...
    
    @abstractmethod
    def add_turn(
        self,
        query: str,
        response: str,
        contexts: List[str],
        relevance_scores: List[float]
    ) -> None:
        """Add conversation turn to memory"""
        ...
    
    @abstractmethod
    def start_session(self, session_id: str) -> None:
        """Start new conversation session"""
        ...
    
    @abstractmethod
    def end_session(self) -> None:
        """End current conversation session"""
        ...

class MonitoringServiceInterface(Protocol):
    """Interface for monitoring and metrics service"""
    
    @abstractmethod
    def track_request(
        self,
        method: str,
        duration_ms: float,
        status: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Track request metrics"""
        ...
    
    @abstractmethod
    def track_retrieval(
        self,
        retriever_type: str,
        latency_ms: float,
        contexts_count: int
    ) -> None:
        """Track retrieval metrics"""
        ...
    
    @abstractmethod
    def track_generation(
        self,
        tokens: int,
        latency_ms: float,
        model: str
    ) -> None:
        """Track generation metrics"""
        ...
    
    @abstractmethod
    def track_error(
        self,
        error_type: str,
        component: str,
        message: str
    ) -> None:
        """Track error"""
        ...
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        ...
    
    @abstractmethod
    def register_health_check(self, name: str, check_func) -> None:
        """Register health check function"""
        ...