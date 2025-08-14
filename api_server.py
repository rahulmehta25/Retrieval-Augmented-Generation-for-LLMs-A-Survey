from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncIterator
import os
import tempfile
import shutil
from datetime import datetime
import uuid
import logging
import json
from pathlib import Path
import asyncio
from sse_starlette.sse import EventSourceResponse

from src.rag.naive_rag import NaiveRAG
from src.rag.advanced_rag import AdvancedRAG
from src.rag.modular_rag import ModularRAG
from src.evaluation.ragas_metrics import RAGASEvaluator
from src.evaluation.benchmark import RAGBenchmark
from src.evaluation.human_eval import HumanEvaluationInterface, EvaluationBatch
from src.streaming.stream_handler import StreamingRAG, StreamEvent, StreamEventType
from src.graph_rag.knowledge_graph import GraphRAG
from src.graph_rag.multi_hop import MultiHopReasoning
from src.advanced_rag.self_rag import SelfRAG
from src.optimization.query_router import QueryRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Knowledge API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system with evaluation enabled
try:
    rag_system = NaiveRAG(config_path='config.yaml', enable_evaluation=True)
    logger.info("RAG system initialized successfully with evaluation enabled")
except Exception as e:
    logger.error(f"Error initializing RAG system: {e}")
    # Create a minimal config if not found
    import yaml
    minimal_config = {
        'text_splitter': {'type': 'fixed_size', 'chunk_size': 500, 'chunk_overlap': 50},
        'embedder': {'type': 'sentence_transformer', 'model_name': 'all-MiniLM-L6-v2'},
        'vector_store': {'type': 'chromadb', 'path': './chroma_db', 'collection_name': 'rag_collection'},
        'generator': {'type': 'ollama', 'model_name': 'gemma:2b', 'host': 'localhost', 'port': 11434}
    }
    with open('config.yaml', 'w') as f:
        yaml.dump(minimal_config, f)
    rag_system = NaiveRAG(config_path='config.yaml', enable_evaluation=True)

# Initialize evaluation components
evaluator = RAGASEvaluator(llm_generator=rag_system.generator)
benchmark = RAGBenchmark(rag_system, evaluator)
human_eval = HumanEvaluationInterface(app)

# Initialize streaming handler
streaming_rag = StreamingRAG(rag_system, llm_type="ollama")

# Initialize advanced RAG systems
try:
    advanced_rag = AdvancedRAG(config_path='config.yaml')
    modular_rag = ModularRAG(config_path='config.yaml')
    graph_rag = GraphRAG()
    multi_hop = MultiHopReasoning(graph_rag, rag_system.generator)
    self_rag = SelfRAG(rag_system, rag_system.generator)
    
    # Initialize query router with all RAG strategies
    rag_strategies = {
        'naive': rag_system,
        'advanced': advanced_rag,
        'modular': modular_rag,
        'graph': graph_rag,
        'self': self_rag
    }
    query_router = QueryRouter(rag_strategies)
    logger.info("All RAG systems initialized successfully")
except Exception as e:
    logger.warning(f"Some advanced RAG systems could not be initialized: {e}")
    advanced_rag = None
    graph_rag = None
    query_router = None

# In-memory storage for documents and authentication
documents_db: Dict[str, Dict] = {}
users_db: Dict[str, Dict] = {
    "demo": {"username": "demo", "password": "password123", "token": None}
}
uploaded_files_dir = Path("./uploaded_documents")
uploaded_files_dir.mkdir(exist_ok=True)

# Load existing documents on startup
def load_existing_documents():
    """Load metadata for documents already in uploaded_documents directory"""
    logger.info("Loading existing documents...")
    for file_path in uploaded_files_dir.iterdir():
        if file_path.is_file():
            file_id = file_path.name.split('_')[0]  # Extract UUID from filename
            if len(file_id) == 36:  # Valid UUID length
                documents_db[file_id] = {
                    "id": file_id,
                    "name": file_path.name[37:],  # Remove UUID prefix
                    "chunks": 10,  # Default, will be updated when queried
                    "upload_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "file_path": str(file_path)
                }
    logger.info(f"Loaded {len(documents_db)} existing documents")

# Load existing documents on startup
load_existing_documents()

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    stream: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class Document(BaseModel):
    id: str
    name: str
    chunks: int
    upload_date: str

class DocumentsResponse(BaseModel):
    documents: List[Document]

class SystemStatus(BaseModel):
    model: str
    documents_loaded: int
    ready: bool

# Authentication endpoints
@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    user = users_db.get(request.username)
    if user and user["password"] == request.password:
        token = f"token_{uuid.uuid4()}"
        user["token"] = token
        return AuthResponse(success=True, token=token, message="Login successful")
    return AuthResponse(success=False, message="Invalid credentials")

@app.post("/api/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    if request.username in users_db:
        return AuthResponse(success=False, message="Username already exists")
    
    if len(request.username) < 3 or len(request.password) < 6:
        return AuthResponse(
            success=False, 
            message="Username must be 3+ chars, password must be 6+ chars"
        )
    
    token = f"token_{uuid.uuid4()}"
    users_db[request.username] = {
        "username": request.username,
        "password": request.password,
        "token": token
    }
    return AuthResponse(success=True, token=token, message="Registration successful")

# Document management endpoints
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Uploading document: {file.filename}")
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = uploaded_files_dir / f"{file_id}_{file.filename}"
        
        logger.info(f"Saving file to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Index the document
        logger.info(f"Indexing document: {file_path}")
        rag_system.index_documents([str(file_path)])
        logger.info("Document indexed successfully")
        
        # Get actual chunk count from vector store
        chunk_count = 10  # Default
        try:
            # Query to get document info
            test_contexts = rag_system.retrieve("test", k=1, filters={"source": str(file_path)})
            if test_contexts:
                # Count actual chunks by querying with high k
                all_contexts = rag_system.retrieve("", k=1000, filters={"source": str(file_path)})
                chunk_count = len(all_contexts)
        except:
            pass
        
        # Store document metadata
        documents_db[file_id] = {
            "id": file_id,
            "name": file.filename,
            "chunks": chunk_count,
            "upload_date": datetime.now().isoformat(),
            "file_path": str(file_path)
        }
        
        logger.info(f"Document stored with {chunk_count} chunks")
        
        return {
            "status": "success",
            "message": f"Processed {file.filename}",
            "document_count": len(documents_db)
        }
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents", response_model=DocumentsResponse)
async def get_documents():
    documents = [
        Document(**doc_data)
        for doc_data in documents_db.values()
    ]
    return DocumentsResponse(documents=documents)

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file from disk
    doc = documents_db[document_id]
    if "file_path" in doc and os.path.exists(doc["file_path"]):
        os.remove(doc["file_path"])
    
    # Remove from database
    del documents_db[document_id]
    
    return {"status": "success"}

# Chat/Query endpoint
@app.post("/api/chat/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    logger.info(f"Received query: {request.question}")
    try:
        # Check if we have any documents indexed
        if len(documents_db) == 0:
            logger.warning("No documents in database")
            return QueryResponse(
                answer="Please upload some documents first. I need documents in my knowledge base to answer your questions.",
                sources=[]
            )
        
        logger.info(f"Documents in DB: {len(documents_db)}")
        
        # Query the RAG system
        logger.info("Calling RAG system query...")
        answer = rag_system.query(request.question, k=5)
        logger.info(f"RAG answer: {answer[:100]}...")
        
        # Get sources from retrieved contexts
        sources = []
        try:
            retrieved_contexts = rag_system.retrieve(request.question, k=5)
            logger.info(f"Retrieved {len(retrieved_contexts)} contexts")
            for ctx in retrieved_contexts:
                logger.debug(f"Context: {ctx}")
                if 'metadata' in ctx and 'source' in ctx['metadata']:
                    source = os.path.basename(ctx['metadata']['source'])
                    chunk_id = ctx.get('id', 'unknown')
                    sources.append(f"{source}:chunk{chunk_id}")
        except Exception as e:
            logger.warning(f"Could not retrieve sources: {e}")
        
        logger.info(f"Returning answer with {len(sources)} sources")
        return QueryResponse(answer=answer, sources=sources[:2])  # Return top 2 sources
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        # Return a user-friendly error message
        return QueryResponse(
            answer="I'm sorry, I encountered an error processing your question. Please make sure you have uploaded documents and try again.",
            sources=[]
        )

# Streaming chat endpoint
@app.get("/api/chat/stream")
async def stream_query(question: str):
    """
    Stream RAG responses using Server-Sent Events
    
    Args:
        question: User query
    
    Returns:
        Server-Sent Events stream
    """
    async def event_generator():
        try:
            # Check if we have documents
            if len(documents_db) == 0:
                event = StreamEvent(
                    type=StreamEventType.ERROR,
                    content="Please upload some documents first."
                )
                yield {
                    "event": "error",
                    "data": json.dumps(event.to_dict())
                }
                return
            
            # Start retrieval
            event = StreamEvent(type=StreamEventType.RETRIEVAL_START)
            yield {
                "event": "retrieval_start",
                "data": json.dumps(event.to_dict())
            }
            
            # Perform retrieval
            contexts = rag_system.retrieve(question, k=5)
            
            # Send retrieval complete event
            event = StreamEvent(
                type=StreamEventType.RETRIEVAL_COMPLETE,
                contexts=contexts[:2],  # Send top 2 contexts as sources
                metadata={"num_contexts": len(contexts)}
            )
            yield {
                "event": "retrieval_complete",
                "data": json.dumps(event.to_dict())
            }
            
            # Start generation
            event = StreamEvent(type=StreamEventType.GENERATION_START)
            yield {
                "event": "generation_start",
                "data": json.dumps(event.to_dict())
            }
            
            # Stream tokens
            async for token in streaming_rag.stream_generate(question, contexts):
                event = StreamEvent(
                    type=StreamEventType.TOKEN,
                    content=token
                )
                yield {
                    "event": "token",
                    "data": json.dumps(event.to_dict())
                }
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
            
            # Send complete event
            event = StreamEvent(type=StreamEventType.COMPLETE)
            yield {
                "event": "complete",
                "data": json.dumps(event.to_dict())
            }
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            event = StreamEvent(
                type=StreamEventType.ERROR,
                content=str(e)
            )
            yield {
                "event": "error",
                "data": json.dumps(event.to_dict())
            }
    
    return EventSourceResponse(event_generator())

# Advanced RAG endpoints
@app.post("/api/chat/query/advanced")
async def query_advanced(request: QueryRequest):
    """Query using Advanced RAG with query optimization"""
    if not advanced_rag:
        raise HTTPException(status_code=503, detail="Advanced RAG not available")
    
    try:
        answer = advanced_rag.query_optimized(request.question)
        contexts = advanced_rag.last_contexts if hasattr(advanced_rag, 'last_contexts') else []
        return QueryResponse(answer=answer, sources=contexts[:2])
    except Exception as e:
        logger.error(f"Advanced RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/query/graph")
async def query_graph(request: QueryRequest):
    """Query using Graph RAG for knowledge graph-based retrieval"""
    if not graph_rag:
        raise HTTPException(status_code=503, detail="Graph RAG not available")
    
    try:
        contexts = graph_rag.graph_retrieve(request.question)
        answer = rag_system.generator.generate(
            request.question,
            "\n".join(contexts) if contexts else ""
        )
        return QueryResponse(answer=answer, sources=contexts[:2])
    except Exception as e:
        logger.error(f"Graph RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/query/multihop")
async def query_multihop(request: QueryRequest):
    """Query using multi-hop reasoning"""
    if not multi_hop:
        raise HTTPException(status_code=503, detail="Multi-hop reasoning not available")
    
    try:
        result = await multi_hop.multi_hop_query(request.question)
        return QueryResponse(
            answer=result.get('answer', ''),
            sources=result.get('reasoning_chain', [])[:2]
        )
    except Exception as e:
        logger.error(f"Multi-hop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/query/self")
async def query_self_rag(request: QueryRequest):
    """Query using Self-RAG with reflection"""
    if not self_rag:
        raise HTTPException(status_code=503, detail="Self-RAG not available")
    
    try:
        result = await self_rag.query_with_reflection(request.question)
        return QueryResponse(
            answer=result.get('answer', ''),
            sources=result.get('contexts', [])[:2]
        )
    except Exception as e:
        logger.error(f"Self-RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/query/auto")
async def query_auto_route(request: QueryRequest):
    """Automatically route query to best RAG strategy"""
    if not query_router:
        # Fallback to naive RAG
        return await query_rag(request)
    
    try:
        result = await query_router.route_query(request.question)
        return QueryResponse(
            answer=result.get('answer', ''),
            sources=result.get('sources', [])[:2]
        )
    except Exception as e:
        logger.error(f"Query routing error: {e}")
        # Fallback to naive RAG
        return await query_rag(request)

# System status endpoint
@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    return SystemStatus(
        model="gemma:2b",  # From config
        documents_loaded=len(documents_db),
        ready=True
    )

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

# ============== RAGAS Evaluation Endpoints ==============

class EvaluationRequest(BaseModel):
    question: str
    ground_truth: Optional[str] = None

class EvaluationResponse(BaseModel):
    answer: str
    contexts: List[str]
    ragas_scores: Optional[Dict[str, float]] = None

@app.post("/api/evaluate/query", response_model=EvaluationResponse)
async def evaluate_query(request: EvaluationRequest):
    """Evaluate a single query with RAGAS metrics"""
    try:
        # Get answer with contexts
        result = rag_system.query_with_contexts(request.question)
        answer = result['answer']
        contexts = result['contexts']
        
        # Evaluate with RAGAS
        ragas_score = None
        if contexts:
            ragas_score = evaluator.evaluate(
                question=request.question,
                answer=answer,
                contexts=contexts,
                ground_truth=request.ground_truth
            )
            
        return EvaluationResponse(
            answer=answer,
            contexts=contexts,
            ragas_scores=ragas_score.to_dict() if ragas_score else None
        )
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class BenchmarkRequest(BaseModel):
    dataset: str = "custom"
    max_examples: int = 10
    save_results: bool = True

class BenchmarkResponse(BaseModel):
    dataset_name: str
    num_examples: int
    metrics: Dict[str, float]
    latency_stats: Dict[str, float]
    token_efficiency: float

@app.post("/api/benchmark/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest):
    """Run comprehensive benchmark on a dataset"""
    try:
        result = benchmark.run_benchmark(
            dataset=request.dataset,
            max_examples=request.max_examples,
            save_results=request.save_results
        )
        
        return BenchmarkResponse(
            dataset_name=result.dataset_name,
            num_examples=result.num_examples,
            metrics=result.metrics,
            latency_stats=result.latency_stats,
            token_efficiency=result.token_efficiency
        )
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmark/results")
async def get_benchmark_results():
    """Get all benchmark results"""
    try:
        results_dir = Path("benchmark_results")
        if not results_dir.exists():
            return {"results": []}
        
        results = []
        for file_path in results_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                results.append(json.load(f))
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error loading benchmark results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class BatchEvaluationRequest(BaseModel):
    examples: List[Dict[str, Any]]

@app.post("/api/evaluate/batch")
async def evaluate_batch(request: BatchEvaluationRequest):
    """Evaluate multiple examples and get aggregated metrics"""
    try:
        all_results = []
        
        for example in request.examples:
            result = rag_system.query_with_contexts(example['question'])
            
            # Evaluate each example
            if result['contexts']:
                ragas_score = evaluator.evaluate(
                    question=example['question'],
                    answer=result['answer'],
                    contexts=result['contexts'],
                    ground_truth=example.get('ground_truth')
                )
                
                all_results.append({
                    'question': example['question'],
                    'answer': result['answer'],
                    'contexts': result['contexts'],
                    'ragas_scores': ragas_score.to_dict()
                })
        
        # Aggregate metrics
        aggregated = evaluator.batch_evaluate([
            {
                'question': r['question'],
                'answer': r['answer'],
                'contexts': r['contexts'],
                'ground_truth': example.get('ground_truth')
            }
            for r, example in zip(all_results, request.examples)
        ])
        
        return {
            'individual_results': all_results,
            'aggregated_metrics': aggregated
        }
    except Exception as e:
        logger.error(f"Error in batch evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Human evaluation endpoints are handled by HumanEvaluationInterface
# which adds its own routes to the FastAPI app

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAG API server on http://localhost:8090")
    logger.info("Visit http://localhost:8090/docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8090)