"""
Human Evaluation Interface for RAG System

Provides a web interface for human evaluation of RAG outputs.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationItem(BaseModel):
    """Model for evaluation items"""
    id: str
    question: str
    answer: str
    contexts: List[str]
    metadata: Optional[Dict[str, Any]] = None


class HumanEvaluation(BaseModel):
    """Model for human evaluation scores"""
    item_id: str
    relevance_score: float  # 0-5
    accuracy_score: float  # 0-5
    completeness_score: float  # 0-5
    fluency_score: float  # 0-5
    overall_score: float  # 0-5
    comments: Optional[str] = None
    evaluator_id: Optional[str] = None
    timestamp: Optional[str] = None


class EvaluationBatch(BaseModel):
    """Model for batch evaluation request"""
    items: List[Dict[str, Any]]
    batch_name: Optional[str] = None


class HumanEvaluationInterface:
    """Web interface for human evaluation of RAG outputs"""
    
    def __init__(self, app: Optional[FastAPI] = None):
        """
        Initialize human evaluation interface
        
        Args:
            app: FastAPI app instance (creates new if None)
        """
        self.app = app or FastAPI(title="RAG Human Evaluation Interface")
        self.evaluation_queue = asyncio.Queue()
        self.results = []
        self.active_connections = []
        self.evaluation_sessions = {}
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def get_interface():
            """Serve the evaluation interface"""
            return HTMLResponse(self._get_html_interface())
        
        @self.app.websocket("/ws/evaluate")
        async def evaluate_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time evaluation"""
            await websocket.accept()
            connection_id = str(uuid.uuid4())
            self.active_connections.append(websocket)
            
            try:
                # Create session for this evaluator
                self.evaluation_sessions[connection_id] = {
                    'start_time': datetime.now(),
                    'evaluations': []
                }
                
                # Send initial message
                await websocket.send_json({
                    "type": "connected",
                    "connection_id": connection_id,
                    "message": "Connected to evaluation server"
                })
                
                while True:
                    # Check if there are items to evaluate
                    if not self.evaluation_queue.empty():
                        eval_item = await self.evaluation_queue.get()
                        
                        # Send item for evaluation
                        await websocket.send_json({
                            "type": "evaluate",
                            "item": eval_item
                        })
                        
                        # Receive evaluation
                        evaluation_data = await websocket.receive_json()
                        
                        # Process evaluation
                        evaluation = HumanEvaluation(
                            item_id=eval_item['id'],
                            evaluator_id=connection_id,
                            timestamp=datetime.now().isoformat(),
                            **evaluation_data
                        )
                        
                        # Store result
                        self.results.append(evaluation.dict())
                        self.evaluation_sessions[connection_id]['evaluations'].append(evaluation.dict())
                        
                        # Send confirmation
                        await websocket.send_json({
                            "type": "evaluation_received",
                            "item_id": eval_item['id'],
                            "message": "Evaluation saved successfully"
                        })
                    else:
                        # No items to evaluate, wait
                        await websocket.send_json({
                            "type": "waiting",
                            "message": "Waiting for items to evaluate..."
                        })
                        await asyncio.sleep(5)
            
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
                if connection_id in self.evaluation_sessions:
                    session = self.evaluation_sessions[connection_id]
                    logger.info(f"Evaluator {connection_id} disconnected. "
                              f"Completed {len(session['evaluations'])} evaluations")
            
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
        
        @self.app.post("/evaluation/batch")
        async def submit_batch(batch: EvaluationBatch):
            """Submit batch for human evaluation"""
            batch_id = str(uuid.uuid4())
            
            # Add items to queue with batch metadata
            for item in batch.items:
                eval_item = {
                    'id': str(uuid.uuid4()),
                    'batch_id': batch_id,
                    'batch_name': batch.batch_name,
                    **item
                }
                await self.evaluation_queue.put(eval_item)
            
            return {
                "status": "queued",
                "batch_id": batch_id,
                "count": len(batch.items),
                "queue_size": self.evaluation_queue.qsize()
            }
        
        @self.app.get("/evaluation/results")
        async def get_results(batch_id: Optional[str] = None):
            """Get evaluation results"""
            if batch_id:
                # Filter results by batch_id
                batch_results = [r for r in self.results 
                                if r.get('batch_id') == batch_id]
                return {
                    "batch_id": batch_id,
                    "count": len(batch_results),
                    "results": batch_results
                }
            else:
                return {
                    "total_count": len(self.results),
                    "results": self.results
                }
        
        @self.app.get("/evaluation/stats")
        async def get_stats():
            """Get evaluation statistics"""
            if not self.results:
                return {"message": "No evaluations completed yet"}
            
            # Calculate aggregate statistics
            stats = {
                'total_evaluations': len(self.results),
                'active_evaluators': len(self.active_connections),
                'queue_size': self.evaluation_queue.qsize(),
                'average_scores': {}
            }
            
            # Calculate average scores
            score_fields = ['relevance_score', 'accuracy_score', 
                          'completeness_score', 'fluency_score', 'overall_score']
            
            for field in score_fields:
                scores = [r[field] for r in self.results if field in r]
                if scores:
                    stats['average_scores'][field] = sum(scores) / len(scores)
            
            # Calculate evaluator statistics
            evaluator_stats = {}
            for result in self.results:
                evaluator_id = result.get('evaluator_id', 'unknown')
                if evaluator_id not in evaluator_stats:
                    evaluator_stats[evaluator_id] = 0
                evaluator_stats[evaluator_id] += 1
            
            stats['evaluator_contributions'] = evaluator_stats
            
            return stats
        
        @self.app.post("/evaluation/export")
        async def export_results(format: str = "json"):
            """Export evaluation results"""
            if format == "json":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"human_eval_results_{timestamp}.json"
                
                # Save to file
                output_dir = Path("evaluation_results")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / filename
                
                with open(output_path, 'w') as f:
                    json.dump(self.results, f, indent=2)
                
                return {
                    "status": "exported",
                    "format": format,
                    "filename": str(output_path),
                    "count": len(self.results)
                }
            else:
                raise HTTPException(status_code=400, 
                                  detail=f"Unsupported format: {format}")
    
    def _get_html_interface(self) -> str:
        """Get HTML for the evaluation interface"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Human Evaluation Interface</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .container {
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }
                .evaluation-item {
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                }
                .question {
                    font-weight: bold;
                    color: #495057;
                    margin-bottom: 10px;
                    font-size: 18px;
                }
                .answer {
                    background: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                    border-left: 4px solid #667eea;
                }
                .contexts {
                    background: #e9ecef;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                    max-height: 200px;
                    overflow-y: auto;
                }
                .context-item {
                    background: white;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 3px;
                    font-size: 14px;
                }
                .scoring {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }
                .score-item {
                    display: flex;
                    flex-direction: column;
                }
                .score-item label {
                    font-weight: bold;
                    margin-bottom: 5px;
                    color: #495057;
                }
                .score-slider {
                    width: 100%;
                    margin: 5px 0;
                }
                .score-value {
                    text-align: center;
                    font-weight: bold;
                    color: #667eea;
                }
                .comments {
                    width: 100%;
                    min-height: 100px;
                    padding: 10px;
                    border: 1px solid #dee2e6;
                    border-radius: 5px;
                    font-family: inherit;
                    margin: 10px 0;
                }
                .submit-btn {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    padding: 12px 30px;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    display: block;
                    margin: 20px auto;
                    transition: transform 0.2s;
                }
                .submit-btn:hover {
                    transform: scale(1.05);
                }
                .status {
                    text-align: center;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .status.connected {
                    background: #d4edda;
                    color: #155724;
                }
                .status.waiting {
                    background: #fff3cd;
                    color: #856404;
                }
                .status.error {
                    background: #f8d7da;
                    color: #721c24;
                }
                .stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 10px;
                    margin: 20px 0;
                }
                .stat-item {
                    background: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                }
                .stat-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #667eea;
                }
                .stat-label {
                    font-size: 12px;
                    color: #6c757d;
                    text-transform: uppercase;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 RAG Human Evaluation Interface</h1>
                
                <div id="status" class="status waiting">
                    Connecting to evaluation server...
                </div>
                
                <div class="stats" id="stats">
                    <div class="stat-item">
                        <div class="stat-value" id="evaluated-count">0</div>
                        <div class="stat-label">Evaluated</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="queue-size">0</div>
                        <div class="stat-label">In Queue</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="avg-score">-</div>
                        <div class="stat-label">Avg Score</div>
                    </div>
                </div>
                
                <div id="evaluation-container"></div>
            </div>
            
            <script>
                let ws = null;
                let currentItem = null;
                let evaluatedCount = 0;
                let totalScore = 0;
                
                function connect() {
                    ws = new WebSocket('ws://localhost:8090/ws/evaluate');
                    
                    ws.onopen = function() {
                        updateStatus('Connected to evaluation server', 'connected');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        handleMessage(data);
                    };
                    
                    ws.onerror = function(error) {
                        updateStatus('Connection error', 'error');
                        setTimeout(connect, 5000);
                    };
                    
                    ws.onclose = function() {
                        updateStatus('Disconnected. Reconnecting...', 'error');
                        setTimeout(connect, 5000);
                    };
                }
                
                function handleMessage(data) {
                    switch(data.type) {
                        case 'connected':
                            console.log('Connected:', data.connection_id);
                            break;
                        case 'evaluate':
                            currentItem = data.item;
                            displayEvaluationItem(data.item);
                            updateStatus('Please evaluate this item', 'connected');
                            break;
                        case 'waiting':
                            updateStatus(data.message, 'waiting');
                            document.getElementById('evaluation-container').innerHTML = 
                                '<p style="text-align: center; color: #6c757d;">Waiting for items to evaluate...</p>';
                            break;
                        case 'evaluation_received':
                            evaluatedCount++;
                            updateStats();
                            updateStatus('Evaluation saved! Waiting for next item...', 'connected');
                            break;
                    }
                }
                
                function displayEvaluationItem(item) {
                    const contextsHtml = item.contexts.map((ctx, i) => 
                        `<div class="context-item">Context ${i + 1}: ${ctx}</div>`
                    ).join('');
                    
                    const html = `
                        <div class="evaluation-item">
                            <div class="question">Question: ${item.question}</div>
                            <div class="answer">
                                <strong>Generated Answer:</strong><br>
                                ${item.answer}
                            </div>
                            <div class="contexts">
                                <strong>Retrieved Contexts:</strong>
                                ${contextsHtml}
                            </div>
                            
                            <div class="scoring">
                                <div class="score-item">
                                    <label>Relevance (0-5)</label>
                                    <input type="range" class="score-slider" id="relevance" 
                                           min="0" max="5" step="0.5" value="2.5"
                                           oninput="updateScoreDisplay(this)">
                                    <div class="score-value" id="relevance-value">2.5</div>
                                </div>
                                
                                <div class="score-item">
                                    <label>Accuracy (0-5)</label>
                                    <input type="range" class="score-slider" id="accuracy" 
                                           min="0" max="5" step="0.5" value="2.5"
                                           oninput="updateScoreDisplay(this)">
                                    <div class="score-value" id="accuracy-value">2.5</div>
                                </div>
                                
                                <div class="score-item">
                                    <label>Completeness (0-5)</label>
                                    <input type="range" class="score-slider" id="completeness" 
                                           min="0" max="5" step="0.5" value="2.5"
                                           oninput="updateScoreDisplay(this)">
                                    <div class="score-value" id="completeness-value">2.5</div>
                                </div>
                                
                                <div class="score-item">
                                    <label>Fluency (0-5)</label>
                                    <input type="range" class="score-slider" id="fluency" 
                                           min="0" max="5" step="0.5" value="2.5"
                                           oninput="updateScoreDisplay(this)">
                                    <div class="score-value" id="fluency-value">2.5</div>
                                </div>
                                
                                <div class="score-item">
                                    <label>Overall (0-5)</label>
                                    <input type="range" class="score-slider" id="overall" 
                                           min="0" max="5" step="0.5" value="2.5"
                                           oninput="updateScoreDisplay(this)">
                                    <div class="score-value" id="overall-value">2.5</div>
                                </div>
                            </div>
                            
                            <textarea class="comments" id="comments" 
                                      placeholder="Additional comments (optional)"></textarea>
                            
                            <button class="submit-btn" onclick="submitEvaluation()">
                                Submit Evaluation
                            </button>
                        </div>
                    `;
                    
                    document.getElementById('evaluation-container').innerHTML = html;
                }
                
                function updateScoreDisplay(slider) {
                    document.getElementById(slider.id + '-value').textContent = slider.value;
                }
                
                function submitEvaluation() {
                    if (!currentItem) return;
                    
                    const evaluation = {
                        relevance_score: parseFloat(document.getElementById('relevance').value),
                        accuracy_score: parseFloat(document.getElementById('accuracy').value),
                        completeness_score: parseFloat(document.getElementById('completeness').value),
                        fluency_score: parseFloat(document.getElementById('fluency').value),
                        overall_score: parseFloat(document.getElementById('overall').value),
                        comments: document.getElementById('comments').value
                    };
                    
                    totalScore += evaluation.overall_score;
                    
                    ws.send(JSON.stringify(evaluation));
                    currentItem = null;
                    
                    document.getElementById('evaluation-container').innerHTML = 
                        '<p style="text-align: center; color: #28a745;">Evaluation submitted!</p>';
                }
                
                function updateStatus(message, type) {
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = message;
                    statusEl.className = 'status ' + type;
                }
                
                function updateStats() {
                    document.getElementById('evaluated-count').textContent = evaluatedCount;
                    if (evaluatedCount > 0) {
                        const avgScore = (totalScore / evaluatedCount).toFixed(2);
                        document.getElementById('avg-score').textContent = avgScore;
                    }
                }
                
                // Start connection
                connect();
            </script>
        </body>
        </html>
        """
    
    async def add_evaluation_items(self, items: List[Dict[str, Any]]):
        """
        Add items to evaluation queue
        
        Args:
            items: List of items to evaluate
        """
        for item in items:
            if 'id' not in item:
                item['id'] = str(uuid.uuid4())
            await self.evaluation_queue.put(item)
        
        logger.info(f"Added {len(items)} items to evaluation queue")
    
    def get_evaluation_results(self) -> List[Dict[str, Any]]:
        """
        Get all evaluation results
        
        Returns:
            List of evaluation results
        """
        return self.results
    
    def export_results(self, output_path: str):
        """
        Export evaluation results to file
        
        Args:
            output_path: Path to save results
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Exported {len(self.results)} evaluation results to {output_path}")