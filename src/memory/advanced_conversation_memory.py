"""
Advanced Conversation Memory and Multi-turn Support
"""

import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class Message:
    """Single conversation message"""
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    sentiment: float = 0.0

@dataclass
class ConversationTurn:
    """A turn in conversation (query + response + context)"""
    query: Message
    response: Message
    contexts: List[str] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)
    turn_id: str = ""
    parent_turn_id: Optional[str] = None
    children_turn_ids: List[str] = field(default_factory=list)

@dataclass
class ConversationSession:
    """Complete conversation session"""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    turns: List[ConversationTurn]
    metadata: Dict[str, Any]
    summary: Optional[str] = None
    topic_graph: Optional[nx.DiGraph] = None

class AdvancedConversationMemory:
    """
    Production conversation memory with:
    - Multi-turn context tracking
    - Entity and topic continuity
    - Conversation summarization
    - Reference resolution
    - Memory consolidation
    - Episodic and semantic memory
    """
    
    def __init__(
        self,
        max_turns: int = 20,
        max_sessions: int = 100,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_path: Optional[str] = "./conversation_memory"
    ):
        """Initialize conversation memory"""
        
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self.persist_path = persist_path
        
        # Current session
        self.current_session: Optional[ConversationSession] = None
        
        # Session history
        self.sessions: deque = deque(maxlen=max_sessions)
        
        # Episodic memory (recent conversations)
        self.episodic_memory: deque = deque(maxlen=max_turns * 3)
        
        # Semantic memory (learned facts and patterns)
        self.semantic_memory: Dict[str, Any] = {
            "entities": defaultdict(list),  # Entity -> contexts
            "topics": defaultdict(list),    # Topic -> turns
            "patterns": defaultdict(int),    # Query pattern -> count
            "facts": {}                      # Extracted facts
        }
        
        # Embedding model
        self.embedder = SentenceTransformer(embedding_model)
        
        # Entity tracker
        self.entity_context: Dict[str, List[str]] = {}
        
        # Reference resolver
        self.reference_map: Dict[str, str] = {}
        
        # Load existing memory
        if persist_path:
            self.load_memory()
    
    def start_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Start a new conversation session"""
        
        # Save current session if exists
        if self.current_session:
            self.end_session()
        
        self.current_session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            end_time=None,
            turns=[],
            metadata=metadata or {},
            topic_graph=nx.DiGraph()
        )
        
        logger.info(f"Started session {session_id}")
    
    def end_session(self, summary: Optional[str] = None):
        """End current conversation session"""
        
        if not self.current_session:
            return
        
        self.current_session.end_time = datetime.now()
        
        # Generate summary if not provided
        if not summary and len(self.current_session.turns) > 3:
            summary = self.generate_session_summary()
        
        self.current_session.summary = summary
        
        # Add to session history
        self.sessions.append(self.current_session)
        
        # Update semantic memory
        self._update_semantic_memory()
        
        # Save to disk
        if self.persist_path:
            self.save_memory()
        
        logger.info(f"Ended session {self.current_session.session_id}")
        
        self.current_session = None
    
    def add_turn(
        self,
        query: str,
        response: str,
        contexts: List[str] = None,
        relevance_scores: List[float] = None,
        metadata: Optional[Dict] = None
    ) -> ConversationTurn:
        """Add a conversation turn"""
        
        if not self.current_session:
            self.start_session(f"session_{datetime.now().timestamp()}")
        
        # Create messages
        query_msg = Message(
            role=MessageRole.USER,
            content=query,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        response_msg = Message(
            role=MessageRole.ASSISTANT,
            content=response,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Add embeddings
        query_msg.embedding = self.embedder.encode(query)
        response_msg.embedding = self.embedder.encode(response)
        
        # Extract entities and intent
        query_msg.entities = self._extract_entities(query)
        query_msg.intent = self._classify_intent(query)
        
        # Create turn
        turn = ConversationTurn(
            query=query_msg,
            response=response_msg,
            contexts=contexts or [],
            relevance_scores=relevance_scores or [],
            turn_id=f"turn_{len(self.current_session.turns)}"
        )
        
        # Link to previous turn
        if self.current_session.turns:
            prev_turn = self.current_session.turns[-1]
            turn.parent_turn_id = prev_turn.turn_id
            prev_turn.children_turn_ids.append(turn.turn_id)
        
        # Add to session
        self.current_session.turns.append(turn)
        
        # Add to episodic memory
        self.episodic_memory.append(turn)
        
        # Update entity context
        for entity in query_msg.entities:
            if entity not in self.entity_context:
                self.entity_context[entity] = []
            self.entity_context[entity].append(query)
        
        # Update topic graph
        self._update_topic_graph(turn)
        
        # Manage memory size
        if len(self.current_session.turns) > self.max_turns:
            self._consolidate_memory()
        
        return turn
    
    def get_relevant_context(
        self,
        query: str,
        k: int = 5,
        include_semantic: bool = True,
        include_episodic: bool = True
    ) -> List[Dict[str, Any]]:
        """Get relevant context for query"""
        
        contexts = []
        
        # Encode query
        query_embedding = self.embedder.encode(query)
        
        # Search episodic memory
        if include_episodic and self.episodic_memory:
            episodic_contexts = self._search_episodic_memory(query_embedding, k)
            contexts.extend(episodic_contexts)
        
        # Search semantic memory
        if include_semantic:
            semantic_contexts = self._search_semantic_memory(query, query_embedding, k)
            contexts.extend(semantic_contexts)
        
        # Check for entity continuity
        query_entities = self._extract_entities(query)
        for entity in query_entities:
            if entity in self.entity_context:
                for prev_query in self.entity_context[entity][-3:]:
                    contexts.append({
                        "type": "entity_context",
                        "entity": entity,
                        "content": prev_query,
                        "relevance": 0.8
                    })
        
        # Sort by relevance and deduplicate
        contexts = self._deduplicate_contexts(contexts)
        contexts.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        return contexts[:k]
    
    def _search_episodic_memory(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> List[Dict[str, Any]]:
        """Search episodic memory for relevant turns"""
        
        contexts = []
        
        for turn in self.episodic_memory:
            # Calculate similarity with query
            query_sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                turn.query.embedding.reshape(1, -1)
            )[0][0]
            
            # Calculate similarity with response
            response_sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                turn.response.embedding.reshape(1, -1)
            )[0][0]
            
            relevance = max(query_sim, response_sim * 0.8)
            
            contexts.append({
                "type": "episodic",
                "turn_id": turn.turn_id,
                "query": turn.query.content,
                "response": turn.response.content,
                "content": f"Q: {turn.query.content}\nA: {turn.response.content}",
                "relevance": relevance,
                "timestamp": turn.query.timestamp
            })
        
        return contexts
    
    def _search_semantic_memory(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int
    ) -> List[Dict[str, Any]]:
        """Search semantic memory for relevant information"""
        
        contexts = []
        
        # Search by entities
        query_entities = self._extract_entities(query)
        for entity in query_entities:
            if entity in self.semantic_memory["entities"]:
                for context in self.semantic_memory["entities"][entity][-k:]:
                    contexts.append({
                        "type": "semantic_entity",
                        "entity": entity,
                        "content": context,
                        "relevance": 0.7
                    })
        
        # Search facts
        for fact_key, fact_value in self.semantic_memory["facts"].items():
            if any(entity in fact_key for entity in query_entities):
                contexts.append({
                    "type": "semantic_fact",
                    "content": f"{fact_key}: {fact_value}",
                    "relevance": 0.8
                })
        
        return contexts
    
    def resolve_references(self, query: str) -> str:
        """Resolve pronouns and references in query"""
        
        if not self.current_session or not self.current_session.turns:
            return query
        
        resolved = query
        
        # Common pronouns to resolve
        pronouns = {
            "it": self._find_last_entity("NOUN"),
            "they": self._find_last_entity("PERSON", plural=True),
            "he": self._find_last_entity("PERSON", gender="male"),
            "she": self._find_last_entity("PERSON", gender="female"),
            "this": self._find_last_entity("NOUN"),
            "that": self._find_last_entity("NOUN"),
            "these": self._find_last_entity("NOUN", plural=True),
            "those": self._find_last_entity("NOUN", plural=True)
        }
        
        # Replace pronouns with resolved entities
        for pronoun, entity in pronouns.items():
            if entity and pronoun in resolved.lower():
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(pronoun), re.IGNORECASE)
                resolved = pattern.sub(entity, resolved)
        
        # Handle "the same" references
        if "the same" in resolved.lower():
            last_entity = self._find_last_entity("NOUN")
            if last_entity:
                resolved = resolved.replace("the same", last_entity)
        
        return resolved
    
    def _find_last_entity(
        self,
        entity_type: str,
        plural: bool = False,
        gender: Optional[str] = None
    ) -> Optional[str]:
        """Find last mentioned entity of given type"""
        
        if not self.current_session:
            return None
        
        # Search recent turns
        for turn in reversed(self.current_session.turns[-5:]):
            for entity in turn.query.entities:
                # Simple type matching (can be enhanced)
                if entity_type == "NOUN" or entity_type in entity.upper():
                    return entity
        
        return None
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        
        # Simple entity extraction (can be enhanced with NER)
        import re
        
        entities = []
        
        # Extract capitalized words (proper nouns)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities.extend(re.findall(pattern, text))
        
        # Extract quoted phrases
        pattern = r'"([^"]*)"'
        entities.extend(re.findall(pattern, text))
        
        # Extract technical terms (can be domain-specific)
        tech_terms = ['API', 'SDK', 'ML', 'AI', 'RAG', 'LLM']
        for term in tech_terms:
            if term in text:
                entities.append(term)
        
        return list(set(entities))
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'which', 'who', 'when', 'where']):
            return "question"
        elif any(word in query_lower for word in ['how', 'explain', 'describe']):
            return "explanation"
        elif any(word in query_lower for word in ['why', 'because', 'reason']):
            return "reasoning"
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            return "comparison"
        elif any(word in query_lower for word in ['suggest', 'recommend', 'should']):
            return "recommendation"
        else:
            return "general"
    
    def _update_topic_graph(self, turn: ConversationTurn):
        """Update topic graph with new turn"""
        
        if not self.current_session:
            return
        
        # Extract topics from turn
        topics = self._extract_topics(turn.query.content)
        
        # Add nodes for topics
        for topic in topics:
            if topic not in self.current_session.topic_graph:
                self.current_session.topic_graph.add_node(
                    topic,
                    weight=1,
                    turns=[turn.turn_id]
                )
            else:
                # Update existing node
                self.current_session.topic_graph.nodes[topic]['weight'] += 1
                self.current_session.topic_graph.nodes[topic]['turns'].append(turn.turn_id)
        
        # Add edges between co-occurring topics
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                if self.current_session.topic_graph.has_edge(topic1, topic2):
                    self.current_session.topic_graph[topic1][topic2]['weight'] += 1
                else:
                    self.current_session.topic_graph.add_edge(topic1, topic2, weight=1)
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        
        # Simple keyword extraction (can be enhanced)
        import re
        from collections import Counter
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
        
        words = re.findall(r'\b[a-z]+\b', text.lower())
        words = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Get most common words as topics
        word_freq = Counter(words)
        topics = [word for word, _ in word_freq.most_common(5)]
        
        # Add entities as topics
        entities = self._extract_entities(text)
        topics.extend(entities)
        
        return list(set(topics))
    
    def _consolidate_memory(self):
        """Consolidate old turns into semantic memory"""
        
        if not self.current_session:
            return
        
        # Get oldest turns
        old_turns = self.current_session.turns[:-self.max_turns]
        
        for turn in old_turns:
            # Extract and store facts
            facts = self._extract_facts(turn)
            self.semantic_memory["facts"].update(facts)
            
            # Update entity memory
            for entity in turn.query.entities:
                self.semantic_memory["entities"][entity].append(turn.query.content)
            
            # Update pattern memory
            pattern = self._get_query_pattern(turn.query.content)
            self.semantic_memory["patterns"][pattern] += 1
        
        # Keep only recent turns
        self.current_session.turns = self.current_session.turns[-self.max_turns:]
    
    def _extract_facts(self, turn: ConversationTurn) -> Dict[str, str]:
        """Extract facts from conversation turn"""
        
        facts = {}
        
        # Look for definition patterns
        import re
        
        # Pattern: X is Y
        pattern = r'(\w+)\s+is\s+(.+?)(?:\.|$)'
        matches = re.findall(pattern, turn.response.content)
        for subject, predicate in matches:
            facts[subject] = predicate
        
        # Pattern: X means Y
        pattern = r'(\w+)\s+means\s+(.+?)(?:\.|$)'
        matches = re.findall(pattern, turn.response.content)
        for term, definition in matches:
            facts[f"{term}_definition"] = definition
        
        return facts
    
    def _get_query_pattern(self, query: str) -> str:
        """Extract query pattern for learning"""
        
        # Replace specific terms with placeholders
        import re
        
        pattern = query.lower()
        
        # Replace numbers
        pattern = re.sub(r'\d+', '<NUM>', pattern)
        
        # Replace quoted strings
        pattern = re.sub(r'"[^"]*"', '<STR>', pattern)
        
        # Replace entities
        for entity in self._extract_entities(query):
            pattern = pattern.replace(entity.lower(), '<ENT>')
        
        return pattern
    
    def _deduplicate_contexts(self, contexts: List[Dict]) -> List[Dict]:
        """Remove duplicate contexts"""
        
        seen = set()
        unique = []
        
        for context in contexts:
            content = context.get("content", "")
            if content not in seen:
                seen.add(content)
                unique.append(context)
        
        return unique
    
    def _update_semantic_memory(self):
        """Update semantic memory from completed session"""
        
        if not self.current_session:
            return
        
        # Extract session-level patterns
        intents = [turn.query.intent for turn in self.current_session.turns]
        
        # Store topic progression
        if self.current_session.topic_graph:
            # Get central topics
            if self.current_session.topic_graph.nodes():
                centrality = nx.degree_centrality(self.current_session.topic_graph)
                top_topics = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for topic, score in top_topics:
                    self.semantic_memory["topics"][topic].append(self.current_session.session_id)
    
    def generate_session_summary(self) -> str:
        """Generate summary of current session"""
        
        if not self.current_session or not self.current_session.turns:
            return ""
        
        # Extract key information
        topics = []
        questions = []
        conclusions = []
        
        for turn in self.current_session.turns:
            # Collect questions
            if turn.query.intent in ["question", "explanation"]:
                questions.append(turn.query.content)
            
            # Extract topics
            turn_topics = self._extract_topics(turn.query.content)
            topics.extend(turn_topics)
            
            # Look for conclusions
            if any(word in turn.response.content.lower() 
                  for word in ["therefore", "thus", "in conclusion", "summary"]):
                conclusions.append(turn.response.content[:200])
        
        # Build summary
        summary_parts = []
        
        if topics:
            from collections import Counter
            top_topics = [t for t, _ in Counter(topics).most_common(3)]
            summary_parts.append(f"Topics discussed: {', '.join(top_topics)}")
        
        if questions:
            summary_parts.append(f"Key questions: {questions[0][:100]}")
        
        if conclusions:
            summary_parts.append(f"Main points: {conclusions[0][:150]}")
        
        summary = ". ".join(summary_parts)
        
        return summary
    
    def get_conversation_flow(self) -> List[Dict[str, Any]]:
        """Get conversation flow analysis"""
        
        if not self.current_session:
            return []
        
        flow = []
        
        for i, turn in enumerate(self.current_session.turns):
            flow_item = {
                "turn": i,
                "intent": turn.query.intent,
                "entities": turn.query.entities,
                "sentiment": turn.query.sentiment,
                "topic_shift": False
            }
            
            # Detect topic shifts
            if i > 0:
                prev_topics = self._extract_topics(self.current_session.turns[i-1].query.content)
                curr_topics = self._extract_topics(turn.query.content)
                
                overlap = set(prev_topics) & set(curr_topics)
                if len(overlap) < len(curr_topics) / 2:
                    flow_item["topic_shift"] = True
            
            flow.append(flow_item)
        
        return flow
    
    def save_memory(self, path: Optional[str] = None):
        """Save memory to disk"""
        
        save_path = path or self.persist_path
        if not save_path:
            return
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save current session
        if self.current_session:
            session_file = f"{save_path}/current_session.pkl"
            with open(session_file, 'wb') as f:
                pickle.dump(self.current_session, f)
        
        # Save session history
        history_file = f"{save_path}/session_history.pkl"
        with open(history_file, 'wb') as f:
            pickle.dump(list(self.sessions), f)
        
        # Save semantic memory
        semantic_file = f"{save_path}/semantic_memory.json"
        # Convert to JSON-serializable format
        semantic_data = {
            "entities": dict(self.semantic_memory["entities"]),
            "topics": dict(self.semantic_memory["topics"]),
            "patterns": dict(self.semantic_memory["patterns"]),
            "facts": self.semantic_memory["facts"]
        }
        with open(semantic_file, 'w') as f:
            json.dump(semantic_data, f, indent=2, default=str)
        
        logger.info(f"Memory saved to {save_path}")
    
    def load_memory(self, path: Optional[str] = None):
        """Load memory from disk"""
        
        load_path = path or self.persist_path
        if not load_path:
            return
        
        import os
        
        # Load current session
        session_file = f"{load_path}/current_session.pkl"
        if os.path.exists(session_file):
            with open(session_file, 'rb') as f:
                self.current_session = pickle.load(f)
        
        # Load session history
        history_file = f"{load_path}/session_history.pkl"
        if os.path.exists(history_file):
            with open(history_file, 'rb') as f:
                sessions = pickle.load(f)
                self.sessions = deque(sessions, maxlen=self.max_sessions)
        
        # Load semantic memory
        semantic_file = f"{load_path}/semantic_memory.json"
        if os.path.exists(semantic_file):
            with open(semantic_file, 'r') as f:
                semantic_data = json.load(f)
                self.semantic_memory["entities"] = defaultdict(list, semantic_data["entities"])
                self.semantic_memory["topics"] = defaultdict(list, semantic_data["topics"])
                self.semantic_memory["patterns"] = defaultdict(int, semantic_data["patterns"])
                self.semantic_memory["facts"] = semantic_data["facts"]
        
        logger.info(f"Memory loaded from {load_path}")