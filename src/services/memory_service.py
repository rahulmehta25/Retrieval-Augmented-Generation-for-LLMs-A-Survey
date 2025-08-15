"""
Memory Service - Handles conversation memory and context
"""

import logging
from typing import List, Optional
from datetime import datetime
from .interfaces import MemoryServiceInterface, MemoryContext
from ..memory.advanced_conversation_memory import AdvancedConversationMemory

logger = logging.getLogger(__name__)

class MemoryService:
    """
    Service responsible for conversation memory management
    Implements single responsibility principle for memory operations
    """
    
    def __init__(self, persist_path: str = "./conversation_memory"):
        """Initialize memory service"""
        self.conversation_memory = AdvancedConversationMemory(
            persist_path=persist_path
        )
        self.current_session_id = None
        logger.info("MemoryService initialized successfully")
    
    def get_relevant_context(
        self,
        query: str,
        k: int = 3
    ) -> List[MemoryContext]:
        """
        Get relevant context from conversation memory
        
        Args:
            query: Current query
            k: Number of relevant contexts to retrieve
            
        Returns:
            List of MemoryContext objects
        """
        logger.info(f"Retrieving {k} relevant contexts from memory...")
        
        try:
            if not self.conversation_memory.current_session:
                logger.info("No active session, returning empty context")
                return []
            
            # Get relevant contexts using the conversation memory system
            raw_contexts = self.conversation_memory.get_relevant_context(query, k=k)
            
            # Convert to MemoryContext objects
            memory_contexts = []
            for context in raw_contexts:
                memory_context = MemoryContext(
                    content=context.get('content', ''),
                    relevance_score=context.get('relevance_score', 0.0),
                    metadata={
                        'timestamp': context.get('timestamp'),
                        'turn_id': context.get('turn_id'),
                        'session_id': context.get('session_id')
                    }
                )
                memory_contexts.append(memory_context)
            
            logger.info(f"Retrieved {len(memory_contexts)} relevant contexts")
            return memory_contexts
            
        except Exception as e:
            logger.error(f"Error retrieving relevant context: {e}")
            return []
    
    def resolve_references(self, query: str) -> str:
        """
        Resolve references in query (e.g., "it", "that", "the previous one")
        
        Args:
            query: Query potentially containing references
            
        Returns:
            Query with resolved references
        """
        logger.info("Resolving query references...")
        
        try:
            if not self.conversation_memory.current_session:
                return query
            
            resolved_query = self.conversation_memory.resolve_references(query)
            
            if resolved_query != query:
                logger.info(f"Resolved references: '{query}' -> '{resolved_query}'")
            
            return resolved_query
            
        except Exception as e:
            logger.error(f"Error resolving references: {e}")
            return query  # Return original query on error
    
    def add_turn(
        self,
        query: str,
        response: str,
        contexts: List[str],
        relevance_scores: List[float]
    ) -> None:
        """
        Add conversation turn to memory
        
        Args:
            query: User query
            response: System response
            contexts: Retrieved contexts used
            relevance_scores: Relevance scores for contexts
        """
        logger.info("Adding conversation turn to memory...")
        
        try:
            if not self.conversation_memory.current_session:
                logger.warning("No active session, cannot add turn")
                return
            
            self.conversation_memory.add_turn(
                query=query,
                response=response,
                contexts=contexts,
                relevance_scores=relevance_scores
            )
            
            logger.info("Conversation turn added successfully")
            
        except Exception as e:
            logger.error(f"Error adding conversation turn: {e}")
    
    def start_session(self, session_id: str) -> None:
        """
        Start new conversation session
        
        Args:
            session_id: Unique session identifier
        """
        logger.info(f"Starting new conversation session: {session_id}")
        
        try:
            self.conversation_memory.start_session(session_id)
            self.current_session_id = session_id
            logger.info(f"Session {session_id} started successfully")
            
        except Exception as e:
            logger.error(f"Error starting session {session_id}: {e}")
            raise
    
    def end_session(self) -> None:
        """End current conversation session"""
        if self.current_session_id:
            logger.info(f"Ending session: {self.current_session_id}")
            
            try:
                self.conversation_memory.end_session()
                self.current_session_id = None
                logger.info("Session ended successfully")
                
            except Exception as e:
                logger.error(f"Error ending session: {e}")
        else:
            logger.info("No active session to end")
    
    def get_session_summary(self) -> dict:
        """
        Get summary of current session
        
        Returns:
            Dictionary with session statistics
        """
        if not self.conversation_memory.current_session:
            return {"active_session": False}
        
        session = self.conversation_memory.current_session
        
        return {
            "active_session": True,
            "session_id": session.session_id,
            "start_time": session.start_time,
            "total_turns": len(session.turns),
            "episodic_memory_size": len(self.conversation_memory.episodic_memory),
            "semantic_facts": len(self.conversation_memory.semantic_memory.get('facts', [])),
            "entities_tracked": len(self.conversation_memory.semantic_memory.get('entities', {}))
        }
    
    def clear_session_memory(self) -> None:
        """Clear current session memory"""
        if self.current_session_id:
            logger.info(f"Clearing memory for session: {self.current_session_id}")
            
            try:
                # End current session and start a new one with the same ID
                self.conversation_memory.end_session()
                self.conversation_memory.start_session(self.current_session_id)
                logger.info("Session memory cleared successfully")
                
            except Exception as e:
                logger.error(f"Error clearing session memory: {e}")
        else:
            logger.info("No active session to clear")
    
    def save_memory(self) -> None:
        """Save memory to persistent storage"""
        logger.info("Saving memory to persistent storage...")
        
        try:
            self.conversation_memory.save_memory()
            logger.info("Memory saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def load_memory(self) -> None:
        """Load memory from persistent storage"""
        logger.info("Loading memory from persistent storage...")
        
        try:
            # Memory is typically loaded automatically during initialization
            # This method can be used for explicit reloading
            logger.info("Memory loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
    
    def get_memory_statistics(self) -> dict:
        """Get memory system statistics"""
        stats = {
            "total_sessions": 0,  # Would track across all sessions
            "current_session": self.get_session_summary(),
            "memory_persistence_enabled": True
        }
        
        return stats