"""
Generation Service - Handles response generation
"""

import logging
import time
from typing import List, Optional, Dict, Any
from .interfaces import GenerationServiceInterface, GenerationResult
from ..generation.generator import LLMGenerator

logger = logging.getLogger(__name__)

class GenerationService:
    """
    Service responsible for response generation
    Implements single responsibility principle for generation operations
    """
    
    def __init__(self, model_name: str = "gemma:2b"):
        """Initialize generation service"""
        self.generator = LLMGenerator(model_name=model_name)
        self.model_name = model_name
        logger.info(f"GenerationService initialized with model: {model_name}")
    
    def generate_response(
        self,
        query: str,
        contexts: List[str],
        conversation_history: Optional[List[Dict]] = None,
        max_tokens: int = 500,
        temperature: float = 0.1
    ) -> GenerationResult:
        """
        Generate response based on contexts and query
        
        Args:
            query: User query
            contexts: Retrieved contexts
            conversation_history: Optional conversation history
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            GenerationResult with generated response and metadata
        """
        start_time = time.time()
        logger.info(f"Generating response for query: {query[:50]}...")
        
        try:
            # Build prompt with contexts
            prompt = self._build_prompt(query, contexts, conversation_history)
            
            # Generate response
            response = self.generator.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Calculate metrics
            generation_time = (time.time() - start_time) * 1000
            tokens_generated = len(response.split())
            
            result = GenerationResult(
                answer=response,
                tokens_generated=tokens_generated,
                generation_time_ms=generation_time
            )
            
            logger.info(f"Response generated in {generation_time:.0f}ms with {tokens_generated} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Return error response
            return GenerationResult(
                answer=f"I apologize, but I encountered an error generating a response: {str(e)}",
                tokens_generated=0,
                generation_time_ms=(time.time() - start_time) * 1000
            )
    
    def _build_prompt(
        self,
        query: str,
        contexts: List[str],
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Build prompt for response generation
        
        Args:
            query: User query
            contexts: Retrieved contexts
            conversation_history: Optional conversation history
            
        Returns:
            Formatted prompt string
        """
        # Build context section
        context_section = ""
        if contexts:
            context_section = "Context:\n" + "\n\n".join(contexts) + "\n\n"
        
        # Build conversation history section
        history_section = ""
        if conversation_history:
            history_section = "Previous conversation:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns
                if "query" in turn and "response" in turn:
                    history_section += f"Human: {turn['query']}\n"
                    history_section += f"Assistant: {turn['response']}\n\n"
        
        # Build complete prompt
        prompt = f"""Based on the following context, answer the question accurately and helpfully.

{context_section}{history_section}Question: {query}

Answer:"""
        
        return prompt
    
    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """
        Generate a summary of the given text
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Generated summary
        """
        logger.info("Generating text summary...")
        
        try:
            prompt = f"""Please provide a concise summary of the following text in no more than {max_length} words:

{text}

Summary:"""
            
            summary = self.generator.generate(prompt, max_tokens=max_length)
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary."
    
    def generate_follow_up_questions(self, query: str, response: str) -> List[str]:
        """
        Generate follow-up questions based on query and response
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            List of follow-up questions
        """
        logger.info("Generating follow-up questions...")
        
        try:
            prompt = f"""Based on the following question and answer, suggest 3 relevant follow-up questions:

Question: {query}
Answer: {response}

Follow-up questions:
1."""
            
            followup_text = self.generator.generate(prompt, max_tokens=200)
            
            # Parse follow-up questions
            questions = []
            lines = followup_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and any(line.startswith(str(i) + '.') for i in range(1, 6)):
                    question = line.split('.', 1)[1].strip() if '.' in line else line
                    if question and question.endswith('?'):
                        questions.append(question)
            
            return questions[:3]  # Return max 3 questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "model_name": self.model_name,
            "total_responses_generated": 0,  # Would track actual counts
            "avg_generation_time": 0.0,
            "avg_tokens_per_response": 0.0
        }