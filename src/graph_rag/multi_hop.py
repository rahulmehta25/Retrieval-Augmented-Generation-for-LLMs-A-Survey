"""
Multi-Hop Reasoning for Graph-Based RAG

Implements iterative retrieval with reasoning chains.
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import asyncio

from .knowledge_graph import GraphRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Container for a reasoning step"""
    step_number: int
    sub_question: str
    retrieved_contexts: List[str]
    reasoning: str
    confidence: float


class MultiHopReasoning:
    """Implements multi-hop reasoning over knowledge graph"""
    
    def __init__(self, graph_rag: GraphRAG, llm_generator=None):
        """
        Initialize multi-hop reasoning
        
        Args:
            graph_rag: GraphRAG instance
            llm_generator: LLM for reasoning
        """
        self.graph = graph_rag
        self.llm = llm_generator
        self.max_hops = 3
        
    async def multi_hop_retrieve(self, query: str, max_hops: Optional[int] = None) -> Dict[str, Any]:
        """
        Iterative retrieval with reasoning
        
        Args:
            query: Initial question
            max_hops: Maximum reasoning steps
        
        Returns:
            Multi-hop retrieval results
        """
        max_hops = max_hops or self.max_hops
        
        reasoning_chain = []
        all_contexts = []
        
        # Step 1: Decompose query
        sub_questions = await self._decompose_query(query)
        
        # Step 2: Iterative retrieval and reasoning
        for hop in range(min(max_hops, len(sub_questions))):
            sub_question = sub_questions[hop]
            
            # Retrieve for current sub-question
            hop_contexts = self.graph.graph_retrieve(sub_question, k=3)
            all_contexts.extend(hop_contexts)
            
            # Reason about retrieved information
            reasoning = await self._reason_about_context(
                sub_question, 
                hop_contexts,
                reasoning_chain
            )
            
            step = ReasoningStep(
                step_number=hop + 1,
                sub_question=sub_question,
                retrieved_contexts=hop_contexts,
                reasoning=reasoning,
                confidence=self._calculate_confidence(hop_contexts)
            )
            reasoning_chain.append(step)
            
            # Check if we have enough information
            if await self._is_sufficient(query, reasoning_chain):
                break
            
            # Generate follow-up question if needed
            if hop < max_hops - 1 and hop + 1 >= len(sub_questions):
                next_query = await self._generate_followup_query(
                    query, reasoning_chain, all_contexts
                )
                if next_query:
                    sub_questions.append(next_query)
        
        # Step 3: Synthesize final answer
        final_answer = await self._synthesize_answer(query, reasoning_chain)
        
        return {
            'query': query,
            'reasoning_chain': reasoning_chain,
            'all_contexts': all_contexts,
            'final_answer': final_answer,
            'num_hops': len(reasoning_chain)
        }
    
    async def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-questions
        
        Args:
            query: Original question
        
        Returns:
            List of sub-questions
        """
        if not self.llm:
            # Simple heuristic decomposition
            return self._heuristic_decompose(query)
        
        prompt = f"""Decompose this complex question into simpler sub-questions that can be answered sequentially.
Each sub-question should build upon the previous ones.

Question: {query}

List each sub-question on a new line:"""
        
        try:
            response = self.llm.generate(prompt, max_new_tokens=200)
            sub_questions = [q.strip() for q in response.split('\n') if q.strip()]
            return sub_questions[:self.max_hops]
        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}")
            return self._heuristic_decompose(query)
    
    def _heuristic_decompose(self, query: str) -> List[str]:
        """
        Heuristic query decomposition
        
        Args:
            query: Original question
        
        Returns:
            List of sub-questions
        """
        # Check for multi-part questions
        if " and " in query.lower():
            parts = query.split(" and ")
            return [part.strip() + "?" for part in parts]
        
        # Check for causal questions
        if "why" in query.lower() or "how" in query.lower():
            return [
                query,
                "What are the main factors involved?",
                "What is the underlying mechanism?"
            ]
        
        # Default: return original query
        return [query]
    
    async def _reason_about_context(self, sub_question: str, contexts: List[str], 
                                   previous_reasoning: List[ReasoningStep]) -> str:
        """
        Reason about retrieved context
        
        Args:
            sub_question: Current sub-question
            contexts: Retrieved contexts
            previous_reasoning: Previous reasoning steps
        
        Returns:
            Reasoning about the context
        """
        if not self.llm:
            return self._simple_reasoning(sub_question, contexts)
        
        # Build context from previous reasoning
        prev_context = ""
        if previous_reasoning:
            prev_context = "Previous reasoning:\n"
            for step in previous_reasoning[-2:]:  # Last 2 steps
                prev_context += f"- {step.sub_question}: {step.reasoning[:100]}...\n"
        
        prompt = f"""{prev_context}

Current question: {sub_question}

Context:
{' '.join(contexts[:2])}

Based on the context, what can we conclude about the question? 
Provide a brief reasoning:"""
        
        try:
            reasoning = self.llm.generate(prompt, max_new_tokens=150)
            return reasoning.strip()
        except Exception as e:
            logger.warning(f"LLM reasoning failed: {e}")
            return self._simple_reasoning(sub_question, contexts)
    
    def _simple_reasoning(self, sub_question: str, contexts: List[str]) -> str:
        """
        Simple heuristic reasoning
        
        Args:
            sub_question: Current question
            contexts: Retrieved contexts
        
        Returns:
            Simple reasoning
        """
        if contexts:
            return f"Found {len(contexts)} relevant contexts for: {sub_question}"
        else:
            return f"No direct information found for: {sub_question}"
    
    def _calculate_confidence(self, contexts: List[str]) -> float:
        """
        Calculate confidence score for retrieved contexts
        
        Args:
            contexts: Retrieved contexts
        
        Returns:
            Confidence score (0-1)
        """
        if not contexts:
            return 0.0
        
        # Simple heuristic: more contexts = higher confidence
        base_confidence = min(len(contexts) / 5, 1.0)
        
        # Adjust based on context length
        avg_length = sum(len(c) for c in contexts) / len(contexts)
        length_factor = min(avg_length / 500, 1.0)
        
        return base_confidence * 0.7 + length_factor * 0.3
    
    async def _is_sufficient(self, original_query: str, 
                           reasoning_chain: List[ReasoningStep]) -> bool:
        """
        Check if we have sufficient information
        
        Args:
            original_query: Original question
            reasoning_chain: Current reasoning chain
        
        Returns:
            True if sufficient information gathered
        """
        if not reasoning_chain:
            return False
        
        # Check confidence scores
        avg_confidence = sum(step.confidence for step in reasoning_chain) / len(reasoning_chain)
        if avg_confidence > 0.8:
            return True
        
        # Check if we have enough context
        total_contexts = sum(len(step.retrieved_contexts) for step in reasoning_chain)
        if total_contexts >= 10:
            return True
        
        return False
    
    async def _generate_followup_query(self, original_query: str,
                                      reasoning_chain: List[ReasoningStep],
                                      contexts: List[str]) -> Optional[str]:
        """
        Generate follow-up query based on current knowledge
        
        Args:
            original_query: Original question
            reasoning_chain: Current reasoning chain
            contexts: All retrieved contexts
        
        Returns:
            Follow-up question or None
        """
        if not self.llm:
            return None
        
        # Build summary of what we know
        knowledge_summary = "What we know so far:\n"
        for step in reasoning_chain[-2:]:  # Last 2 steps
            knowledge_summary += f"- {step.reasoning[:100]}...\n"
        
        prompt = f"""Original question: {original_query}

{knowledge_summary}

What additional information do we need to fully answer the original question?
Generate a specific follow-up question:"""
        
        try:
            followup = self.llm.generate(prompt, max_new_tokens=50)
            return followup.strip() if followup.strip() else None
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")
            return None
    
    async def _synthesize_answer(self, query: str, 
                                reasoning_chain: List[ReasoningStep]) -> str:
        """
        Synthesize final answer from reasoning chain
        
        Args:
            query: Original question
            reasoning_chain: Complete reasoning chain
        
        Returns:
            Final synthesized answer
        """
        if not reasoning_chain:
            return "Unable to find relevant information to answer the question."
        
        if not self.llm:
            # Simple concatenation
            answer_parts = []
            for step in reasoning_chain:
                if step.reasoning:
                    answer_parts.append(step.reasoning)
            return " ".join(answer_parts)
        
        # Build context from reasoning chain
        reasoning_context = ""
        for step in reasoning_chain:
            reasoning_context += f"Step {step.step_number}: {step.reasoning}\n"
        
        prompt = f"""Based on the following reasoning chain, provide a comprehensive answer to the question.

Question: {query}

Reasoning chain:
{reasoning_context}

Final answer:"""
        
        try:
            answer = self.llm.generate(prompt, max_new_tokens=300)
            return answer.strip()
        except Exception as e:
            logger.warning(f"Answer synthesis failed: {e}")
            # Fallback to simple concatenation
            return " ".join(step.reasoning for step in reasoning_chain if step.reasoning)
    
    def visualize_reasoning_chain(self, reasoning_chain: List[ReasoningStep]) -> str:
        """
        Create a text visualization of the reasoning chain
        
        Args:
            reasoning_chain: Reasoning steps
        
        Returns:
            Text visualization
        """
        visualization = "Multi-Hop Reasoning Chain\n"
        visualization += "=" * 50 + "\n\n"
        
        for step in reasoning_chain:
            visualization += f"🔍 Step {step.step_number}\n"
            visualization += f"   Question: {step.sub_question}\n"
            visualization += f"   Contexts Found: {len(step.retrieved_contexts)}\n"
            visualization += f"   Confidence: {step.confidence:.2f}\n"
            visualization += f"   Reasoning: {step.reasoning[:150]}...\n"
            visualization += "\n"
        
        return visualization