"""
Self-RAG Implementation

Self-Reflecting RAG with critique and refinement mechanisms.
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass 
class ReflectionResult:
    """Container for self-reflection results"""
    iteration: int
    retrieval_score: float
    answer: str
    critique: Dict[str, Any]
    is_satisfactory: bool
    confidence: float


class SelfRAG:
    """Self-Reflecting RAG with critique mechanisms"""
    
    def __init__(self, base_rag, llm_generator=None):
        """
        Initialize Self-RAG
        
        Args:
            base_rag: Base RAG system
            llm_generator: LLM for reflection
        """
        self.rag = base_rag
        self.llm = llm_generator
        self.max_iterations = 3
        
    async def query_with_reflection(self, query: str) -> Dict[str, Any]:
        """
        RAG with self-reflection loop
        
        Args:
            query: User question
        
        Returns:
            Query results with reflection history
        """
        reflection_history = []
        best_answer = None
        best_confidence = 0.0
        
        for iteration in range(self.max_iterations):
            # Step 1: Retrieve
            contexts = await self._retrieve_async(query)
            
            # Step 2: Assess retrieval quality
            retrieval_score = await self._assess_retrieval(query, contexts)
            
            # Step 3: Decide if we need better retrieval
            if retrieval_score < 0.7 and iteration < self.max_iterations - 1:
                # Reformulate query for better retrieval
                query = await self._reformulate_query(query, contexts, reflection_history)
                continue
            
            # Step 4: Generate answer
            answer = await self._generate_answer(query, contexts)
            
            # Step 5: Self-critique
            critique = await self._critique_answer(query, answer, contexts)
            
            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(retrieval_score, critique)
            
            # Create reflection result
            reflection = ReflectionResult(
                iteration=iteration + 1,
                retrieval_score=retrieval_score,
                answer=answer,
                critique=critique,
                is_satisfactory=critique.get("is_satisfactory", False),
                confidence=confidence
            )
            
            reflection_history.append(reflection)
            
            # Track best answer
            if confidence > best_confidence:
                best_answer = answer
                best_confidence = confidence
            
            # Check if satisfactory
            if critique.get("is_satisfactory", False):
                break
            
            # Step 7: Refine if needed
            if iteration < self.max_iterations - 1:
                answer = await self._refine_answer(
                    query, answer, critique, contexts
                )
                reflection.answer = answer  # Update with refined answer
        
        return {
            'answer': best_answer or answer,
            'contexts': contexts,
            'reflection_history': reflection_history,
            'confidence': best_confidence,
            'iterations': len(reflection_history)
        }
    
    async def _retrieve_async(self, query: str) -> List[str]:
        """
        Async wrapper for retrieval
        
        Args:
            query: User question
        
        Returns:
            Retrieved contexts
        """
        loop = asyncio.get_event_loop()
        
        def retrieve_sync():
            docs = self.rag.retrieve(query)
            return [doc.get('content', str(doc)) for doc in docs]
        
        return await loop.run_in_executor(None, retrieve_sync)
    
    async def _assess_retrieval(self, query: str, contexts: List[str]) -> float:
        """
        Assess if retrieved contexts are sufficient
        
        Args:
            query: User question
            contexts: Retrieved contexts
        
        Returns:
            Retrieval quality score (0-1)
        """
        if not contexts:
            return 0.0
        
        if not self.llm:
            # Simple heuristic
            return min(len(contexts) / 5, 1.0)
        
        prompt = f"""Question: {query}

Retrieved Contexts:
{' '.join(contexts[:3])}

Rate how well these contexts can answer the question (0-1):
Consider: relevance, completeness, and clarity.
Output only the numerical score:"""
        
        try:
            score_str = self.llm.generate(prompt, max_new_tokens=10)
            score = float(score_str.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Retrieval assessment failed: {e}")
            return 0.5
    
    async def _reformulate_query(self, query: str, contexts: List[str], 
                                history: List[ReflectionResult]) -> str:
        """
        Reformulate query based on reflection
        
        Args:
            query: Original query
            contexts: Current contexts
            history: Reflection history
        
        Returns:
            Reformulated query
        """
        if not self.llm:
            return query
        
        # Build context from history
        history_context = ""
        if history:
            last = history[-1]
            history_context = f"Previous attempt scored {last.retrieval_score:.2f}"
        
        prompt = f"""Original question: {query}
{history_context}

The current retrieval is insufficient. Reformulate the question to be more specific and clear:"""
        
        try:
            reformulated = self.llm.generate(prompt, max_new_tokens=100)
            return reformulated.strip() if reformulated.strip() else query
        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}")
            return query
    
    async def _generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generate answer from contexts
        
        Args:
            query: User question
            contexts: Retrieved contexts
        
        Returns:
            Generated answer
        """
        if hasattr(self.rag, 'generate_answer'):
            # Build context dicts for compatibility
            context_dicts = [{'content': c} for c in contexts]
            return self.rag.generate_answer(query, context_dicts)
        
        # Fallback generation
        context_str = '\n'.join(contexts)
        prompt = f"""Context: {context_str}

Question: {query}

Answer:"""
        
        if self.llm:
            return self.llm.generate(prompt, max_new_tokens=200)
        else:
            return f"Based on the context: {contexts[0][:200]}..." if contexts else "No answer available"
    
    async def _critique_answer(self, query: str, answer: str, 
                              contexts: List[str]) -> Dict[str, Any]:
        """
        Critique the generated answer
        
        Args:
            query: Original question
            answer: Generated answer
            contexts: Retrieved contexts
        
        Returns:
            Critique dictionary
        """
        critique = {
            'is_satisfactory': False,
            'grounded': False,
            'addresses_question': False,
            'errors': [],
            'suggestions': []
        }
        
        if not self.llm:
            # Simple heuristic critique
            critique['grounded'] = any(
                context_snippet in answer 
                for context in contexts 
                for context_snippet in context.split('.')[:2]
            )
            critique['addresses_question'] = len(answer) > 50
            critique['is_satisfactory'] = critique['grounded'] and critique['addresses_question']
            return critique
        
        prompt = f"""Question: {query}
Answer: {answer}
Contexts: {' '.join(contexts[:2])}

Critique this answer:
1. Is it factually grounded in the contexts? (YES/NO)
2. Does it fully address the question? (YES/NO)
3. List any errors or issues:
4. Suggestions for improvement:

Format: 
Grounded: [YES/NO]
Addresses: [YES/NO]
Errors: [list]
Suggestions: [list]"""
        
        try:
            critique_text = self.llm.generate(prompt, max_new_tokens=200)
            
            # Parse critique
            lines = critique_text.lower().split('\n')
            for line in lines:
                if 'grounded:' in line:
                    critique['grounded'] = 'yes' in line
                elif 'addresses:' in line:
                    critique['addresses_question'] = 'yes' in line
                elif 'errors:' in line:
                    critique['errors'].append(line.split(':', 1)[1].strip())
                elif 'suggestions:' in line:
                    critique['suggestions'].append(line.split(':', 1)[1].strip())
            
            critique['is_satisfactory'] = (
                critique['grounded'] and 
                critique['addresses_question'] and 
                len(critique['errors']) == 0
            )
            
        except Exception as e:
            logger.warning(f"Answer critique failed: {e}")
        
        return critique
    
    async def _refine_answer(self, query: str, answer: str, 
                            critique: Dict[str, Any], contexts: List[str]) -> str:
        """
        Refine answer based on critique
        
        Args:
            query: Original question
            answer: Current answer
            critique: Critique results
            contexts: Retrieved contexts
        
        Returns:
            Refined answer
        """
        if not self.llm:
            return answer
        
        # Build refinement prompt
        issues = '\n'.join(critique.get('errors', []))
        suggestions = '\n'.join(critique.get('suggestions', []))
        
        prompt = f"""Original question: {query}
Current answer: {answer}

Issues identified:
{issues}

Suggestions:
{suggestions}

Context:
{' '.join(contexts[:2])}

Provide an improved answer that addresses the issues:"""
        
        try:
            refined = self.llm.generate(prompt, max_new_tokens=300)
            return refined.strip() if refined.strip() else answer
        except Exception as e:
            logger.warning(f"Answer refinement failed: {e}")
            return answer
    
    def _calculate_confidence(self, retrieval_score: float, 
                            critique: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score
        
        Args:
            retrieval_score: Retrieval quality score
            critique: Critique results
        
        Returns:
            Confidence score (0-1)
        """
        # Weight different factors
        weights = {
            'retrieval': 0.3,
            'grounded': 0.3,
            'addresses': 0.2,
            'no_errors': 0.2
        }
        
        scores = {
            'retrieval': retrieval_score,
            'grounded': 1.0 if critique.get('grounded') else 0.0,
            'addresses': 1.0 if critique.get('addresses_question') else 0.0,
            'no_errors': 1.0 if len(critique.get('errors', [])) == 0 else 0.0
        }
        
        confidence = sum(scores[k] * weights[k] for k in weights)
        return max(0.0, min(1.0, confidence))