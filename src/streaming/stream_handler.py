"""
Streaming Handler for RAG System

Implements streaming token generation for real-time responses.
"""

from typing import AsyncIterator, Dict, Any, List, Optional
import asyncio
import json
import logging
from dataclasses import dataclass
import aiohttp
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events"""
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_COMPLETE = "retrieval_complete"
    GENERATION_START = "generation_start"
    TOKEN = "token"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class StreamEvent:
    """Container for streaming events"""
    type: StreamEventType
    content: Optional[str] = None
    contexts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type.value,
            'content': self.content,
            'contexts': self.contexts,
            'metadata': self.metadata
        }


class StreamingRAG:
    """Streaming RAG implementation with async token generation"""
    
    def __init__(self, rag_system, llm_type: str = "ollama"):
        """
        Initialize streaming RAG
        
        Args:
            rag_system: Base RAG system
            llm_type: Type of LLM (ollama, openai, huggingface)
        """
        self.rag = rag_system
        self.llm_type = llm_type
        logger.info(f"Initialized StreamingRAG with {llm_type}")
    
    async def stream_generate(self, 
                             query: str, 
                             contexts: List[str]) -> AsyncIterator[str]:
        """
        Stream tokens from LLM generation
        
        Args:
            query: User question
            contexts: Retrieved contexts
        
        Yields:
            Generated tokens
        """
        
        if self.llm_type == "ollama":
            async for token in self._stream_ollama(query, contexts):
                yield token
        
        elif self.llm_type == "openai":
            async for token in self._stream_openai(query, contexts):
                yield token
        
        else:
            # Fallback to non-streaming
            answer = await self._generate_fallback(query, contexts)
            for word in answer.split():
                yield word + " "
                await asyncio.sleep(0.05)  # Simulate streaming
    
    async def _stream_ollama(self, query: str, contexts: List[str]) -> AsyncIterator[str]:
        """
        Stream from Ollama API
        
        Args:
            query: User question
            contexts: Retrieved contexts
        
        Yields:
            Generated tokens
        """
        prompt = self._format_prompt(query, contexts)
        
        async with aiohttp.ClientSession() as session:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "gemma:2b",
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            try:
                async with session.post(url, json=payload) as response:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if 'response' in data:
                                    yield data['response']
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"Ollama streaming error: {e}")
                yield f"Error: {str(e)}"
    
    async def _stream_openai(self, query: str, contexts: List[str]) -> AsyncIterator[str]:
        """
        Stream from OpenAI API
        
        Args:
            query: User question
            contexts: Retrieved contexts
        
        Yields:
            Generated tokens
        """
        try:
            import openai
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
                {"role": "user", "content": self._format_prompt(query, contexts)}
            ]
            
            stream = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=500
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except ImportError:
            logger.warning("OpenAI not installed, falling back to non-streaming")
            answer = await self._generate_fallback(query, contexts)
            yield answer
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield f"Error: {str(e)}"
    
    async def _generate_fallback(self, query: str, contexts: List[str]) -> str:
        """
        Fallback non-streaming generation
        
        Args:
            query: User question
            contexts: Retrieved contexts
        
        Returns:
            Generated answer
        """
        prompt = self._format_prompt(query, contexts)
        
        # Use the base RAG system's generator
        if hasattr(self.rag, 'generator'):
            return self.rag.generator.generate(prompt)
        else:
            return "Unable to generate response"
    
    def _format_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Format prompt for generation
        
        Args:
            query: User question
            contexts: Retrieved contexts
        
        Returns:
            Formatted prompt
        """
        context_str = "\n\n".join(contexts)
        
        prompt = f"""Based on the following context, please answer the question.

Context:
{context_str}

Question: {query}

Answer: """
        
        return prompt
    
    async def query_stream(self, query: str, k: int = 5) -> AsyncIterator[StreamEvent]:
        """
        Complete RAG pipeline with streaming
        
        Args:
            query: User question
            k: Number of contexts to retrieve
        
        Yields:
            Stream events
        """
        
        # Step 1: Start retrieval
        yield StreamEvent(
            type=StreamEventType.RETRIEVAL_START,
            metadata={'query': query, 'k': k}
        )
        
        # Step 2: Retrieve contexts
        try:
            contexts = await self._retrieve_async(query, k)
            
            yield StreamEvent(
                type=StreamEventType.RETRIEVAL_COMPLETE,
                contexts=contexts,
                metadata={'num_contexts': len(contexts)}
            )
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=f"Retrieval error: {str(e)}"
            )
            return
        
        # Step 3: Start generation
        yield StreamEvent(
            type=StreamEventType.GENERATION_START,
            metadata={'model': self.llm_type}
        )
        
        # Step 4: Stream generation
        try:
            async for token in self.stream_generate(query, contexts):
                yield StreamEvent(
                    type=StreamEventType.TOKEN,
                    content=token
                )
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=f"Generation error: {str(e)}"
            )
            return
        
        # Step 5: Complete
        yield StreamEvent(type=StreamEventType.COMPLETE)
    
    async def _retrieve_async(self, query: str, k: int = 5) -> List[str]:
        """
        Async wrapper for retrieval
        
        Args:
            query: User question
            k: Number of contexts
        
        Returns:
            Retrieved contexts
        """
        # Run synchronous retrieval in executor
        loop = asyncio.get_event_loop()
        
        def retrieve_sync():
            retrieved = self.rag.retrieve(query, k=k)
            return [doc.get('content', str(doc)) for doc in retrieved]
        
        contexts = await loop.run_in_executor(None, retrieve_sync)
        return contexts
    
    async def benchmark_streaming(self, queries: List[str]) -> Dict[str, Any]:
        """
        Benchmark streaming performance
        
        Args:
            queries: List of test queries
        
        Returns:
            Performance metrics
        """
        import time
        
        results = {
            'total_queries': len(queries),
            'avg_first_token_time': [],
            'avg_tokens_per_second': [],
            'total_time': []
        }
        
        for query in queries:
            start_time = time.time()
            first_token_time = None
            token_count = 0
            
            async for event in self.query_stream(query):
                if event.type == StreamEventType.TOKEN:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    token_count += 1
                elif event.type == StreamEventType.COMPLETE:
                    total_time = time.time() - start_time
                    
                    results['avg_first_token_time'].append(first_token_time)
                    results['total_time'].append(total_time)
                    
                    if total_time > 0:
                        results['avg_tokens_per_second'].append(token_count / total_time)
        
        # Calculate averages
        results['avg_first_token_time'] = sum(results['avg_first_token_time']) / len(results['avg_first_token_time'])
        results['avg_tokens_per_second'] = sum(results['avg_tokens_per_second']) / len(results['avg_tokens_per_second'])
        results['avg_total_time'] = sum(results['total_time']) / len(results['total_time'])
        
        return results