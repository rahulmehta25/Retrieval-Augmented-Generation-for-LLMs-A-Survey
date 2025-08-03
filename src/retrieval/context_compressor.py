"""
Context Compression Module

This module implements various strategies for compressing and managing context
to fit within LLM context windows and improve memory efficiency.
"""

import abc
from typing import List, Dict, Any
from transformers import pipeline

class ContextCompressor(abc.ABC):
    """
    Abstract base class for context compression strategies.
    """
    @abc.abstractmethod
    def compress(self, context: List[str], query: str, max_tokens: int = 1000) -> List[str]:
        """
        Compresses context to fit within specified token limit.
        
        Args:
            context: List of context strings to compress
            query: The user query for context relevance
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Compressed list of context strings
        """
        pass

class ExtractiveContextCompressor(ContextCompressor):
    """
    Extractive compression that selects the most relevant sentences/segments
    from the context based on relevance to the query.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-distilled-squad"):
        try:
            self.model = pipeline("question-answering", model=model_name)
        except Exception as e:
            print(f"Warning: Could not load {model_name}, using fallback compression")
            self.model = None
    
    def compress(self, context: List[str], query: str, max_tokens: int = 1000) -> List[str]:
        """
        Compresses context by extracting the most relevant segments.
        """
        if not context:
            return []
        
        # Simple fallback compression if model is not available
        if self.model is None:
            return self._simple_compression(context, max_tokens)
        
        try:
            # Score each context segment for relevance
            scored_segments = []
            for segment in context:
                if len(segment.strip()) < 10:  # Skip very short segments
                    continue
                    
                # Use the model to get relevance score
                result = self.model(question=query, context=segment)
                score = result.get('score', 0.0)
                scored_segments.append((segment, score))
            
            # Sort by relevance score
            scored_segments.sort(key=lambda x: x[1], reverse=True)
            
            # Select segments until we reach max_tokens
            compressed_context = []
            current_tokens = 0
            
            for segment, score in scored_segments:
                # Rough token estimation (4 chars per token)
                segment_tokens = len(segment) // 4
                
                if current_tokens + segment_tokens <= max_tokens:
                    compressed_context.append(segment)
                    current_tokens += segment_tokens
                else:
                    break
            
            return compressed_context
            
        except Exception as e:
            print(f"Warning: Extractive compression failed: {e}. Using fallback.")
            return self._simple_compression(context, max_tokens)
    
    def _simple_compression(self, context: List[str], max_tokens: int) -> List[str]:
        """
        Simple compression that takes the first segments until token limit.
        """
        compressed_context = []
        current_tokens = 0
        
        for segment in context:
            segment_tokens = len(segment) // 4  # Rough estimation
            if current_tokens + segment_tokens <= max_tokens:
                compressed_context.append(segment)
                current_tokens += segment_tokens
            else:
                break
        
        return compressed_context

class AbstractiveContextCompressor(ContextCompressor):
    """
    Abstractive compression that generates summaries of context segments.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        try:
            self.model = pipeline("summarization", model=model_name)
        except Exception as e:
            print(f"Warning: Could not load {model_name}, using fallback compression")
            self.model = None
    
    def compress(self, context: List[str], query: str, max_tokens: int = 1000) -> List[str]:
        """
        Compresses context by generating summaries of segments.
        """
        if not context or self.model is None:
            return self._simple_compression(context, max_tokens)
        
        try:
            compressed_context = []
            current_tokens = 0
            
            for segment in context:
                if len(segment.strip()) < 50:  # Skip very short segments
                    continue
                
                # Generate summary
                summary = self.model(segment, max_length=100, min_length=30, do_sample=False)
                summarized_text = summary[0]['summary_text']
                
                # Check token limit
                summary_tokens = len(summarized_text) // 4
                if current_tokens + summary_tokens <= max_tokens:
                    compressed_context.append(summarized_text)
                    current_tokens += summary_tokens
                else:
                    break
            
            return compressed_context
            
        except Exception as e:
            print(f"Warning: Abstractive compression failed: {e}. Using fallback.")
            return self._simple_compression(context, max_tokens)
    
    def _simple_compression(self, context: List[str], max_tokens: int) -> List[str]:
        """
        Simple compression fallback.
        """
        return ExtractiveContextCompressor()._simple_compression(context, max_tokens)

class SlidingWindowContextCompressor(ContextCompressor):
    """
    Sliding window compression that maintains a fixed-size window of recent context.
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
    
    def compress(self, context: List[str], query: str, max_tokens: int = 1000) -> List[str]:
        """
        Compresses context using a sliding window approach.
        """
        if not context:
            return []
        
        # Take the most recent segments up to window size
        recent_context = context[-self.window_size:]
        
        # Further compress if needed to fit token limit
        compressed_context = []
        current_tokens = 0
        
        for segment in recent_context:
            segment_tokens = len(segment) // 4
            if current_tokens + segment_tokens <= max_tokens:
                compressed_context.append(segment)
                current_tokens += segment_tokens
            else:
                break
        
        return compressed_context

class HybridContextCompressor(ContextCompressor):
    """
    Hybrid compression that combines multiple strategies.
    """
    
    def __init__(self, extractive_compressor: ExtractiveContextCompressor = None,
                 abstractive_compressor: AbstractiveContextCompressor = None):
        self.extractive_compressor = extractive_compressor or ExtractiveContextCompressor()
        self.abstractive_compressor = abstractive_compressor or AbstractiveContextCompressor()
    
    def compress(self, context: List[str], query: str, max_tokens: int = 1000) -> List[str]:
        """
        Compresses context using a hybrid approach.
        """
        if not context:
            return []
        
        # First, use extractive compression to get relevant segments
        relevant_segments = self.extractive_compressor.compress(context, query, max_tokens)
        
        # If we still have too many tokens, use abstractive compression
        total_tokens = sum(len(seg) // 4 for seg in relevant_segments)
        
        if total_tokens > max_tokens:
            # Use abstractive compression on the most relevant segments
            return self.abstractive_compressor.compress(relevant_segments, query, max_tokens)
        
        return relevant_segments 