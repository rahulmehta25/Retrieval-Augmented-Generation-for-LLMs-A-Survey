"""
Advanced Context Compression and Summarization
"""

import re
import nltk
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer
import networkx as nx
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompressedContext:
    """Container for compressed context"""
    original_text: str
    compressed_text: str
    compression_ratio: float
    method: str
    relevance_score: float
    key_sentences: List[str]
    removed_sentences: List[str]
    metadata: Dict[str, Any]

class AdvancedContextCompressor:
    """
    Production context compression with:
    - Extractive summarization
    - Abstractive summarization
    - Query-focused compression
    - Redundancy removal
    - Information density optimization
    - Token budget management
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        summarization_model: Optional[str] = "facebook/bart-large-cnn",
        max_tokens: int = 2048
    ):
        """Initialize context compressor"""
        
        self.max_tokens = max_tokens
        
        # Embedding model for similarity
        self.embedder = SentenceTransformer(embedding_model)
        
        # Tokenizer for counting tokens
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except:
            self.tokenizer = None
        
        # Summarization model
        try:
            self.summarizer = pipeline(
                "summarization",
                model=summarization_model,
                device=-1  # CPU
            )
        except:
            logger.warning("Summarization model not available")
            self.summarizer = None
        
        # Download NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: 1 token ≈ 0.75 words
            return int(len(text.split()) * 1.33)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to simple splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def query_focused_compression(
        self,
        contexts: List[str],
        query: str,
        target_tokens: Optional[int] = None
    ) -> CompressedContext:
        """Compress contexts focusing on query relevance"""
        
        target_tokens = target_tokens or self.max_tokens
        
        # Combine contexts
        full_text = "\n\n".join(contexts)
        original_tokens = self.count_tokens(full_text)
        
        if original_tokens <= target_tokens:
            # No compression needed
            return CompressedContext(
                original_text=full_text,
                compressed_text=full_text,
                compression_ratio=1.0,
                method="none",
                relevance_score=1.0,
                key_sentences=[],
                removed_sentences=[],
                metadata={"original_tokens": original_tokens}
            )
        
        # Extract all sentences
        all_sentences = []
        for context in contexts:
            sentences = self.extract_sentences(context)
            all_sentences.extend(sentences)
        
        # Score sentences by relevance to query
        query_embedding = self.embedder.encode(query)
        sentence_scores = []
        
        for sentence in all_sentences:
            # Calculate relevance
            sent_embedding = self.embedder.encode(sentence)
            relevance = cosine_similarity(
                query_embedding.reshape(1, -1),
                sent_embedding.reshape(1, -1)
            )[0][0]
            
            # Calculate importance (length, position, keywords)
            importance = self._calculate_sentence_importance(sentence, all_sentences)
            
            # Combined score
            score = relevance * 0.7 + importance * 0.3
            sentence_scores.append((sentence, score, relevance))
        
        # Sort by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select sentences within token budget
        selected_sentences = []
        removed_sentences = []
        current_tokens = 0
        
        for sentence, score, relevance in sentence_scores:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= target_tokens:
                selected_sentences.append((sentence, score))
                current_tokens += sentence_tokens
            else:
                removed_sentences.append(sentence)
        
        # Reorder selected sentences to maintain flow
        selected_sentences = self._reorder_sentences(
            selected_sentences,
            all_sentences
        )
        
        # Create compressed text
        compressed_text = " ".join([s for s, _ in selected_sentences])
        
        # Calculate metrics
        compression_ratio = current_tokens / original_tokens
        avg_relevance = np.mean([r for _, _, r in sentence_scores[:len(selected_sentences)]])
        
        return CompressedContext(
            original_text=full_text,
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            method="query_focused",
            relevance_score=avg_relevance,
            key_sentences=[s for s, _ in selected_sentences[:5]],
            removed_sentences=removed_sentences[:5],
            metadata={
                "original_tokens": original_tokens,
                "compressed_tokens": current_tokens,
                "sentences_kept": len(selected_sentences),
                "sentences_removed": len(removed_sentences)
            }
        )
    
    def _calculate_sentence_importance(
        self,
        sentence: str,
        all_sentences: List[str]
    ) -> float:
        """Calculate sentence importance"""
        
        importance = 0.0
        
        # Length factor (not too short, not too long)
        words = sentence.split()
        word_count = len(words)
        if 5 <= word_count <= 30:
            importance += 0.3
        elif word_count > 30:
            importance += 0.1
        
        # Position factor (beginning and end are important)
        position = all_sentences.index(sentence) if sentence in all_sentences else 0
        if position < 3:
            importance += 0.2
        elif position >= len(all_sentences) - 3:
            importance += 0.15
        
        # Keyword density
        important_keywords = ['important', 'significant', 'key', 'main', 'primary',
                             'conclusion', 'summary', 'result', 'finding']
        keyword_count = sum(1 for word in words if word.lower() in important_keywords)
        importance += min(keyword_count * 0.1, 0.3)
        
        # Contains numbers or statistics
        if re.search(r'\d+', sentence):
            importance += 0.1
        
        # Question or definition
        if sentence.endswith('?') or ':' in sentence:
            importance += 0.1
        
        return min(importance, 1.0)
    
    def _reorder_sentences(
        self,
        selected_sentences: List[Tuple[str, float]],
        original_order: List[str]
    ) -> List[Tuple[str, float]]:
        """Reorder sentences to maintain coherence"""
        
        # Create mapping of original positions
        position_map = {sent: i for i, sent in enumerate(original_order)}
        
        # Sort selected sentences by original position
        reordered = sorted(
            selected_sentences,
            key=lambda x: position_map.get(x[0], float('inf'))
        )
        
        return reordered
    
    def remove_redundancy(
        self,
        contexts: List[str],
        similarity_threshold: float = 0.9
    ) -> CompressedContext:
        """Remove redundant information from contexts"""
        
        full_text = "\n\n".join(contexts)
        sentences = self.extract_sentences(full_text)
        
        if len(sentences) < 2:
            return CompressedContext(
                original_text=full_text,
                compressed_text=full_text,
                compression_ratio=1.0,
                method="redundancy_removal",
                relevance_score=1.0,
                key_sentences=[],
                removed_sentences=[],
                metadata={}
            )
        
        # Encode all sentences
        embeddings = self.embedder.encode(sentences)
        
        # Find redundant sentences
        kept_indices = [0]  # Keep first sentence
        removed_sentences = []
        
        for i in range(1, len(sentences)):
            is_redundant = False
            
            for j in kept_indices:
                similarity = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                
                if similarity > similarity_threshold:
                    is_redundant = True
                    removed_sentences.append(sentences[i])
                    break
            
            if not is_redundant:
                kept_indices.append(i)
        
        # Create compressed text
        kept_sentences = [sentences[i] for i in kept_indices]
        compressed_text = " ".join(kept_sentences)
        
        # Calculate compression ratio
        original_tokens = self.count_tokens(full_text)
        compressed_tokens = self.count_tokens(compressed_text)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        return CompressedContext(
            original_text=full_text,
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            method="redundancy_removal",
            relevance_score=1.0,
            key_sentences=kept_sentences[:5],
            removed_sentences=removed_sentences[:5],
            metadata={
                "original_sentences": len(sentences),
                "kept_sentences": len(kept_sentences),
                "removed_redundant": len(removed_sentences)
            }
        )
    
    def extractive_summarization(
        self,
        text: str,
        num_sentences: int = 5,
        use_textrank: bool = True
    ) -> CompressedContext:
        """Extract key sentences using TextRank or similar"""
        
        sentences = self.extract_sentences(text)
        
        if len(sentences) <= num_sentences:
            return CompressedContext(
                original_text=text,
                compressed_text=text,
                compression_ratio=1.0,
                method="extractive",
                relevance_score=1.0,
                key_sentences=sentences,
                removed_sentences=[],
                metadata={}
            )
        
        if use_textrank:
            # TextRank algorithm
            embeddings = self.embedder.encode(sentences)
            
            # Build similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Create graph
            graph = nx.from_numpy_array(similarity_matrix)
            
            # Calculate PageRank scores
            scores = nx.pagerank(graph)
            
            # Rank sentences
            ranked_sentences = [(sentences[i], score) 
                              for i, score in scores.items()]
            ranked_sentences.sort(key=lambda x: x[1], reverse=True)
        else:
            # Simple frequency-based ranking
            word_freq = Counter()
            for sentence in sentences:
                words = sentence.lower().split()
                words = [w for w in words if w not in self.stopwords]
                word_freq.update(words)
            
            # Score sentences
            ranked_sentences = []
            for sentence in sentences:
                words = sentence.lower().split()
                score = sum(word_freq[w] for w in words if w not in self.stopwords)
                score = score / len(words) if words else 0
                ranked_sentences.append((sentence, score))
            
            ranked_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences
        selected = ranked_sentences[:num_sentences]
        removed = [s for s, _ in ranked_sentences[num_sentences:]]
        
        # Reorder by original position
        selected_sentences = [s for s, _ in selected]
        ordered_selected = []
        for sentence in sentences:
            if sentence in selected_sentences:
                ordered_selected.append(sentence)
        
        compressed_text = " ".join(ordered_selected)
        
        # Calculate metrics
        original_tokens = self.count_tokens(text)
        compressed_tokens = self.count_tokens(compressed_text)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        return CompressedContext(
            original_text=text,
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            method="extractive_textrank" if use_textrank else "extractive_frequency",
            relevance_score=np.mean([s for _, s in selected]),
            key_sentences=ordered_selected,
            removed_sentences=removed[:5],
            metadata={
                "algorithm": "textrank" if use_textrank else "frequency",
                "original_sentences": len(sentences),
                "selected_sentences": num_sentences
            }
        )
    
    def abstractive_summarization(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50
    ) -> CompressedContext:
        """Generate abstractive summary using transformer model"""
        
        if not self.summarizer:
            # Fallback to extractive
            return self.extractive_summarization(text, num_sentences=3)
        
        try:
            # Truncate input if too long
            max_input_length = 1024
            if self.count_tokens(text) > max_input_length:
                sentences = self.extract_sentences(text)
                truncated_text = ""
                current_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    if current_tokens + sentence_tokens <= max_input_length:
                        truncated_text += sentence + " "
                        current_tokens += sentence_tokens
                    else:
                        break
                
                text = truncated_text
            
            # Generate summary
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            compressed_text = summary[0]['summary_text']
            
            # Calculate metrics
            original_tokens = self.count_tokens(text)
            compressed_tokens = self.count_tokens(compressed_text)
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            
            return CompressedContext(
                original_text=text,
                compressed_text=compressed_text,
                compression_ratio=compression_ratio,
                method="abstractive",
                relevance_score=0.8,  # Assumed good relevance
                key_sentences=self.extract_sentences(compressed_text),
                removed_sentences=[],
                metadata={
                    "model": "BART",
                    "max_length": max_length,
                    "min_length": min_length
                }
            )
        
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            # Fallback to extractive
            return self.extractive_summarization(text, num_sentences=3)
    
    def hierarchical_compression(
        self,
        contexts: List[str],
        levels: int = 2
    ) -> List[CompressedContext]:
        """Multi-level hierarchical compression"""
        
        compressed_levels = []
        current_contexts = contexts
        
        for level in range(levels):
            # Compress each context
            level_compressed = []
            
            for context in current_contexts:
                if level == 0:
                    # First level: remove redundancy
                    compressed = self.remove_redundancy([context])
                else:
                    # Subsequent levels: extractive summarization
                    compressed = self.extractive_summarization(
                        context,
                        num_sentences=max(3, len(self.extract_sentences(context)) // 2)
                    )
                
                level_compressed.append(compressed)
            
            compressed_levels.append(level_compressed)
            
            # Prepare for next level
            current_contexts = [c.compressed_text for c in level_compressed]
        
        return compressed_levels[0]  # Return first level compression
    
    def adaptive_compression(
        self,
        contexts: List[str],
        query: str,
        target_tokens: Optional[int] = None
    ) -> CompressedContext:
        """Adaptive compression based on content and constraints"""
        
        target_tokens = target_tokens or self.max_tokens
        full_text = "\n\n".join(contexts)
        current_tokens = self.count_tokens(full_text)
        
        # No compression needed
        if current_tokens <= target_tokens:
            return CompressedContext(
                original_text=full_text,
                compressed_text=full_text,
                compression_ratio=1.0,
                method="none",
                relevance_score=1.0,
                key_sentences=[],
                removed_sentences=[],
                metadata={"tokens": current_tokens}
            )
        
        # Mild compression (< 50% reduction needed)
        if current_tokens <= target_tokens * 2:
            return self.remove_redundancy(contexts)
        
        # Moderate compression (50-75% reduction)
        if current_tokens <= target_tokens * 4:
            return self.query_focused_compression(contexts, query, target_tokens)
        
        # Heavy compression (> 75% reduction)
        # Use abstractive summarization for each context, then combine
        summaries = []
        for context in contexts:
            summary = self.abstractive_summarization(
                context,
                max_length=target_tokens // len(contexts)
            )
            summaries.append(summary.compressed_text)
        
        combined_text = "\n\n".join(summaries)
        
        return CompressedContext(
            original_text=full_text,
            compressed_text=combined_text,
            compression_ratio=self.count_tokens(combined_text) / current_tokens,
            method="adaptive_heavy",
            relevance_score=0.7,
            key_sentences=summaries[:3],
            removed_sentences=[],
            metadata={
                "original_tokens": current_tokens,
                "compression_level": "heavy",
                "strategy": "abstractive"
            }
        )