"""
Advanced Semantic Chunking with Multiple Strategies
"""

import re
import nltk
import spacy
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Container for document chunk"""
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]
    chunk_type: str  # 'semantic', 'sentence', 'paragraph', 'sliding', 'recursive'
    embedding: Optional[np.ndarray] = None
    tokens: int = 0
    sentences: int = 0

class SemanticChunker:
    """
    Advanced chunking with semantic boundaries:
    - Semantic similarity-based splitting
    - Sentence boundary preservation
    - Paragraph structure awareness
    - Topic modeling for coherent chunks
    - Recursive splitting for optimal size
    - Context-aware overlapping
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize semantic chunker"""
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Sentence segmentation
        try:
            nltk.download('punkt', quiet=True)
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        except:
            self.sent_tokenizer = None
        
        # Embeddings for semantic similarity
        self.embedder = SentenceTransformer(embedding_model)
        
        # Patterns for structure detection
        self.heading_pattern = re.compile(
            r'^(?:#{1,6}\s+|(?:[A-Z][^.!?]*(?:[.!?](?:\s|$))?){1,3}$|^\d+\.?\s+[A-Z])',
            re.MULTILINE
        )
        
        self.list_pattern = re.compile(
            r'^(?:[•·▪▫◦‣⁃]\s+|\d+\.\s+|[a-z]\.\s+|[ivxIVX]+\.\s+)',
            re.MULTILINE
        )
    
    def chunk_by_sentences(
        self,
        text: str,
        target_size: Optional[int] = None
    ) -> List[Chunk]:
        """Chunk by sentence boundaries"""
        
        target_size = target_size or self.chunk_size
        
        # Get sentences
        if self.sent_tokenizer:
            sentences = self.sent_tokenizer.tokenize(text)
        else:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # Check if adding this sentence exceeds target
            if current_length + sentence_length > target_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    metadata={"method": "sentence"},
                    chunk_type="sentence",
                    tokens=current_length,
                    sentences=len(current_chunk)
                ))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sents = []
                    overlap_length = 0
                    for sent in reversed(current_chunk):
                        overlap_length += len(sent.split())
                        overlap_sents.insert(0, sent)
                        if overlap_length >= self.chunk_overlap:
                            break
                    current_chunk = overlap_sents
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
                    start_idx = start_idx + len(chunk_text) + 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=len(text),
                metadata={"method": "sentence"},
                chunk_type="sentence",
                tokens=current_length,
                sentences=len(current_chunk)
            ))
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[Chunk]:
        """Chunk by paragraph boundaries"""
        
        # Split by paragraph markers
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para.split())
            
            # Check if paragraph is too large
            if para_length > self.chunk_size:
                # Split large paragraph by sentences
                if current_chunk:
                    # Save current chunk
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_idx=start_idx,
                        end_idx=start_idx + len(chunk_text),
                        metadata={"method": "paragraph"},
                        chunk_type="paragraph",
                        tokens=current_length
                    ))
                    current_chunk = []
                    current_length = 0
                
                # Split large paragraph
                para_chunks = self.chunk_by_sentences(para, self.chunk_size)
                chunks.extend(para_chunks)
                start_idx = start_idx + len(para) + 2
            
            elif current_length + para_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    metadata={"method": "paragraph"},
                    chunk_type="paragraph",
                    tokens=current_length
                ))
                
                # Start new chunk
                current_chunk = [para]
                current_length = para_length
                start_idx = start_idx + len(chunk_text) + 2
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=len(text),
                metadata={"method": "paragraph"},
                chunk_type="paragraph",
                tokens=current_length
            ))
        
        return chunks
    
    def chunk_by_semantic_similarity(
        self,
        text: str,
        similarity_threshold: float = 0.5
    ) -> List[Chunk]:
        """Chunk based on semantic similarity between sentences"""
        
        # Get sentences
        if self.sent_tokenizer:
            sentences = self.sent_tokenizer.tokenize(text)
        else:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        
        if len(sentences) < 2:
            return [Chunk(
                text=text,
                start_idx=0,
                end_idx=len(text),
                metadata={"method": "semantic"},
                chunk_type="semantic"
            )]
        
        # Encode sentences
        embeddings = self.embedder.encode(sentences)
        
        # Calculate similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        # Find split points (low similarity)
        split_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < similarity_threshold:
                split_indices.append(i + 1)
        split_indices.append(len(sentences))
        
        # Create chunks
        chunks = []
        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]
            
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)
            
            # Check size and split if necessary
            if len(chunk_text.split()) > self.chunk_size:
                # Recursive split
                sub_chunks = self.chunk_by_sentences(chunk_text, self.chunk_size)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=sum(len(s) + 1 for s in sentences[:start]),
                    end_idx=sum(len(s) + 1 for s in sentences[:end]),
                    metadata={
                        "method": "semantic",
                        "avg_similarity": np.mean(similarities[start:end-1]) if start < end-1 else 1.0
                    },
                    chunk_type="semantic",
                    sentences=len(chunk_sentences)
                ))
        
        return chunks
    
    def chunk_by_topic_segments(self, text: str) -> List[Chunk]:
        """Chunk based on topic changes using TextTiling-like approach"""
        
        # Get sentences
        if self.sent_tokenizer:
            sentences = self.sent_tokenizer.tokenize(text)
        else:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        
        if len(sentences) < 3:
            return [Chunk(
                text=text,
                start_idx=0,
                end_idx=len(text),
                metadata={"method": "topic"},
                chunk_type="topic"
            )]
        
        # Create windows of sentences
        window_size = 3
        windows = []
        for i in range(len(sentences) - window_size + 1):
            window_text = " ".join(sentences[i:i+window_size])
            windows.append(window_text)
        
        # Encode windows
        if windows:
            embeddings = self.embedder.encode(windows)
            
            # Calculate cohesion scores
            cohesion_scores = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[i + 1].reshape(1, -1)
                )[0][0]
                cohesion_scores.append(sim)
            
            # Find valleys (topic boundaries)
            boundaries = [0]
            for i in range(1, len(cohesion_scores) - 1):
                if cohesion_scores[i] < cohesion_scores[i-1] and \
                   cohesion_scores[i] < cohesion_scores[i+1]:
                    boundaries.append(i + window_size // 2)
            boundaries.append(len(sentences))
            
            # Create chunks from boundaries
            chunks = []
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                
                chunk_text = " ".join(sentences[start:end])
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=sum(len(s) + 1 for s in sentences[:start]),
                    end_idx=sum(len(s) + 1 for s in sentences[:end]),
                    metadata={"method": "topic", "boundaries": boundaries},
                    chunk_type="topic",
                    sentences=end - start
                ))
        else:
            chunks = self.chunk_by_sentences(text)
        
        return chunks
    
    def chunk_with_sliding_window(
        self,
        text: str,
        window_size: Optional[int] = None,
        stride: Optional[int] = None
    ) -> List[Chunk]:
        """Sliding window chunking with overlap"""
        
        window_size = window_size or self.chunk_size
        stride = stride or (window_size - self.chunk_overlap)
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + window_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=i,
                end_idx=min(i + window_size, len(words)),
                metadata={
                    "method": "sliding",
                    "window_size": window_size,
                    "stride": stride
                },
                chunk_type="sliding",
                tokens=len(chunk_words)
            ))
            
            if i + window_size >= len(words):
                break
        
        return chunks
    
    def recursive_chunk(
        self,
        text: str,
        separators: Optional[List[str]] = None
    ) -> List[Chunk]:
        """Recursive chunking with multiple separators"""
        
        if separators is None:
            separators = ["\n\n\n", "\n\n", "\n", ". ", ", ", " "]
        
        def split_text(text: str, sep_idx: int = 0) -> List[str]:
            if sep_idx >= len(separators):
                # No more separators, split by chunk size
                words = text.split()
                return [" ".join(words[i:i+self.chunk_size]) 
                       for i in range(0, len(words), self.chunk_size)]
            
            separator = separators[sep_idx]
            parts = text.split(separator)
            
            result = []
            current = []
            current_length = 0
            
            for part in parts:
                part_length = len(part.split())
                
                if part_length > self.chunk_size:
                    # Part too large, split recursively
                    if current:
                        result.append(separator.join(current))
                        current = []
                        current_length = 0
                    
                    sub_parts = split_text(part, sep_idx + 1)
                    result.extend(sub_parts)
                
                elif current_length + part_length > self.chunk_size:
                    # Would exceed size, save current and start new
                    if current:
                        result.append(separator.join(current))
                    current = [part]
                    current_length = part_length
                
                else:
                    # Add to current chunk
                    current.append(part)
                    current_length += part_length
            
            if current:
                result.append(separator.join(current))
            
            return result
        
        # Perform recursive splitting
        chunk_texts = split_text(text)
        
        # Create chunk objects
        chunks = []
        start_idx = 0
        
        for chunk_text in chunk_texts:
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text),
                metadata={"method": "recursive", "separators": separators},
                chunk_type="recursive",
                tokens=len(chunk_text.split())
            ))
            start_idx += len(chunk_text) + 1
        
        return chunks
    
    def chunk_markdown(self, text: str) -> List[Chunk]:
        """Special chunking for Markdown documents"""
        
        # Parse markdown structure
        lines = text.split('\n')
        sections = []
        current_section = {
            'level': 0,
            'title': '',
            'content': [],
            'start_line': 0
        }
        
        for i, line in enumerate(lines):
            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save current section if it has content
                if current_section['content']:
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_section = {
                    'level': level,
                    'title': title,
                    'content': [],
                    'start_line': i
                }
            else:
                current_section['content'].append(line)
        
        # Add final section
        if current_section['content']:
            sections.append(current_section)
        
        # Create chunks from sections
        chunks = []
        for section in sections:
            section_text = '\n'.join(section['content']).strip()
            
            if not section_text:
                continue
            
            # Check if section is too large
            if len(section_text.split()) > self.chunk_size:
                # Split large section
                sub_chunks = self.chunk_by_paragraphs(section_text)
                for chunk in sub_chunks:
                    chunk.metadata['section_title'] = section['title']
                    chunk.metadata['section_level'] = section['level']
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    text=section_text,
                    start_idx=section['start_line'],
                    end_idx=section['start_line'] + len(section['content']),
                    metadata={
                        'method': 'markdown',
                        'section_title': section['title'],
                        'section_level': section['level']
                    },
                    chunk_type='markdown',
                    tokens=len(section_text.split())
                ))
        
        return chunks
    
    def chunk_code(self, text: str, language: str = 'python') -> List[Chunk]:
        """Special chunking for code files"""
        
        # Simple function/class detection for Python
        if language == 'python':
            pattern = re.compile(
                r'^(class\s+\w+.*?:|def\s+\w+.*?:)',
                re.MULTILINE
            )
            
            matches = list(pattern.finditer(text))
            
            if not matches:
                return self.chunk_by_sentences(text)
            
            chunks = []
            
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                
                chunk_text = text[start:end].strip()
                
                # Extract function/class name
                name_match = re.match(r'(?:class|def)\s+(\w+)', chunk_text)
                name = name_match.group(1) if name_match else 'unknown'
                
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=start,
                    end_idx=end,
                    metadata={
                        'method': 'code',
                        'language': language,
                        'name': name,
                        'type': 'class' if chunk_text.startswith('class') else 'function'
                    },
                    chunk_type='code',
                    tokens=len(chunk_text.split())
                ))
        
        else:
            # Fallback for other languages
            chunks = self.chunk_by_sentences(text)
        
        return chunks
    
    def smart_chunk(self, text: str, doc_type: Optional[str] = None) -> List[Chunk]:
        """Intelligent chunking based on document type and content"""
        
        # Detect document type if not provided
        if doc_type is None:
            if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
                doc_type = 'markdown'
            elif re.search(r'^(?:class|def)\s+\w+', text, re.MULTILINE):
                doc_type = 'code'
            elif len(re.findall(r'\n\n+', text)) > 5:
                doc_type = 'article'
            else:
                doc_type = 'general'
        
        # Choose chunking strategy
        if doc_type == 'markdown':
            chunks = self.chunk_markdown(text)
        elif doc_type == 'code':
            chunks = self.chunk_code(text)
        elif doc_type == 'article':
            chunks = self.chunk_by_topic_segments(text)
        else:
            # Use semantic chunking for general text
            chunks = self.chunk_by_semantic_similarity(text)
        
        # Add embeddings to chunks
        for chunk in chunks:
            chunk.embedding = self.embedder.encode(chunk.text)
        
        return chunks
    
    def get_optimal_chunks(
        self,
        text: str,
        strategies: Optional[List[str]] = None
    ) -> List[Chunk]:
        """Get optimal chunks using multiple strategies and scoring"""
        
        if strategies is None:
            strategies = ['semantic', 'paragraph', 'topic']
        
        all_chunks = {}
        
        # Apply different strategies
        if 'semantic' in strategies:
            all_chunks['semantic'] = self.chunk_by_semantic_similarity(text)
        if 'paragraph' in strategies:
            all_chunks['paragraph'] = self.chunk_by_paragraphs(text)
        if 'topic' in strategies:
            all_chunks['topic'] = self.chunk_by_topic_segments(text)
        if 'sentence' in strategies:
            all_chunks['sentence'] = self.chunk_by_sentences(text)
        
        # Score chunks based on quality metrics
        best_strategy = None
        best_score = -1
        
        for strategy, chunks in all_chunks.items():
            score = self._score_chunks(chunks)
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        logger.info(f"Selected {best_strategy} strategy with score {best_score:.2f}")
        
        return all_chunks[best_strategy]
    
    def _score_chunks(self, chunks: List[Chunk]) -> float:
        """Score chunk quality"""
        
        if not chunks:
            return 0
        
        # Calculate metrics
        sizes = [chunk.tokens for chunk in chunks if chunk.tokens > 0]
        
        if not sizes:
            sizes = [len(chunk.text.split()) for chunk in chunks]
        
        # Size consistency (lower variance is better)
        size_variance = np.var(sizes) if len(sizes) > 1 else 0
        size_score = 1 / (1 + size_variance / 1000)
        
        # Average size (closer to target is better)
        avg_size = np.mean(sizes)
        size_diff = abs(avg_size - self.chunk_size)
        target_score = 1 / (1 + size_diff / self.chunk_size)
        
        # Coverage (less overlap is better)
        total_text = sum(len(chunk.text) for chunk in chunks)
        original_length = chunks[-1].end_idx if chunks else 0
        coverage_score = min(original_length / total_text, 1.0) if total_text > 0 else 0
        
        # Combined score
        score = (size_score * 0.3 + target_score * 0.4 + coverage_score * 0.3)
        
        return score