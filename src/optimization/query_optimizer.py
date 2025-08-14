"""
Advanced Query Optimization with multiple techniques
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedQuery:
    """Container for optimized query components"""
    original: str
    rewritten: Optional[str] = None
    expanded: List[str] = None
    decomposed: List[str] = None
    keywords: List[str] = None
    intent: str = None
    complexity: float = 0.0

class QueryOptimizer:
    """
    Production query optimizer with multiple techniques:
    1. Query understanding and intent classification
    2. Query rewriting for clarity
    3. Query expansion with synonyms
    4. Query decomposition for complex questions
    5. Keyword extraction
    """
    
    def __init__(self, llm_generator=None, embedder=None):
        """Initialize query optimizer"""
        self.llm = llm_generator
        self.embedder = embedder or SentenceTransformer('all-MiniLM-L6-v2')
        
        # Intent patterns
        self.intent_patterns = {
            'definition': r'what is|define|explain what|meaning of',
            'comparison': r'difference between|compare|versus|vs\.|better than',
            'process': r'how to|how does|steps to|process of',
            'reason': r'why|reason for|cause of|because',
            'list': r'list of|enumerate|what are all|types of',
            'example': r'example of|instance of|such as|for example',
            'evaluation': r'is it|should i|can i|would it',
            'factual': r'when|where|who|which|how many|how much'
        }
        
        # Query complexity indicators
        self.complexity_indicators = {
            'conjunctions': ['and', 'or', 'but', 'while', 'whereas'],
            'conditionals': ['if', 'when', 'unless', 'provided', 'assuming'],
            'comparisons': ['more than', 'less than', 'better', 'worse', 'versus'],
            'multi_part': ['first', 'second', 'then', 'finally', 'additionally']
        }
    
    def classify_intent(self, query: str) -> str:
        """Classify query intent"""
        query_lower = query.lower()
        
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                return intent
        
        return 'general'
    
    def assess_complexity(self, query: str) -> float:
        """Assess query complexity (0-1 scale)"""
        score = 0.0
        query_lower = query.lower()
        
        # Length factor
        word_count = len(query.split())
        score += min(word_count / 20, 0.3)  # Max 0.3 for length
        
        # Conjunction factor
        for conj in self.complexity_indicators['conjunctions']:
            if conj in query_lower:
                score += 0.1
        
        # Conditional factor
        for cond in self.complexity_indicators['conditionals']:
            if cond in query_lower:
                score += 0.15
        
        # Multi-part factor
        for part in self.complexity_indicators['multi_part']:
            if part in query_lower:
                score += 0.1
        
        # Question marks (multiple questions)
        score += query.count('?') * 0.1
        
        return min(score, 1.0)
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove stopwords
        stopwords = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', query.lower())
        
        # Filter stopwords and short words
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Extract noun phrases (simple approach)
        noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        keywords.extend([np.lower() for np in noun_phrases])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def rewrite_query(self, query: str) -> str:
        """Rewrite query for clarity"""
        if not self.llm:
            # Fallback to rule-based rewriting
            return self.rule_based_rewrite(query)
        
        prompt = f"""Rewrite this question to be clearer and more specific.
Keep the same meaning but improve clarity.

Original: {query}
Rewritten:"""
        
        try:
            rewritten = self.llm.generate(prompt, "")
            return rewritten.strip() if rewritten else query
        except:
            return self.rule_based_rewrite(query)
    
    def rule_based_rewrite(self, query: str) -> str:
        """Rule-based query rewriting"""
        rewritten = query
        
        # Expand contractions
        contractions = {
            "what's": "what is",
            "it's": "it is",
            "don't": "do not",
            "doesn't": "does not",
            "won't": "will not",
            "can't": "cannot",
            "wouldn't": "would not",
            "shouldn't": "should not"
        }
        
        for contraction, expansion in contractions.items():
            rewritten = rewritten.replace(contraction, expansion)
        
        # Fix common patterns
        rewritten = re.sub(r'\s+', ' ', rewritten)  # Multiple spaces
        rewritten = re.sub(r'\?+', '?', rewritten)  # Multiple question marks
        
        # Ensure question mark
        if not rewritten.endswith('?') and any(rewritten.lower().startswith(q) for q in ['what', 'why', 'how', 'when', 'where', 'who', 'which']):
            rewritten += '?'
        
        return rewritten
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with variations"""
        expanded = [query]
        keywords = self.extract_keywords(query)
        
        # Synonym expansion (simplified)
        synonyms = {
            'ai': ['artificial intelligence', 'machine intelligence'],
            'ml': ['machine learning', 'statistical learning'],
            'dl': ['deep learning', 'neural networks'],
            'nlp': ['natural language processing', 'text processing'],
            'rag': ['retrieval augmented generation', 'retrieval-augmented generation']
        }
        
        # Generate expansions
        for keyword in keywords:
            if keyword.lower() in synonyms:
                for syn in synonyms[keyword.lower()]:
                    expanded_query = query.replace(keyword, syn)
                    if expanded_query not in expanded:
                        expanded.append(expanded_query)
        
        return expanded[:3]  # Limit to 3 variations
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-questions"""
        complexity = self.assess_complexity(query)
        
        if complexity < 0.5:
            return [query]  # Simple query, no decomposition needed
        
        sub_questions = []
        
        # Split by conjunctions
        parts = re.split(r'\s+(?:and|but|also|furthermore)\s+', query, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            if len(part) > 10:  # Meaningful part
                # Ensure it's a complete question
                if not any(part.lower().startswith(q) for q in ['what', 'why', 'how', 'when', 'where', 'who', 'which']):
                    # Add question word based on context
                    if 'is' in part.lower() or 'are' in part.lower():
                        part = 'What ' + part
                    else:
                        part = 'How does ' + part
                
                if not part.endswith('?'):
                    part += '?'
                
                sub_questions.append(part)
        
        return sub_questions if sub_questions else [query]
    
    def optimize(self, query: str) -> OptimizedQuery:
        """
        Main optimization method
        
        Returns:
            OptimizedQuery object with all optimizations
        """
        
        # Create optimized query object
        optimized = OptimizedQuery(original=query)
        
        # Classify intent
        optimized.intent = self.classify_intent(query)
        
        # Assess complexity
        optimized.complexity = self.assess_complexity(query)
        
        # Extract keywords
        optimized.keywords = self.extract_keywords(query)
        
        # Rewrite for clarity
        optimized.rewritten = self.rewrite_query(query)
        
        # Expand with variations
        optimized.expanded = self.expand_query(query)
        
        # Decompose if complex
        if optimized.complexity > 0.5:
            optimized.decomposed = self.decompose_query(query)
        
        logger.info(f"Optimized query: intent={optimized.intent}, complexity={optimized.complexity:.2f}")
        
        return optimized

class HyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE) generator
    Creates hypothetical perfect answers for better retrieval
    """
    
    def __init__(self, llm_generator, embedder=None):
        self.llm = llm_generator
        self.embedder = embedder or SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_hypothetical_answer(self, query: str) -> str:
        """Generate a hypothetical perfect answer"""
        
        prompt = f"""Write a comprehensive answer to this question as if you were writing a textbook entry.
Include specific details and examples.

Question: {query}

Comprehensive Answer:"""
        
        try:
            hypothetical = self.llm.generate(prompt, "")
            return hypothetical if hypothetical else query
        except:
            # Fallback to template-based generation
            return self.template_based_hyde(query)
    
    def template_based_hyde(self, query: str) -> str:
        """Template-based hypothetical answer generation"""
        
        templates = {
            'definition': "{subject} is a {category} that {properties}. It is characterized by {features} and is used for {applications}.",
            'process': "The process of {subject} involves the following steps: First, {step1}. Next, {step2}. Then, {step3}. Finally, {result}.",
            'comparison': "{subject1} and {subject2} differ in several ways. {subject1} is {property1}, while {subject2} is {property2}. The main distinction is {difference}.",
            'reason': "{subject} occurs because {cause}. This is due to {explanation}. The underlying mechanism involves {process}."
        }
        
        # Simple keyword extraction
        keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        if not keywords:
            keywords = query.split()[:3]
        
        # Determine intent
        intent = 'definition'  # Default
        if 'how' in query.lower():
            intent = 'process'
        elif 'why' in query.lower():
            intent = 'reason'
        elif 'difference' in query.lower() or 'compare' in query.lower():
            intent = 'comparison'
        
        # Generate using template
        template = templates.get(intent, templates['definition'])
        
        # Simple filling (this is basic, can be improved)
        hypothetical = template
        for keyword in keywords:
            hypothetical = hypothetical.replace('{subject}', keyword, 1)
            hypothetical = hypothetical.replace('{subject1}', keyword, 1)
        
        # Fill remaining placeholders with generic text
        hypothetical = re.sub(r'\{[^}]+\}', 'relevant information', hypothetical)
        
        return f"{query} {hypothetical}"