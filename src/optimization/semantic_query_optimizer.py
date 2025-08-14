"""
Advanced Semantic Query Optimizer with Deep NLP Analysis
"""

import re
import spacy
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import nltk
from nltk.corpus import wordnet
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SemanticQuery:
    """Enhanced query with semantic analysis"""
    original: str
    cleaned: str
    entities: List[Dict[str, str]] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    intent: str = "general"
    sentiment: float = 0.0
    complexity: float = 0.0
    keywords: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    semantic_expansions: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    context_required: bool = False
    temporal_focus: Optional[str] = None
    domain: Optional[str] = None

class SemanticQueryOptimizer:
    """
    Production-grade semantic query optimizer with:
    - Named Entity Recognition (NER)
    - Relation extraction
    - Intent classification with transformers
    - Semantic similarity expansion
    - Concept hierarchy understanding
    - Query decomposition
    - Context requirement detection
    """
    
    def __init__(self):
        """Initialize semantic components"""
        
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Sentence transformer for semantic similarity
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Intent classifier (using zero-shot classification)
        try:
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # CPU
            )
        except:
            logger.warning("Intent classifier not available")
            self.intent_classifier = None
        
        # Download NLTK data
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
        
        # Define intent categories
        self.intent_labels = [
            "definition",
            "explanation", 
            "comparison",
            "procedure",
            "causation",
            "example",
            "evaluation",
            "factual",
            "opinion",
            "recommendation"
        ]
        
        # Domain keywords
        self.domain_keywords = {
            "technical": ["code", "programming", "software", "algorithm", "function", "debug", "api"],
            "scientific": ["research", "hypothesis", "experiment", "theory", "analysis", "data"],
            "business": ["revenue", "profit", "market", "strategy", "customer", "sales"],
            "medical": ["diagnosis", "treatment", "symptoms", "patient", "medicine", "disease"],
            "educational": ["learn", "teach", "understand", "explain", "study", "course"]
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities with enhanced recognition"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Also extract noun phrases as potential entities
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Multi-word phrases
                entities.append({
                    "text": chunk.text,
                    "label": "NOUN_PHRASE",
                    "start": chunk.start_char,
                    "end": chunk.end_char
                })
        
        return entities
    
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract semantic relations between entities"""
        doc = self.nlp(text)
        relations = []
        
        # Extract subject-verb-object triplets
        for token in doc:
            if token.dep_ == "ROOT":
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                    elif child.dep_ in ["dobj", "pobj", "attr"]:
                        obj = child.text
                
                if subject and obj:
                    relations.append((subject, token.lemma_, obj))
        
        # Extract prepositional relations
        for token in doc:
            if token.dep_ == "prep":
                if token.head.pos_ in ["NOUN", "PROPN"] and len(list(token.children)) > 0:
                    obj = list(token.children)[0]
                    if obj.pos_ in ["NOUN", "PROPN"]:
                        relations.append((token.head.text, token.text, obj.text))
        
        return relations
    
    def classify_intent(self, query: str) -> str:
        """Classify query intent using transformers"""
        
        if self.intent_classifier:
            try:
                result = self.intent_classifier(
                    query,
                    candidate_labels=self.intent_labels,
                    multi_label=False
                )
                return result['labels'][0]
            except:
                pass
        
        # Fallback to rule-based
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what is", "define", "meaning"]):
            return "definition"
        elif any(word in query_lower for word in ["how", "steps", "process"]):
            return "procedure"
        elif any(word in query_lower for word in ["why", "because", "reason"]):
            return "causation"
        elif any(word in query_lower for word in ["compare", "difference", "versus"]):
            return "comparison"
        elif any(word in query_lower for word in ["example", "instance", "such as"]):
            return "example"
        elif any(word in query_lower for word in ["should", "recommend", "best"]):
            return "recommendation"
        elif any(word in query_lower for word in ["explain", "describe", "tell"]):
            return "explanation"
        
        return "general"
    
    def extract_concepts(self, query: str) -> List[str]:
        """Extract high-level concepts using WordNet"""
        doc = self.nlp(query)
        concepts = set()
        
        for token in doc:
            if token.pos_ in ["NOUN", "VERB"] and not token.is_stop:
                # Get WordNet synsets
                synsets = wordnet.synsets(token.lemma_)
                
                for syn in synsets[:2]:  # Top 2 senses
                    # Get hypernyms (more general concepts)
                    for hypernym in syn.hypernyms():
                        concept = hypernym.lemmas()[0].name().replace('_', ' ')
                        concepts.add(concept)
        
        return list(concepts)
    
    def semantic_expansion(self, query: str) -> List[str]:
        """Expand query using semantic similarity"""
        expansions = [query]
        
        # Extract key terms
        doc = self.nlp(query)
        key_terms = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"] and not token.is_stop]
        
        # Expand using WordNet synonyms
        for term in key_terms:
            synsets = wordnet.synsets(term)
            for syn in synsets[:1]:  # Top sense only
                for lemma in syn.lemmas()[:3]:  # Top 3 synonyms
                    if lemma.name() != term and '_' not in lemma.name():
                        expanded = query.replace(term, lemma.name())
                        if expanded not in expansions:
                            expansions.append(expanded)
        
        # Generate paraphrases using templates
        if "what is" in query.lower():
            expansions.append(query.replace("what is", "define"))
            expansions.append(query.replace("what is", "explain"))
        
        if "how to" in query.lower():
            expansions.append(query.replace("how to", "steps to"))
            expansions.append(query.replace("how to", "process of"))
        
        return expansions[:5]  # Limit expansions
    
    def decompose_complex_query(self, query: str) -> List[str]:
        """Decompose complex queries into sub-queries"""
        doc = self.nlp(query)
        sub_queries = []
        
        # Split by conjunctions and punctuation
        splits = []
        current = []
        
        for token in doc:
            if token.text in [",", ";", "and", "but", "or", "also", "then"]:
                if current:
                    splits.append(" ".join([t.text for t in current]))
                    current = []
            else:
                current.append(token)
        
        if current:
            splits.append(" ".join([t.text for t in current]))
        
        # Process each split
        for split in splits:
            split = split.strip()
            if len(split.split()) > 3:  # Meaningful sub-query
                # Ensure it's a complete question
                if not any(split.lower().startswith(q) for q in ["what", "how", "why", "when", "where", "who"]):
                    # Infer question type
                    if "is" in split or "are" in split:
                        split = "What " + split
                    else:
                        split = "Explain " + split
                
                if not split.endswith("?"):
                    split += "?"
                
                sub_queries.append(split)
        
        # If no decomposition, return original
        return sub_queries if sub_queries else [query]
    
    def detect_temporal_focus(self, query: str) -> Optional[str]:
        """Detect temporal aspects in query"""
        doc = self.nlp(query)
        
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                return ent.text
        
        # Check for temporal keywords
        temporal_keywords = {
            "recent": "recent",
            "latest": "latest",
            "current": "current",
            "historical": "historical",
            "future": "future",
            "past": "past"
        }
        
        query_lower = query.lower()
        for keyword, focus in temporal_keywords.items():
            if keyword in query_lower:
                return focus
        
        return None
    
    def identify_domain(self, query: str) -> Optional[str]:
        """Identify query domain"""
        query_lower = query.lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return None
    
    def requires_context(self, query: str) -> bool:
        """Determine if query requires context"""
        
        # Pronouns without clear antecedents
        pronouns = ["it", "this", "that", "these", "those", "they", "them"]
        doc = self.nlp(query)
        
        for token in doc:
            if token.text.lower() in pronouns and token.dep_ in ["nsubj", "dobj"]:
                return True
        
        # References to previous discussion
        context_phrases = ["mentioned", "discussed", "above", "previous", "earlier", "before"]
        query_lower = query.lower()
        
        return any(phrase in query_lower for phrase in context_phrases)
    
    def calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score"""
        doc = self.nlp(query)
        
        # Factors contributing to complexity
        complexity = 0.0
        
        # Length factor
        word_count = len([token for token in doc if not token.is_punct])
        complexity += min(word_count / 20, 0.3)
        
        # Syntactic complexity
        depth = 0
        for token in doc:
            ancestors = list(token.ancestors)
            depth = max(depth, len(ancestors))
        complexity += min(depth / 5, 0.2)
        
        # Number of clauses
        clauses = len([token for token in doc if token.dep_ in ["ROOT", "advcl", "relcl", "ccomp"]])
        complexity += min(clauses / 3, 0.2)
        
        # Named entities
        entities = len(doc.ents)
        complexity += min(entities / 5, 0.15)
        
        # Conjunctions
        conjunctions = len([token for token in doc if token.pos_ == "CCONJ"])
        complexity += min(conjunctions / 2, 0.15)
        
        return min(complexity, 1.0)
    
    def optimize(self, query: str) -> SemanticQuery:
        """
        Main optimization method with full semantic analysis
        """
        
        # Clean query
        cleaned = re.sub(r'\s+', ' ', query).strip()
        
        # Create semantic query object
        semantic_query = SemanticQuery(
            original=query,
            cleaned=cleaned
        )
        
        # Extract entities
        semantic_query.entities = self.extract_entities(cleaned)
        
        # Extract relations
        semantic_query.relations = self.extract_relations(cleaned)
        
        # Classify intent
        semantic_query.intent = self.classify_intent(cleaned)
        
        # Calculate complexity
        semantic_query.complexity = self.calculate_complexity(cleaned)
        
        # Extract concepts
        semantic_query.concepts = self.extract_concepts(cleaned)
        
        # Semantic expansion
        semantic_query.semantic_expansions = self.semantic_expansion(cleaned)
        
        # Decompose if complex
        if semantic_query.complexity > 0.6:
            semantic_query.sub_queries = self.decompose_complex_query(cleaned)
        
        # Check context requirement
        semantic_query.context_required = self.requires_context(cleaned)
        
        # Detect temporal focus
        semantic_query.temporal_focus = self.detect_temporal_focus(cleaned)
        
        # Identify domain
        semantic_query.domain = self.identify_domain(cleaned)
        
        # Extract keywords (important terms)
        doc = self.nlp(cleaned)
        semantic_query.keywords = [
            token.lemma_ for token in doc 
            if token.pos_ in ["NOUN", "VERB", "PROPN"] and not token.is_stop
        ]
        
        logger.info(f"Semantic optimization complete: intent={semantic_query.intent}, "
                   f"complexity={semantic_query.complexity:.2f}, domain={semantic_query.domain}")
        
        return semantic_query

class QueryRewriter:
    """Advanced query rewriting with multiple strategies"""
    
    def __init__(self, embedder=None):
        self.embedder = embedder or SentenceTransformer('all-MiniLM-L6-v2')
        self.optimizer = SemanticQueryOptimizer()
    
    def rewrite_for_clarity(self, query: str) -> str:
        """Rewrite query for maximum clarity"""
        
        # Semantic analysis
        semantic = self.optimizer.optimize(query)
        
        # Build clearer query based on intent
        if semantic.intent == "definition":
            if semantic.entities:
                main_entity = semantic.entities[0]["text"]
                return f"What is the definition of {main_entity}?"
        
        elif semantic.intent == "comparison":
            if len(semantic.entities) >= 2:
                return f"What are the differences between {semantic.entities[0]['text']} and {semantic.entities[1]['text']}?"
        
        elif semantic.intent == "procedure":
            if semantic.keywords:
                return f"What are the steps to {' '.join(semantic.keywords[:3])}?"
        
        # Default: clean and structured
        return semantic.cleaned
    
    def generate_hypothetical_document(self, query: str) -> str:
        """Generate HyDE (Hypothetical Document Embeddings)"""
        
        semantic = self.optimizer.optimize(query)
        
        # Template based on intent
        templates = {
            "definition": "{entity} is defined as a {concept} that {relation}. "
                         "It is characterized by {features} and is primarily used for {purpose}.",
            
            "explanation": "To understand {topic}, we need to consider {aspects}. "
                          "The main principle is {principle}. This works by {mechanism}. "
                          "The key components are {components}.",
            
            "procedure": "The process of {action} involves several steps: "
                        "1. {step1} 2. {step2} 3. {step3}. "
                        "The expected outcome is {outcome}.",
            
            "comparison": "{entity1} and {entity2} can be compared on several dimensions. "
                         "{entity1} is characterized by {feature1}, while {entity2} has {feature2}. "
                         "The main difference is {difference}."
        }
        
        template = templates.get(semantic.intent, templates["explanation"])
        
        # Fill template with extracted information
        hypothetical = template
        
        # Replace with entities
        for i, entity in enumerate(semantic.entities[:3]):
            hypothetical = hypothetical.replace(f"{{entity{i+1}}}", entity["text"])
            hypothetical = hypothetical.replace("{entity}", entity["text"], 1)
            hypothetical = hypothetical.replace("{topic}", entity["text"], 1)
        
        # Replace with keywords
        for i, keyword in enumerate(semantic.keywords[:5]):
            hypothetical = hypothetical.replace(f"{{keyword{i+1}}}", keyword)
            hypothetical = hypothetical.replace("{action}", keyword, 1)
        
        # Replace with concepts
        for concept in semantic.concepts[:2]:
            hypothetical = hypothetical.replace("{concept}", concept, 1)
        
        # Fill remaining placeholders
        hypothetical = re.sub(r'\{[^}]+\}', 'relevant information', hypothetical)
        
        return f"{query}\n\n{hypothetical}"