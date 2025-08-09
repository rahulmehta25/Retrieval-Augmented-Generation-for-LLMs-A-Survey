"""
RAGAS (Retrieval Augmented Generation Assessment) Metrics Implementation

This module implements the core RAGAS metrics for evaluating RAG systems:
- Faithfulness: How grounded the answer is in the retrieved contexts
- Answer Relevancy: How relevant the answer is to the question
- Context Relevancy: How relevant retrieved contexts are to the question
- Context Precision: Precision of context retrieval against ground truth
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.tokenize import sent_tokenize
import torch
import logging
import json
from dataclasses import dataclass

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGASScore:
    """Container for RAGAS evaluation scores"""
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
    context_precision: Optional[float] = None
    overall: Optional[float] = None
    
    def __post_init__(self):
        """Calculate overall score if not provided"""
        if self.overall is None:
            scores = [self.faithfulness, self.answer_relevancy, self.context_relevancy]
            if self.context_precision is not None:
                scores.append(self.context_precision)
            self.overall = np.mean(scores)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'faithfulness': self.faithfulness,
            'answer_relevancy': self.answer_relevancy,
            'context_relevancy': self.context_relevancy,
            'context_precision': self.context_precision,
            'overall': self.overall
        }


class RAGASEvaluator:
    """Implements RAGAS (Retrieval Augmented Generation Assessment) metrics"""
    
    def __init__(self, llm_generator=None, embedder_model: str = 'all-MiniLM-L6-v2',
                 cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize RAGAS evaluator
        
        Args:
            llm_generator: LLM for generating evaluation prompts
            embedder_model: Model for sentence embeddings
            cross_encoder_model: Model for cross-encoding
        """
        self.embedder = SentenceTransformer(embedder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.llm = llm_generator
        
        logger.info(f"Initialized RAGAS evaluator with embedder: {embedder_model}")
    
    def evaluate(self, question: str, answer: str, contexts: List[str],
                ground_truth: Optional[str] = None) -> RAGASScore:
        """
        Comprehensive RAGAS evaluation
        
        Args:
            question: The input question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
        
        Returns:
            RAGASScore with all metrics
        """
        faithfulness = self.faithfulness_score(answer, contexts)
        answer_relevancy = self.answer_relevancy_score(question, answer)
        context_relevancy = self.context_relevancy_score(question, contexts)
        
        context_precision = None
        if ground_truth:
            context_precision = self.context_precision_score(question, contexts, ground_truth)
        
        return RAGASScore(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_relevancy=context_relevancy,
            context_precision=context_precision
        )
    
    def faithfulness_score(self, answer: str, contexts: List[str]) -> float:
        """
        Measures how grounded the answer is in the retrieved contexts.
        Decomposes answer into statements and verifies each against context.
        
        Args:
            answer: Generated answer
            contexts: Retrieved contexts
        
        Returns:
            Faithfulness score (0-1)
        """
        if not answer or not contexts:
            return 0.0
        
        # Decompose answer into atomic statements
        statements = self._decompose_into_statements(answer)
        
        if not statements:
            return 0.0
        
        # Verify each statement against contexts
        verification_scores = []
        combined_context = ' '.join(contexts)
        
        for statement in statements:
            if self.llm:
                # Use LLM for verification
                prompt = f"""
                Context: {combined_context}
                Statement: {statement}
                
                Can this statement be directly inferred from the context? 
                Answer only YES or NO.
                """
                
                try:
                    response = self.llm.generate(prompt, max_new_tokens=10)
                    verification_scores.append(1.0 if 'YES' in response.upper() else 0.0)
                except Exception as e:
                    logger.warning(f"LLM verification failed: {e}")
                    # Fallback to embedding similarity
                    verification_scores.append(self._verify_statement_with_embedding(
                        statement, combined_context))
            else:
                # Use embedding similarity as fallback
                verification_scores.append(self._verify_statement_with_embedding(
                    statement, combined_context))
        
        return np.mean(verification_scores) if verification_scores else 0.0
    
    def answer_relevancy_score(self, question: str, answer: str) -> float:
        """
        Measures how relevant the answer is to the question.
        Generates potential questions from answer and compares to original.
        
        Args:
            question: Original question
            answer: Generated answer
        
        Returns:
            Answer relevancy score (0-1)
        """
        if not question or not answer:
            return 0.0
        
        if self.llm:
            # Generate potential questions from the answer
            prompt = f"""
            Given this answer: {answer}
            Generate 3 questions that this answer would appropriately address.
            List each question on a new line.
            """
            
            try:
                response = self.llm.generate(prompt, max_new_tokens=150)
                generated_questions = [q.strip() for q in response.split('\n') if q.strip()]
            except Exception as e:
                logger.warning(f"Question generation failed: {e}")
                # Fallback to direct similarity
                return self._calculate_semantic_similarity(question, answer)
        else:
            # Direct similarity as fallback
            return self._calculate_semantic_similarity(question, answer)
        
        if not generated_questions:
            return self._calculate_semantic_similarity(question, answer)
        
        # Calculate similarity between generated and original question
        orig_embedding = self.embedder.encode(question)
        gen_embeddings = self.embedder.encode(generated_questions)
        
        similarities = []
        for gen_emb in gen_embeddings:
            similarity = np.dot(orig_embedding, gen_emb) / (
                np.linalg.norm(orig_embedding) * np.linalg.norm(gen_emb)
            )
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def context_relevancy_score(self, question: str, contexts: List[str]) -> float:
        """
        Measures how relevant retrieved contexts are to the question.
        
        Args:
            question: The question
            contexts: Retrieved contexts
        
        Returns:
            Context relevancy score (0-1)
        """
        if not question or not contexts:
            return 0.0
        
        # Extract sentences from contexts
        all_sentences = []
        for context in contexts:
            try:
                sentences = sent_tokenize(context)
                all_sentences.extend(sentences)
            except Exception as e:
                logger.warning(f"Sentence tokenization failed: {e}")
                all_sentences.append(context)
        
        if not all_sentences:
            return 0.0
        
        # Score each sentence
        relevancy_scores = []
        
        if self.llm:
            for sentence in all_sentences[:10]:  # Limit to avoid too many LLM calls
                prompt = f"""
                Question: {question}
                Sentence: {sentence}
                
                Is this sentence relevant to answering the question?
                Score from 0 to 1 where 0 is completely irrelevant and 1 is highly relevant.
                Output only the number.
                """
                
                try:
                    score_str = self.llm.generate(prompt, max_new_tokens=10).strip()
                    score = float(score_str) if score_str.replace('.', '').isdigit() else 0.5
                    relevancy_scores.append(score)
                except Exception as e:
                    logger.warning(f"LLM scoring failed: {e}")
                    # Fallback to embedding similarity
                    relevancy_scores.append(self._calculate_semantic_similarity(question, sentence))
        else:
            # Use embedding similarity
            for sentence in all_sentences:
                relevancy_scores.append(self._calculate_semantic_similarity(question, sentence))
        
        return np.mean(relevancy_scores) if relevancy_scores else 0.0
    
    def context_precision_score(self, question: str, contexts: List[str], 
                               ground_truth: str) -> float:
        """
        Measures precision of context retrieval against ground truth.
        
        Args:
            question: The question
            contexts: Retrieved contexts
            ground_truth: Ground truth answer
        
        Returns:
            Context precision score (0-1)
        """
        if not contexts or not ground_truth:
            return 0.0
        
        # Rank contexts by relevance using cross-encoder
        pairs = [[question, ctx] for ctx in contexts]
        scores = self.cross_encoder.predict(pairs)
        
        # Sort contexts by score
        sorted_indices = np.argsort(scores)[::-1]
        
        # Check which contexts contain ground truth information
        relevant_positions = []
        for i, idx in enumerate(sorted_indices):
            if self._contains_ground_truth(contexts[idx], ground_truth):
                relevant_positions.append(i + 1)  # 1-indexed position
        
        # Calculate precision@k for each k
        if not relevant_positions:
            return 0.0
        
        precision_scores = []
        for k in range(1, len(contexts) + 1):
            relevant_at_k = sum(1 for pos in relevant_positions if pos <= k)
            precision_scores.append(relevant_at_k / k)
        
        return np.mean(precision_scores)
    
    def _decompose_into_statements(self, text: str) -> List[str]:
        """
        Decompose text into atomic statements
        
        Args:
            text: Text to decompose
        
        Returns:
            List of atomic statements
        """
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}")
            sentences = [text]
        
        statements = []
        
        if self.llm:
            for sentence in sentences:
                # Use LLM to break complex sentences into atomic statements
                prompt = f"""
                Break this sentence into simple, atomic statements:
                {sentence}
                
                List each statement on a new line.
                """
                
                try:
                    response = self.llm.generate(prompt, max_new_tokens=200)
                    atomic = response.strip().split('\n')
                    statements.extend([s.strip() for s in atomic if s.strip()])
                except Exception as e:
                    logger.warning(f"Statement decomposition failed: {e}")
                    statements.append(sentence)
        else:
            # Fallback to simple sentence splitting
            statements = sentences
        
        return statements
    
    def _verify_statement_with_embedding(self, statement: str, context: str) -> float:
        """
        Verify statement against context using embeddings
        
        Args:
            statement: Statement to verify
            context: Context to verify against
        
        Returns:
            Verification score (0-1)
        """
        statement_emb = self.embedder.encode(statement)
        context_emb = self.embedder.encode(context)
        
        similarity = np.dot(statement_emb, context_emb) / (
            np.linalg.norm(statement_emb) * np.linalg.norm(context_emb)
        )
        
        # Convert similarity to binary score with threshold
        return 1.0 if similarity > 0.7 else 0.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.embedder.encode(text1)
        emb2 = self.embedder.encode(text2)
        
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    def _contains_ground_truth(self, context: str, ground_truth: str) -> bool:
        """
        Check if context contains ground truth information
        
        Args:
            context: Context to check
            ground_truth: Ground truth to look for
        
        Returns:
            True if context contains ground truth
        """
        context_embedding = self.embedder.encode(context)
        truth_embedding = self.embedder.encode(ground_truth)
        
        similarity = np.dot(context_embedding, truth_embedding) / (
            np.linalg.norm(context_embedding) * np.linalg.norm(truth_embedding)
        )
        
        return similarity > 0.7  # Threshold for semantic similarity
    
    def batch_evaluate(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate multiple examples and aggregate results
        
        Args:
            data: List of examples with question, answer, contexts, and optionally ground_truth
        
        Returns:
            Aggregated evaluation results
        """
        all_scores = []
        
        for item in data:
            score = self.evaluate(
                question=item['question'],
                answer=item['answer'],
                contexts=item['contexts'],
                ground_truth=item.get('ground_truth')
            )
            all_scores.append(score)
        
        # Aggregate metrics
        aggregated = {
            'faithfulness': np.mean([s.faithfulness for s in all_scores]),
            'answer_relevancy': np.mean([s.answer_relevancy for s in all_scores]),
            'context_relevancy': np.mean([s.context_relevancy for s in all_scores]),
            'overall': np.mean([s.overall for s in all_scores]),
            'num_examples': len(all_scores)
        }
        
        # Add context precision if available
        precision_scores = [s.context_precision for s in all_scores if s.context_precision is not None]
        if precision_scores:
            aggregated['context_precision'] = np.mean(precision_scores)
        
        return aggregated