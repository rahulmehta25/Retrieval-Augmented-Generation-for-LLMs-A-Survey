from src.rag.naive_rag import NaiveRAG
from src.generation.generator import HuggingFaceGenerator
from typing import List, Dict, Any
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedRAG(NaiveRAG):
    """
    Extends NaiveRAG with advanced query optimization techniques.
    """
    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__(config_path)
        # Initialize an LLM specifically for query rewriting if needed
        self.query_rewriter_llm = HuggingFaceGenerator(model_name='distilgpt2', device='cpu')

    def _rewrite_query_with_llm(self, query: str) -> str:
        """
        Rewrites the query using an LLM to improve retrieval effectiveness.
        This can involve rephrasing, adding context, or clarifying intent.
        """
        # Example prompt for query rewriting
        prompt = f"""
        Rewrite the following question to make it more effective for searching a document database. 
        Focus on extracting key entities and concepts. Only provide the rewritten question.

        Original question: {query}
        Rewritten question:
        """
        try:
            rewritten = self.query_rewriter_llm.generate(prompt, max_new_tokens=50)
            # Clean up the response
            rewritten = rewritten.strip()
            if rewritten and len(rewritten) > 10:
                logging.info(f"Rewritten query: {query} -> {rewritten}")
                return rewritten
        except Exception as e:
            logging.warning(f"Query rewriting failed: {e}")
        
        return query

    def _expand_query_with_synonyms(self, query: str) -> List[str]:
        """
        Expands the query with synonyms or related terms.
        This could use a thesaurus, word embeddings, or an LLM.
        """
        logging.info(f"Expanding query: {query}")
        
        # Simple synonym expansion using a basic dictionary
        synonyms = {
            'capital': ['city', 'center', 'headquarters'],
            'france': ['french', 'paris'],
            'ai': ['artificial intelligence', 'machine learning'],
            'rag': ['retrieval augmented generation', 'retrieval generation'],
            'how': ['what', 'explain', 'describe'],
            'what': ['how', 'explain', 'describe'],
            'when': ['date', 'time', 'year']
        }
        
        expanded_queries = [query]
        
        # Add synonyms for key terms
        query_lower = query.lower()
        for term, syns in synonyms.items():
            if term in query_lower:
                for syn in syns:
                    new_query = query_lower.replace(term, syn)
                    if new_query != query_lower:
                        expanded_queries.append(new_query)
        
        return expanded_queries[:3]  # Limit to 3 expanded queries

    def _generate_hyde_embedding(self, query: str) -> List[float]:
        """
        Generates a Hypothetical Document Embedding (HyDE) for the query.
        This involves generating a hypothetical answer to the query using an LLM,
        and then embedding that hypothetical answer.
        """
        logging.info(f"Generating HyDE embedding for query: {query}")
        
        # Step 1: Generate a hypothetical answer using the LLM
        hyde_prompt = f"""
        Please write a concise, hypothetical answer to the following question. 
        Do not state that it is hypothetical. Just provide the answer.

        Question: {query}
        Answer:
        """
        
        try:
            hypothetical_answer = self.query_rewriter_llm.generate(hyde_prompt, max_new_tokens=100)
            # Clean up the response
            hypothetical_answer = hypothetical_answer.strip()
            if not hypothetical_answer:
                return self.embedder.embed([query])[0]
            
            # Step 2: Embed the hypothetical answer
            hyde_embedding = self.embedder.embed([hypothetical_answer])[0]
            return hyde_embedding
        except Exception as e:
            logging.warning(f"HyDE generation failed: {e}")
            return self.embedder.embed([query])[0]

    def _decompose_query(self, query: str) -> List[str]:
        """
        Decomposes a complex question into simpler sub-queries.
        """
        logging.info(f"Decomposing query: {query}")
        
        # Simple decomposition based on conjunctions
        conjunctions = [' and ', ' or ', ' but ', ' however ', ' also ']
        sub_queries = [query]
        
        for conj in conjunctions:
            if conj in query.lower():
                parts = query.split(conj)
                if len(parts) > 1:
                    sub_queries.extend([part.strip() for part in parts if part.strip()])
        
        return sub_queries[:3]  # Limit to 3 sub-queries

    def retrieve_optimized(self, query: str, k: int = 5, filters: Dict[str, Any] = None, 
                           query_optimization_strategy: str = "none") -> List[Dict[str, Any]]:
        """
        Retrieves documents using various query optimization strategies.
        """
        optimized_queries = [query]
        
        if query_optimization_strategy == "llm_rewrite":
            optimized_queries = [self._rewrite_query_with_llm(query)]
        elif query_optimization_strategy == "expansion":
            optimized_queries = self._expand_query_with_synonyms(query)
        elif query_optimization_strategy == "hyde":
            # For HyDE, we generate an embedding from a hypothetical answer
            # and use it directly for retrieval.
            hyde_embedding = self._generate_hyde_embedding(query)
            retrieved_docs = self.vector_store.search(hyde_embedding, k=k, filters=filters)
            return retrieved_docs
        elif query_optimization_strategy == "decomposition":
            sub_queries = self._decompose_query(query)
            all_retrieved_docs = []
            for sq in sub_queries:
                sq_embedding = self.embedder.embed([sq])[0]
                all_retrieved_docs.extend(self.vector_store.search(sq_embedding, k=k, filters=filters))
            # Deduplicate and re-rank if necessary
            return list({doc["content"]: doc for doc in all_retrieved_docs}.values())

        # For strategies that produce multiple queries or a single rewritten query
        all_retrieved_docs = []
        for opt_query in optimized_queries:
            query_embedding = self.embedder.embed([opt_query])[0]
            all_retrieved_docs.extend(self.vector_store.search(query_embedding, k=k, filters=filters))
        
        # Deduplicate and return
        return list({doc["content"]: doc for doc in all_retrieved_docs}.values())

    def query_optimized(self, question: str, k: int = 5, filters: Dict[str, Any] = None,
                        query_optimization_strategy: str = "none") -> str:
        """
        End-to-end query method for Advanced RAG with query optimization.
        """
        retrieved_contexts = self.retrieve_optimized(question, k=k, filters=filters, 
                                                     query_optimization_strategy=query_optimization_strategy)
        if not retrieved_contexts:
            logging.warning("No relevant contexts found after query optimization. Cannot generate an answer.")
            return "I am sorry, but I could not find enough relevant information to answer your question."
        answer = self.generate_answer(question, retrieved_contexts)
        return answer 