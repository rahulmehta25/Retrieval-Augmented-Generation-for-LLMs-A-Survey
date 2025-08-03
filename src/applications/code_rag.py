"""
Code RAG - Domain-Specific RAG for Programming Questions

This module implements a specialized RAG system for answering programming-related
questions by retrieving relevant code snippets, documentation, and examples.
"""

import re
import ast
from typing import List, Dict, Any, Optional
from src.rag.modular_rag import ModularRAG
from src.chunking.text_splitter import TextSplitter

class CodeChunker(TextSplitter):
    """
    Specialized chunker for code that respects code structure.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split code text while preserving function/class boundaries.
        """
        # Try to parse as Python code first
        if self._is_python_code(text):
            return self._split_python_code(text, metadata)
        else:
            # Fallback to general text splitting
            return self._split_general_code(text, metadata)
    
    def _is_python_code(self, text: str) -> bool:
        """
        Check if text appears to be Python code.
        """
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            # Check for common Python patterns
            python_patterns = [
                r'def\s+\w+\s*\(',
                r'class\s+\w+',
                r'import\s+',
                r'from\s+\w+\s+import',
                r'if\s+__name__\s*==\s*[\'"]__main__[\'"]',
                r'return\s+',
                r'print\s*\(',
                r'for\s+\w+\s+in\s+',
                r'while\s+',
                r'try:',
                r'except\s+',
                r'finally:'
            ]
            return any(re.search(pattern, text) for pattern in python_patterns)
    
    def _split_python_code(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split Python code while preserving function and class boundaries.
        """
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # Check if this line starts a new function or class
            is_new_block = (
                line.strip().startswith('def ') or
                line.strip().startswith('class ') or
                line.strip().startswith('async def ')
            )
            
            # If we have a new block and current chunk is getting large, start new chunk
            if (is_new_block and current_size > self.chunk_size * 0.7 and 
                current_chunk and not line.strip().startswith('def ') and not line.strip().startswith('class ')):
                
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'metadata': metadata or {},
                    'type': 'python_code'
                })
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap // 50)  # Rough line estimate
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(line) + 1 for line in current_chunk)
            
            current_chunk.append(line)
            current_size += line_size
            
            # If chunk is too large, force a split
            if current_size > self.chunk_size:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'metadata': metadata or {},
                    'type': 'python_code'
                })
                current_chunk = []
                current_size = 0
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'metadata': metadata or {},
                'type': 'python_code'
            })
        
        return chunks
    
    def _split_general_code(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split general code text using fixed-size chunks.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a newline to avoid cutting in the middle of lines
            if end < len(text):
                last_newline = text.rfind('\n', start, end)
                if last_newline > start:
                    end = last_newline + 1
            
            chunk_text = text[start:end]
            chunks.append({
                'content': chunk_text,
                'metadata': metadata or {},
                'type': 'code'
            })
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks

class CodeRAG(ModularRAG):
    """
    Specialized RAG system for programming questions.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        super().__init__(config_path)
        self.code_chunker = CodeChunker()
        
        # Code-specific prompt template
        self.code_prompt_template = """
        You are a helpful programming assistant. Use the following code examples and documentation to answer the programming question.

        Code Examples and Documentation:
        {context}

        Question: {question}

        Please provide a clear, well-documented answer. If the question asks for code, provide a complete, runnable example. If it's about debugging, explain the issue and provide a solution.

        Answer:
        """
    
    def index_code_files(self, file_paths: List[str]) -> None:
        """
        Index code files with specialized chunking.
        """
        logging.info(f"Indexing {len(file_paths)} code files...")
        
        all_chunks = []
        for file_path in file_paths:
            try:
                # Determine file type and use appropriate chunking
                if file_path.endswith('.py'):
                    chunks = self.code_chunker.split_text_from_file(file_path)
                else:
                    # Use regular chunking for non-Python files
                    chunks = self.components['text_splitter'].split_text_from_file(file_path)
                
                all_chunks.extend(chunks)
                logging.info(f"Processed {len(chunks)} chunks from {file_path}")
            except Exception as e:
                logging.warning(f"Failed to process {file_path}: {e}")
        
        if not all_chunks:
            logging.warning("No chunks to index")
            return
        
        # Generate embeddings
        chunk_texts = [chunk['content'] for chunk in all_chunks]
        logging.info(f"Generating embeddings for {len(chunk_texts)} code chunks...")
        embeddings = self.components['embedder'].embed(chunk_texts)
        
        # Add to vector store
        logging.info(f"Adding {len(all_chunks)} code chunks to vector store...")
        self.components['vector_store'].add_documents(all_chunks, embeddings)
        
        logging.info("Code indexing complete.")
    
    def query_code(self, question: str, k: int = 5) -> str:
        """
        Query the code RAG system with programming-specific processing.
        """
        # Enhance query with programming context
        enhanced_question = self._enhance_programming_query(question)
        
        # Retrieve relevant code snippets
        retrieved_docs = self.retrieve(enhanced_question, k=k)
        
        if not retrieved_docs:
            return "I couldn't find any relevant code examples to answer your programming question."
        
        # Filter for code-specific content
        code_docs = [doc for doc in retrieved_docs if doc.get('metadata', {}).get('type') in ['python_code', 'code']]
        
        if not code_docs:
            code_docs = retrieved_docs  # Use all docs if no code-specific ones found
        
        # Generate answer with code-specific prompt
        context = [doc['content'] for doc in code_docs]
        answer = self._generate_code_answer(enhanced_question, context)
        
        return answer
    
    def _enhance_programming_query(self, question: str) -> str:
        """
        Enhance programming queries with relevant context.
        """
        # Add programming language hints if not present
        if 'python' not in question.lower() and 'code' not in question.lower():
            # Try to infer language from question
            if any(word in question.lower() for word in ['function', 'def ', 'import', 'class']):
                question = f"Python: {question}"
            elif any(word in question.lower() for word in ['javascript', 'js', 'node']):
                question = f"JavaScript: {question}"
            elif any(word in question.lower() for word in ['java', 'public class']):
                question = f"Java: {question}"
        
        return question
    
    def _generate_code_answer(self, question: str, context: List[str]) -> str:
        """
        Generate answer using code-specific prompt template.
        """
        context_text = "\n\n".join(context)
        prompt = self.code_prompt_template.format(
            context=context_text,
            question=question
        )
        
        try:
            answer = self.components['generator'].generate(prompt, max_new_tokens=300)
            return answer
        except Exception as e:
            logging.error(f"Failed to generate code answer: {e}")
            return "I encountered an error while generating the answer. Please try rephrasing your question."
    
    def get_code_snippets(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Get relevant code snippets for a query.
        """
        retrieved_docs = self.retrieve(query, k=k)
        
        code_snippets = []
        for doc in retrieved_docs:
            if doc.get('metadata', {}).get('type') in ['python_code', 'code']:
                code_snippets.append({
                    'content': doc['content'],
                    'source': doc.get('metadata', {}).get('source', 'Unknown'),
                    'relevance_score': doc.get('relevance_score', 0.0)
                })
        
        return code_snippets
    
    def explain_code(self, code_snippet: str) -> str:
        """
        Explain a code snippet using the RAG system.
        """
        question = f"Please explain this code:\n\n{code_snippet}"
        return self.query_code(question)
    
    def debug_code(self, code_snippet: str, error_message: str = None) -> str:
        """
        Debug code using the RAG system.
        """
        if error_message:
            question = f"I'm getting this error in my code:\n\n{code_snippet}\n\nError: {error_message}\n\nHow can I fix this?"
        else:
            question = f"Can you help me debug this code?\n\n{code_snippet}"
        
        return self.query_code(question) 