import abc
from typing import List, Dict, Any
import re

class TextSplitter(abc.ABC):
    """
    Abstract base class for text splitting strategies.
    """
    @abc.abstractmethod
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Splits the input text into chunks.

        Args:
            text: The input text to split.
            metadata: Optional metadata associated with the text (e.g., source, page_number).

        Returns:
            A list of dictionaries, where each dictionary represents a chunk
            and contains 'content' and 'metadata' keys.
        """
        pass

class FixedSizeTextSplitter(TextSplitter):
    """
    Implements fixed-size chunking with overlap.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be non-negative and less than chunk size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not text:
            return []
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_content = text[start:end]
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['start_index'] = start
            chunk_metadata['end_index'] = end
            chunks.append({"content": chunk_content, "metadata": chunk_metadata})
            start += self.chunk_size - self.chunk_overlap
        return chunks

class SentenceTextSplitter(TextSplitter):
    """
    Implements sentence-based chunking.
    """
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not text:
            return []
            
        # A more robust sentence splitter would use a library like NLTK or spaCy.
        # For simplicity, we'll use a basic regex split.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():  # Skip empty sentences
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['sentence_index'] = i
                chunks.append({"content": sentence.strip(), "metadata": chunk_metadata})
        return chunks

class DocumentLoader:
    """
    Handles loading documents from different file formats.
    """
    def load_text_from_file(self, file_path: str) -> str:
        """Load text from a plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_pdf_text(self, file_path: str) -> str:
        """Load text from a PDF file."""
        # Placeholder for PDF text extraction logic
        # Requires libraries like PyPDF2 or pdfminer.six
        # For now, return a placeholder message
        return f"Content from PDF: {file_path}"

    def load_markdown_text(self, file_path: str) -> str:
        """Load text from a Markdown file."""
        # Placeholder for Markdown text extraction logic
        # For now, return a placeholder message
        return f"Content from Markdown: {file_path}" 