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
        try:
            import pypdf
            text_parts = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"Page {page_num + 1}:\n{text}")
            
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return f"Error loading PDF: {file_path}"

    def load_markdown_text(self, file_path: str) -> str:
        """Load text from a Markdown file."""
        # Markdown files are plain text, so we can read them directly
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_excel_text(self, file_path: str) -> str:
        """Load text from an Excel file."""
        try:
            import pandas as pd
            import openpyxl
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            text_parts = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Convert DataFrame to readable text
                text_parts.append(f"\n=== Sheet: {sheet_name} ===\n")
                
                # Convert each row to text
                for idx, row in df.iterrows():
                    row_text = []
                    for col, value in row.items():
                        if pd.notna(value):
                            row_text.append(f"{col}: {value}")
                    if row_text:
                        text_parts.append(" | ".join(row_text))
            
            return "\n".join(text_parts)
        except Exception as e:
            print(f"Error reading Excel {file_path}: {e}")
            return f"Error loading Excel: {file_path}"
    
    def load_docx_text(self, file_path: str) -> str:
        """Load text from a Word document."""
        try:
            from docx import Document
            doc = Document(file_path)
            text_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            return "\n\n".join(text_parts)
        except ImportError:
            return "python-docx not installed. Cannot read Word documents."
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return f"Error loading DOCX: {file_path}" 