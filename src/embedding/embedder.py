import abc
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder(abc.ABC):
    """
    Abstract base class for embedding models.
    """
    @abc.abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of lists of floats, where each inner list is an embedding vector.
        """
        pass

class SentenceTransformerEmbedder(Embedder):
    """
    Embedder using Sentence Transformers models.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = './embedding_cache'):
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.cache = {}

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
            
        embeddings = []
        texts_to_embed = []
        
        for text in texts:
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                texts_to_embed.append(text)

        if texts_to_embed:
            # Batch processing for efficiency
            new_embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True).tolist()
            for i, text in enumerate(texts_to_embed):
                self.cache[text] = new_embeddings[i]
                embeddings.append(new_embeddings[i])
        return embeddings

class OpenAIEmbedder(Embedder):
    """
    Embedder using OpenAI API.
    """
    def __init__(self, model_name: str = 'text-embedding-ada-002'):
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.model_name = model_name
            self.cache = {}
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
            
        embeddings = []
        texts_to_embed = []
        
        for text in texts:
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                texts_to_embed.append(text)

        if texts_to_embed:
            # OpenAI API handles batching internally up to a limit
            response = self.client.embeddings.create(
                input=texts_to_embed,
                model=self.model_name
            )
            new_embeddings = [d.embedding for d in response.data]
            for i, text in enumerate(texts_to_embed):
                self.cache[text] = new_embeddings[i]
                embeddings.append(new_embeddings[i])
        return embeddings 