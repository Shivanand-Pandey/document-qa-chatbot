import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for creating and managing text embeddings."""
    
    def __init__(self):
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimensions = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimensions: {self.dimensions}")
        
    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text string.
        
        Args:
            text: The text to embed
            
        Returns:
            List[float]: The embedding vector
        """
        if not text or not text.strip():
            logger.warning("Attempted to create embedding for empty text")
            # Return zero vector of correct dimension
            return [0.0] * self.dimensions
            
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            # Return zero vector in case of error
            return [0.0] * self.dimensions
            
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            logger.warning("Attempted to create embeddings for empty text list")
            return []
            
        # Filter out empty strings
        filtered_texts = [t for t in texts if t and t.strip()]
        
        if not filtered_texts:
            return []
            
        try:
            embeddings = self.model.encode(filtered_texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings batch: {e}")
            return [[0.0] * self.dimensions] * len(filtered_texts)
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: The text to chunk
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        if not text or not text.strip():
            return []
            
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for i, word in enumerate(words):
            word_size = len(word) + 1  # +1 for space
            
            if current_size + word_size > config.CHUNK_SIZE and current_chunk:
                # Save the current chunk
                chunks.append({
                    "text": " ".join(current_chunk),
                    "metadata": {
                        "chunk_id": len(chunks),
                        "start_idx": i - len(current_chunk),
                        "end_idx": i - 1
                    }
                })
                
                # Start a new chunk with overlap
                overlap_start = max(0, len(current_chunk) - config.CHUNK_OVERLAP)
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(word) + 1 for word in current_chunk)
                
            current_chunk.append(word)
            current_size += word_size
            
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "metadata": {
                    "chunk_id": len(chunks),
                    "start_idx": len(words) - len(current_chunk),
                    "end_idx": len(words) - 1
                }
            })
            
        return chunks
