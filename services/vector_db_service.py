import logging
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import config

logger = logging.getLogger(__name__)

class VectorDBService:
    """Service for managing vector database operations with Chroma."""
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(config.VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        logger.info(f"Initialized Chroma DB at {config.VECTOR_DB_PATH}")
        
    def create_collection(self, collection_name: str, overwrite: bool = False):
        """
        Create a new collection or get an existing one.
        
        Args:
            collection_name: Name of the collection
            overwrite: If True, delete existing collection with the same name
            
        Returns:
            chromadb.Collection: The collection object
        """
        try:
            if overwrite:
                try:
                    self.client.delete_collection(collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                except Exception:
                    # Collection might not exist, which is fine
                    pass
                    
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created collection: {collection_name}")
            return collection
        except Exception as e:
            # Collection might already exist
            logger.info(f"Getting existing collection: {collection_name}")
            return self.client.get_collection(collection_name)
            
    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        """
        Add documents to a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document chunks with text and metadata
            
        Returns:
            bool: Success status
        """
        try:
            collection = self.client.get_collection(collection_name)
            
            # Prepare data for batch insertion
            texts = [doc["text"] for doc in documents]
            ids = [f"chunk_{doc['metadata']['chunk_id']}" for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # Create embeddings
            embeddings = self.embedding_service.create_embeddings(texts)
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to collection: {e}")
            return False
            
    def query_collection(self, collection_name: str, query_text: str, n_results: int = 5):
        """
        Query the collection for relevant documents.
        
        Args:
            collection_name: Name of the collection
            query_text: The query text
            n_results: Number of results to return
            
        Returns:
            List[Dict]: List of matching documents with their metadata
        """
        try:
            collection = self.client.get_collection(collection_name)
            
            # Create query embedding
            query_embedding = self.embedding_service.create_embedding(query_text)
            
            # Query the collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and "documents" in results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 1.0
                    })
                    
            logger.info(f"Found {len(formatted_results)} results for query: {query_text[:50]}...")
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return []
