"""
OpenAI Embeddings Service for ImpactOS.

Provides embedding generation using OpenAI's text-embedding models.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import OpenAI
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings using OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize Embedding Service.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: Embedding model to use (text-embedding-3-small or text-embedding-3-large)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.dimension = 1536 if model == "text-embedding-3-small" else 3072
        self.embedding_cache: Dict[str, List[float]] = {}
        
        logger.info(f"Initialized embedding service with model: {model}")
    
    def generate_embedding(self, text: str, cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            cache: Whether to cache the result
            
        Returns:
            Embedding vector
        """
        try:
            # Check cache
            if cache and text in self.embedding_cache:
                return self.embedding_cache[text]
            
            # Generate embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Cache if requested
            if cache:
                self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], cache: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            cache: Whether to cache results
            
        Returns:
            List of embedding vectors
        """
        try:
            # Separate cached and uncached
            embeddings = []
            texts_to_embed = []
            text_indices = []
            
            for i, text in enumerate(texts):
                if cache and text in self.embedding_cache:
                    embeddings.append((i, self.embedding_cache[text]))
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            
            # Generate new embeddings if needed
            if texts_to_embed:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts_to_embed
                )
                
                for idx, embedding_data in zip(text_indices, response.data):
                    embedding = embedding_data.embedding
                    embeddings.append((idx, embedding))
                    
                    # Cache if requested
                    if cache:
                        self.embedding_cache[texts[idx]] = embedding
            
            # Sort by original index and return
            embeddings.sort(key=lambda x: x[0])
            return [emb for _, emb in embeddings]
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure in range [0, 1]
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_similar(self, query: str, corpus: List[str], top_k: int = 10,
                    threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find similar texts in corpus.
        
        Args:
            query: Query text
            corpus: List of texts to search
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar items with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Generate corpus embeddings
            corpus_embeddings = self.generate_embeddings_batch(corpus)
            
            # Compute similarities
            results = []
            for i, (text, embedding) in enumerate(zip(corpus, corpus_embeddings)):
                similarity = self.compute_similarity(query_embedding, embedding)
                
                if similarity >= threshold:
                    results.append({
                        "text": text,
                        "index": i,
                        "similarity": similarity
                    })
            
            # Sort by similarity and return top k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar texts: {e}")
            return []
    
    def embed_documents(self, documents: List[Dict[str, Any]], 
                       text_field: str = "content") -> List[Dict[str, Any]]:
        """
        Add embeddings to a list of documents.
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing text to embed
            
        Returns:
            Documents with added embedding field
        """
        try:
            # Extract texts
            texts = [doc.get(text_field, "") for doc in documents]
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(texts)
            
            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                doc["embedding"] = embedding
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            return documents
    
    def save_embeddings(self, output_path: str = "embeddings.pkl"):
        """Save embedding cache to file."""
        try:
            with open(output_path, 'wb') as f:
                pickle.dump({
                    "model": self.model,
                    "dimension": self.dimension,
                    "cache": self.embedding_cache
                }, f)
            
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def load_embeddings(self, input_path: str = "embeddings.pkl"):
        """Load embedding cache from file."""
        try:
            if not os.path.exists(input_path):
                logger.warning(f"Embeddings file not found: {input_path}")
                return
            
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            
            # Verify model compatibility
            if data.get("model") != self.model:
                logger.warning(f"Model mismatch: file has {data.get('model')}, using {self.model}")
                return
            
            self.embedding_cache = data.get("cache", {})
            logger.info(f"Loaded {len(self.embedding_cache)} embeddings from {input_path}")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
    
    def create_embedding_index(self, texts: List[str]) -> Dict[str, Any]:
        """
        Create an embedding index for fast similarity search.
        
        Args:
            texts: List of texts to index
            
        Returns:
            Index metadata
        """
        try:
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(texts)
            
            # Convert to numpy array
            embedding_matrix = np.array(embeddings).astype('float32')
            
            # Create index metadata
            index = {
                "texts": texts,
                "embeddings": embedding_matrix,
                "model": self.model,
                "dimension": self.dimension,
                "size": len(texts)
            }
            
            logger.info(f"Created embedding index with {len(texts)} items")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create embedding index: {e}")
            raise
    
    def search_index(self, index: Dict[str, Any], query: str, 
                    top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search embedding index for similar items.
        
        Args:
            index: Embedding index from create_embedding_index
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar items with scores
        """
        try:
            # Generate query embedding
            query_embedding = np.array(self.generate_embedding(query)).astype('float32')
            
            # Compute similarities
            embeddings = index["embeddings"]
            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Build results
            results = []
            for idx in top_indices:
                results.append({
                    "text": index["texts"][idx],
                    "index": int(idx),
                    "similarity": float(similarities[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search index: {e}")
            return []