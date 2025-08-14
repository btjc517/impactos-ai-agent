"""
Retrieval service combining OpenAI's file search with embeddings.

Provides hybrid retrieval using both semantic search and keyword matching.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from openai import OpenAI

from .embedding_service import EmbeddingService
from .file_manager import FileManager

logger = logging.getLogger(__name__)


class RetrievalService:
    """Hybrid retrieval service using OpenAI tools."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Retrieval Service.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_service = EmbeddingService(api_key)
        self.file_manager = FileManager(api_key)
        
        self.document_store: List[Dict[str, Any]] = []
        self.metadata_index: Dict[str, Any] = {}
    
    def index_documents(self, documents: List[Dict[str, Any]], 
                       content_field: str = "content",
                       metadata_fields: Optional[List[str]] = None) -> bool:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of document dictionaries
            content_field: Field containing main text content
            metadata_fields: Fields to include in metadata index
            
        Returns:
            Success status
        """
        try:
            # Add embeddings to documents
            documents = self.embedding_service.embed_documents(
                documents, 
                text_field=content_field
            )
            
            # Store documents
            self.document_store.extend(documents)
            
            # Build metadata index
            if metadata_fields:
                for i, doc in enumerate(documents):
                    doc_id = doc.get("id", f"doc_{len(self.document_store) - len(documents) + i}")
                    self.metadata_index[doc_id] = {
                        field: doc.get(field) 
                        for field in metadata_fields 
                        if field in doc
                    }
            
            logger.info(f"Indexed {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False
    
    def index_files(self, file_paths: List[str]) -> Tuple[bool, List[str]]:
        """
        Index files for retrieval using OpenAI file search.
        
        Args:
            file_paths: List of file paths to index
            
        Returns:
            Tuple of (success status, list of file IDs)
        """
        try:
            # Upload files
            file_ids = self.file_manager.upload_batch(file_paths)
            
            if not file_ids:
                logger.warning("No files uploaded")
                return False, []
            
            logger.info(f"Indexed {len(file_ids)} files for retrieval")
            return True, file_ids
            
        except Exception as e:
            logger.error(f"Failed to index files: {e}")
            return False, []
    
    def hybrid_search(self, query: str, top_k: int = 10,
                     semantic_weight: float = 0.7,
                     keyword_weight: float = 0.3,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching
            filters: Optional metadata filters
            
        Returns:
            List of search results with scores
        """
        try:
            # Apply filters if provided
            search_docs = self.document_store
            if filters:
                search_docs = self._apply_filters(search_docs, filters)
            
            if not search_docs:
                return []
            
            # Semantic search
            semantic_results = self._semantic_search(query, search_docs, top_k * 2)
            
            # Keyword search
            keyword_results = self._keyword_search(query, search_docs, top_k * 2)
            
            # Combine results
            combined_results = self._combine_results(
                semantic_results,
                keyword_results,
                semantic_weight,
                keyword_weight
            )
            
            # Sort by combined score and return top k
            combined_results.sort(key=lambda x: x["score"], reverse=True)
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _semantic_search(self, query: str, documents: List[Dict[str, Any]], 
                        top_k: int) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Compute similarities
            results = []
            for doc in documents:
                if "embedding" not in doc:
                    continue
                
                similarity = self.embedding_service.compute_similarity(
                    query_embedding,
                    doc["embedding"]
                )
                
                results.append({
                    "document": doc,
                    "semantic_score": similarity,
                    "score": similarity
                })
            
            # Sort and return top k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, documents: List[Dict[str, Any]], 
                       top_k: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search."""
        try:
            # Tokenize query
            query_terms = query.lower().split()
            
            results = []
            for doc in documents:
                # Get document text
                doc_text = str(doc.get("content", "")).lower()
                
                # Calculate keyword score
                matches = sum(1 for term in query_terms if term in doc_text)
                score = matches / len(query_terms) if query_terms else 0
                
                if score > 0:
                    results.append({
                        "document": doc,
                        "keyword_score": score,
                        "score": score
                    })
            
            # Sort and return top k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _combine_results(self, semantic_results: List[Dict[str, Any]],
                        keyword_results: List[Dict[str, Any]],
                        semantic_weight: float,
                        keyword_weight: float) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results."""
        try:
            # Create result map
            combined = {}
            
            # Add semantic results
            for result in semantic_results:
                doc_id = id(result["document"])
                combined[doc_id] = {
                    "document": result["document"],
                    "semantic_score": result.get("semantic_score", 0),
                    "keyword_score": 0,
                    "score": result["semantic_score"] * semantic_weight
                }
            
            # Add/update with keyword results
            for result in keyword_results:
                doc_id = id(result["document"])
                if doc_id in combined:
                    combined[doc_id]["keyword_score"] = result.get("keyword_score", 0)
                    combined[doc_id]["score"] = (
                        combined[doc_id]["semantic_score"] * semantic_weight +
                        result["keyword_score"] * keyword_weight
                    )
                else:
                    combined[doc_id] = {
                        "document": result["document"],
                        "semantic_score": 0,
                        "keyword_score": result.get("keyword_score", 0),
                        "score": result["keyword_score"] * keyword_weight
                    }
            
            return list(combined.values())
            
        except Exception as e:
            logger.error(f"Failed to combine results: {e}")
            return []
    
    def _apply_filters(self, documents: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply metadata filters to documents."""
        filtered = []
        
        for doc in documents:
            match = True
            for field, value in filters.items():
                if field not in doc:
                    match = False
                    break
                
                # Handle different filter types
                if isinstance(value, list):
                    # In filter
                    if doc[field] not in value:
                        match = False
                        break
                elif isinstance(value, dict):
                    # Range filter
                    if "min" in value and doc[field] < value["min"]:
                        match = False
                        break
                    if "max" in value and doc[field] > value["max"]:
                        match = False
                        break
                else:
                    # Exact match
                    if doc[field] != value:
                        match = False
                        break
            
            if match:
                filtered.append(doc)
        
        return filtered
    
    def retrieve_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Query to find context for
            max_tokens: Maximum tokens in context
            
        Returns:
            Concatenated context string
        """
        try:
            # Perform hybrid search
            results = self.hybrid_search(query, top_k=20)
            
            if not results:
                return ""
            
            # Build context
            context_parts = []
            token_count = 0
            
            for result in results:
                doc = result["document"]
                content = doc.get("content", "")
                
                # Estimate tokens (rough approximation)
                estimated_tokens = len(content.split()) * 1.3
                
                if token_count + estimated_tokens > max_tokens:
                    break
                
                # Add source citation if available
                source = doc.get("source", doc.get("filename", ""))
                if source:
                    context_parts.append(f"[Source: {source}]")
                
                context_parts.append(content)
                token_count += estimated_tokens
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return ""
    
    def save_index(self, output_path: str = "retrieval_index.json"):
        """Save retrieval index to file."""
        try:
            # Prepare data for serialization
            save_data = {
                "document_store": [
                    {k: v for k, v in doc.items() if k != "embedding"}
                    for doc in self.document_store
                ],
                "metadata_index": self.metadata_index,
                "document_count": len(self.document_store),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            # Save embeddings separately
            self.embedding_service.save_embeddings(output_path.replace('.json', '_embeddings.pkl'))
            
            logger.info(f"Saved retrieval index to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self, input_path: str = "retrieval_index.json"):
        """Load retrieval index from file."""
        try:
            if not os.path.exists(input_path):
                logger.warning(f"Index file not found: {input_path}")
                return
            
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            self.document_store = data.get("document_store", [])
            self.metadata_index = data.get("metadata_index", {})
            
            # Load embeddings
            self.embedding_service.load_embeddings(input_path.replace('.json', '_embeddings.pkl'))
            
            # Re-embed documents if needed
            for doc in self.document_store:
                if "content" in doc and "embedding" not in doc:
                    doc["embedding"] = self.embedding_service.generate_embedding(doc["content"])
            
            logger.info(f"Loaded {len(self.document_store)} documents from {input_path}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")