"""
FAISS Vector Search Implementation for ImpactOS AI Layer MVP.

This module provides proper vector similarity search using FAISS instead of
the previous word overlap approach. Handles vector indexing, storage, and 
efficient similarity search.
"""

import faiss
import numpy as np
import sqlite3
import os
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
from embedding_registry import get_embedding_model

# Logging configured by entrypoint; avoid per-module basicConfig
logger = logging.getLogger(__name__)


class FAISSVectorSearch:
    """Proper FAISS-based vector search system."""
    
    def __init__(self, db_path: str = "db/impactos.db", index_path: str = "db/faiss_index"):
        """
        Initialize FAISS vector search system.
        
        Args:
            db_path: Path to SQLite database
            index_path: Path to FAISS index directory
        """
        # Allow environment overrides for default paths without breaking custom callers
        env_db = os.getenv('IMPACTOS_DB_PATH')
        env_index = os.getenv('IMPACTOS_FAISS_INDEX_PATH')
        self.db_path = env_db if (env_db and db_path == "db/impactos.db") else db_path
        self.index_path = env_index if (env_index and index_path == "db/faiss_index") else index_path
        
        # Load configuration for dynamic values
        try:
            from config import get_config
            self.config = get_config()
            self.embedding_dimension = self.config.vector_search.embedding_dimension
            self.default_batch_size = self.config.vector_search.batch_size
        except ImportError:
            # Fallback if config not available
            self.embedding_dimension = 384
            self.default_batch_size = 100
        
        # Initialize components
        self.embedding_model = self._initialize_embedding_model()
        self.index = None
        self.metadata = []  # Store metadata for each vector
        
        # Create index directory
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Load existing index if available
        self._load_or_create_index()
    
    def _initialize_embedding_model(self) -> Optional["SentenceTransformer"]:
        """Initialize or reuse a shared sentence transformer model."""
        try:
            model = get_embedding_model()
            if model is not None:
                logger.info("FAISS embedding model initialized (shared)")
            else:
                logger.error("FAISS embedding model unavailable; search disabled")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return None
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one."""
        index_file = f"{self.index_path}.faiss"
        metadata_file = f"{self.index_path}.metadata"
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                # Load existing index
                self.index = faiss.read_index(index_file)
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index."""
        # Use IndexFlatIP for inner product (cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.metadata = []
        logger.info("Created new FAISS index")
    
    def add_embeddings(self, embeddings: List[Dict[str, Any]]):
        """
        Add embeddings to FAISS index.
        
        Args:
            embeddings: List of embedding dictionaries with 'vector' and metadata
        """
        if not embeddings:
            return
        
        vectors = []
        metadata_batch = []
        
        for embedding in embeddings:
            if 'vector' in embedding:
                # Normalize vector for cosine similarity
                vector = np.array(embedding['vector'], dtype=np.float32)
                vector = vector / np.linalg.norm(vector)
                vectors.append(vector)
                
                # Store metadata
                metadata_batch.append({
                    'text_chunk': embedding.get('text_chunk', ''),
                    'metric_id': embedding.get('metric_id'),
                    'chunk_type': embedding.get('chunk_type', ''),
                    'metric_name': embedding.get('metric_name', ''),
                    'metric_category': embedding.get('metric_category', ''),
                    'filename': embedding.get('filename', ''),
                    'source_info': embedding.get('source_info', {})
                })
        
        if vectors:
            # Add to FAISS index
            vectors_array = np.array(vectors)
            self.index.add(vectors_array)
            self.metadata.extend(metadata_batch)
            
            logger.info(f"Added {len(vectors)} vectors to FAISS index. Total: {self.index.ntotal}")
            
            # Save index immediately
            self._save_index()
    
    def search(self, query: str, k: int = None, min_similarity: float = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using proper FAISS similarity.
        
        Args:
            query: Query text
            k: Number of results to return (uses config default if None)
            min_similarity: Minimum similarity threshold (uses config default if None)
            
        Returns:
            List of results with similarity scores and metadata
        """
        # Use configuration defaults if not specified
        if k is None:
            k = getattr(self.config.vector_search, 'default_k', 50) if hasattr(self, 'config') else 50
        if min_similarity is None:
            min_similarity = getattr(self.config.vector_search, 'min_similarity_threshold', 0.3) if hasattr(self, 'config') else 0.3
        if not self.embedding_model or self.index.ntotal == 0:
            logger.warning("No embedding model or empty index")
            return []
        
        try:
            # Generate query embedding
            query_vector = self.embedding_model.encode([query])[0]
            query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
            
            # Search FAISS index
            similarities, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1 and similarity >= min_similarity:  # Valid result above threshold
                    metadata = self.metadata[idx]
                    result = {
                        'type': 'faiss_vector_match',
                        'similarity_score': float(similarity),
                        'relevance_score': float(similarity),  # For compatibility
                        'data': {
                            'text_chunk': metadata['text_chunk'],
                            'metric_id': metadata['metric_id'],
                            'chunk_type': metadata['chunk_type'],
                            'metric_name': metadata['metric_name'],
                            'metric_category': metadata['metric_category'],
                            'filename': metadata['filename'],
                            **metadata.get('source_info', {})
                        }
                    }
                    results.append(result)
            
            logger.info(f"FAISS search found {len(results)} results above similarity {min_similarity}")
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            return []
    
    def rebuild_index_from_database(self):
        """
        Rebuild FAISS index from existing database embeddings.
        This should be called once to migrate from the old system.
        """
        logger.info("Rebuilding FAISS index from database...")
        
        # Create new index
        self._create_new_index()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get all text chunks and generate embeddings
                cursor.execute("""
                    SELECT 
                        e.chunk_type,
                        e.metric_id,
                        im.metric_name,
                        im.metric_value,
                        im.metric_unit,
                        im.metric_category,
                        im.context_description,
                        im.source_sheet_name,
                        im.source_column_name,
                        im.source_cell_reference,
                        im.source_formula,
                        im.verification_status,
                        s.filename
                    FROM embeddings e
                    LEFT JOIN impact_metrics im ON e.metric_id = im.id
                    LEFT JOIN sources s ON im.source_id = s.id
                    WHERE e.metric_id IS NOT NULL
                """)
                
                rows = cursor.fetchall()
                
                if not rows:
                    logger.warning("No embeddings found in database")
                    return
                
                # Generate embeddings in batches using configured batch size
                batch_size = self.default_batch_size
                embeddings_to_add = []
                
                for row in rows:
                    # Compose a richer, vendor-aware text chunk for better recall
                    filename = row['filename'] or ''
                    vendor_name = None
                    try:
                        base_no_ext = os.path.splitext(filename)[0]
                        tokens = [t for t in base_no_ext.split('_') if t]
                        if len(tokens) >= 2:
                            vendor_name = tokens[1]
                        elif tokens:
                            vendor_name = tokens[0]
                    except Exception:
                        vendor_name = None
                    vendor_part = f"Vendor: {vendor_name}; " if vendor_name else ""
                    source_part = f"Source: {filename}; " if filename else ""
                    text_chunk = (
                        f"{source_part}{vendor_part}"
                        f"Metric: {row['metric_name'] or ''} "
                        f"Value: {row['metric_value']} {row['metric_unit'] or ''} "
                        f"Category: {row['metric_category'] or ''} "
                        f"Context: {row['context_description'] or ''}"
                    )
                    
                    # Generate vector embedding
                    vector = self.embedding_model.encode([text_chunk])[0]
                    
                    embedding_data = {
                        'vector': vector,
                        'text_chunk': text_chunk,
                        'metric_id': row['metric_id'],
                        'chunk_type': row['chunk_type'],
                        'metric_name': row['metric_name'] or '',
                        'metric_category': row['metric_category'] or '',
                        'filename': filename,
                        'source_info': {
                            'metric_value': row['metric_value'],
                            'metric_unit': row['metric_unit'],
                            'context_description': row['context_description'],
                            'source_sheet_name': row['source_sheet_name'],
                            'source_column_name': row['source_column_name'],
                            'source_cell_reference': row['source_cell_reference'],
                            'source_formula': row['source_formula'],
                            'verification_status': row['verification_status']
                        }
                    }
                    
                    embeddings_to_add.append(embedding_data)
                    
                    # Add in batches
                    if len(embeddings_to_add) >= batch_size:
                        self.add_embeddings(embeddings_to_add)
                        embeddings_to_add = []
                
                # Add remaining embeddings
                if embeddings_to_add:
                    self.add_embeddings(embeddings_to_add)
                
                logger.info(f"Successfully rebuilt FAISS index with {self.index.ntotal} vectors")
                
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {e}")
            raise
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            index_file = f"{self.index_path}.faiss"
            metadata_file = f"{self.index_path}.metadata"
            
            # Save FAISS index
            faiss.write_index(self.index, index_file)
            
            # Save metadata
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
                
            logger.debug(f"Saved FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics."""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.embedding_dimension,
            'index_type': type(self.index).__name__ if self.index else None,
            'metadata_count': len(self.metadata)
        } 