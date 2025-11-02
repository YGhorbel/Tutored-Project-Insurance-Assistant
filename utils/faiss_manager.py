"""FAISS index management"""
import faiss
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from loguru import logger
import pickle

class FAISSManager:
    """Manage FAISS index creation, loading, and searching"""
    
    def __init__(self, dimension: int, index_type: str = "Flat", config: dict = None):
        """
        Initialize FAISS manager
        
        Args:
            dimension: Embedding dimension
            index_type: Type of index (Flat, IVF, IVFPQ, HNSW)
            config: Additional configuration parameters
        """
        self.dimension = dimension
        self.index_type = index_type
        self.config = config or {}
        self.index: Optional[faiss.Index] = None
        self.metadata: List[dict] = []
        
        logger.info(f"Initialized FAISS manager: {index_type}, dim={dimension}")
    
    def create_index(self):
        """Create FAISS index based on type"""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created Flat index")
        
        elif self.index_type == "IVF":
            nlist = self.config.get('nlist', 100)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            logger.info(f"Created IVF index with {nlist} clusters")
        
        elif self.index_type == "IVFPQ":
            nlist = self.config.get('nlist', 100)
            m = self.config.get('m', 8)
            bits = self.config.get('bits', 8)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, bits)
            logger.info(f"Created IVFPQ index: nlist={nlist}, m={m}")
        
        elif self.index_type == "HNSW":
            M = self.config.get('M', 32)
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            self.index.hnsw.efConstruction = self.config.get('ef_construction', 200)
            logger.info(f"Created HNSW index with M={M}")
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def train_index(self, embeddings: np.ndarray):
        """Train index (required for IVF-based indexes)"""
        if self.index is None:
            self.create_index()
        
        if isinstance(self.index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
            logger.info(f"Training index with {len(embeddings)} vectors")
            self.index.train(embeddings)
            logger.info("Index training completed")
    
    def add_vectors(self, embeddings: np.ndarray, metadata: List[dict]):
        """Add vectors to index with metadata"""
        if self.index is None:
            self.create_index()
        
        # Ensure embeddings are contiguous and float32
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        
        # Train if needed
        if not self.index.is_trained:
            self.train_index(embeddings)
        
        # Add vectors
        start_id = len(self.metadata)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} vectors (total: {self.index.ntotal})")
    
    def search(self, query_vector: np.ndarray, k: int = 5, 
               nprobe: int = None) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Search for nearest neighbors
        
        Returns:
            distances: Array of distances
            indices: Array of indices
            metadata: List of metadata for results
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return np.array([]), np.array([]), []
        
        # Set nprobe for IVF indexes
        if nprobe and isinstance(self.index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
            self.index.nprobe = nprobe
        
        # Ensure query is correct shape and type
        query_vector = np.ascontiguousarray(query_vector.astype('float32'))
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Get metadata for results
        result_metadata = []
        for idx_list in indices:
            row_metadata = []
            for idx in idx_list:
                if idx < len(self.metadata) and idx >= 0:
                    row_metadata.append(self.metadata[idx])
                else:
                    row_metadata.append(None)
            result_metadata.append(row_metadata)
        
        logger.debug(f"Search returned {len(result_metadata[0])} results")
        return distances, indices, result_metadata
    
    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata to disk"""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str):
        """Load index and metadata from disk"""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "is_trained": self.index.is_trained,
            "metadata_count": len(self.metadata)
        }