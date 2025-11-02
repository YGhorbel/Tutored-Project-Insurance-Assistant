"""RAG engine for query processing"""
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
import torch

sys.path.append(str(Path(__file__).parent.parent))

from utils.faiss_manager import FAISSManager
from utils.storage import StorageClient
from utils.llm_client import LLMClient


class RAGEngine:
    """Retrieval-Augmented Generation engine"""
    
    def __init__(self, config: dict):
        """Initialize RAG engine"""
        self.config = config
        
        # Initialize storage
        storage_cfg = config['storage']
        self.storage = StorageClient(
            endpoint=storage_cfg['endpoint'],
            access_key=storage_cfg['access_key'],
            secret_key=storage_cfg['secret_key'],
            secure=storage_cfg.get('secure', False)
        )
        
        # Initialize embedding model
        embedding_cfg = config['embedding']
        device = embedding_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = SentenceTransformer(
            embedding_cfg['model'],
            device=device
        )
        
        # Initialize FAISS index
        faiss_cfg = config['faiss']
        self.faiss_manager = FAISSManager(
            dimension=embedding_cfg['dimension'],
            index_type=faiss_cfg['index_type'],
            config=faiss_cfg
        )
        
        # Initialize LLM
        llm_cfg = config['llm']
        self.llm = LLMClient(
            model_name=llm_cfg['model'],
            api_url=llm_cfg.get('api_url'),
            device=device,
            load_in_8bit=llm_cfg.get('load_in_8bit', True)
        )
        
        self.indexes_bucket = storage_cfg['buckets']['indexes']
        self.top_k = config['retrieval'].get('top_k', 5)
        
        # Load index
        self.load_index()
        
        logger.info("RAG Engine initialized")
    
    def load_index(self):
        """Load FAISS index from storage"""
        try:
            index_path = "/tmp/faiss.index"
            metadata_path = "/tmp/faiss_metadata.pkl"
            
            self.storage.download_file(self.indexes_bucket, "faiss.index", index_path)
            self.storage.download_file(self.indexes_bucket, "faiss_metadata.pkl", metadata_path)
            
            self.faiss_manager.load(index_path, metadata_path)
            logger.info(f"Loaded index: {self.faiss_manager.get_stats()}")
        
        except Exception as e:
            logger.warning(f"Could not load index: {e}")
            # Create empty index
            self.faiss_manager.create_index()
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.config['embedding'].get('normalize', True)
        )
        return embedding
    
    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[dict], List[float]]:
        """
        Retrieve relevant chunks for query
        
        Returns:
            Tuple of (metadata_list, scores_list)
        """
        if top_k is None:
            top_k = self.top_k
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search index
        nprobe = self.config['faiss'].get('nprobe', 10)
        distances, indices, metadata_list = self.faiss_manager.search(
            query_embedding,
            k=top_k,
            nprobe=nprobe
        )
        
        # Convert distances to similarity scores (L2 distance to similarity)
        scores = [1.0 / (1.0 + d) for d in distances[0]]
        
        logger.info(f"Retrieved {len(metadata_list[0])} chunks for query")
        return metadata_list[0], scores
    
    def generate_answer(self, question: str, context_chunks: List[dict]) -> str:
        """Generate answer from question and context"""
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks[:3]):  # Use top 3 chunks
            text = chunk.get('text', '')
            summary = chunk.get('summary', '')
            source = chunk.get('metadata', {}).get('source', 'Unknown')
            
            context_parts.append(
                f"[Document {i+1}: {source}]\n"
                f"Summary: {summary}\n"
                f"Content: {text[:500]}..."
            )
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""You are an expert in insurance regulations. Based on the following regulatory documents, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer (be specific and cite relevant regulations):"""
        
        # Generate answer
        answer = self.llm.generate(
            prompt,
            max_tokens=self.config['llm'].get('max_tokens', 256),
            temperature=self.config['llm'].get('temperature', 0.3)
        )
        
        return answer.strip()
    
    def query(self, question: str, top_k: int = None, 
              min_score: float = 0.0) -> dict:
        """
        Full RAG query pipeline
        
        Returns:
            Dict with answer and supporting chunks
        """
        # Retrieve relevant chunks
        metadata_list, scores = self.retrieve(question, top_k)
        
        # Filter by score
        filtered_results = [
            (meta, score)
            for meta, score in zip(metadata_list, scores)
            if meta is not None and score >= min_score
        ]
        
        if not filtered_results:
            return {
                'answer': "I couldn't find relevant information to answer this question.",
                'chunks': [],
                'total_searched': len(metadata_list)
            }
        
        # Generate answer
        answer = self.generate_answer(question, [m for m, _ in filtered_results])
        
        # Format results
        chunks = [
            {
                'chunk_id': meta.get('chunk_id', 'unknown'),
                'text': meta.get('text', ''),
                'summary': meta.get('summary', ''),
                'score': float(score),
                'metadata': meta.get('metadata', {})
            }
            for meta, score in filtered_results
        ]
        
        return {
            'answer': answer,
            'chunks': chunks,
            'total_searched': len(metadata_list)
        }
    
    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return (
            self.faiss_manager.index is not None and
            self.faiss_manager.index.ntotal > 0 and
            self.embedding_model is not None and
            self.llm is not None
        )