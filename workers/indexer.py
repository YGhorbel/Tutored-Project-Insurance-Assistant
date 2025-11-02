"""FAISS indexer - builds index from embeddings"""
import os
import sys
from pathlib import Path
import yaml
import numpy as np
from loguru import logger
import json

sys.path.append(str(Path(__file__).parent.parent))

from utils.faiss_manager import FAISSManager
from utils.storage import StorageClient
from utils.logger import setup_logger

logger = setup_logger("indexer")


class FAISSIndexer:
    """Build FAISS index from embeddings"""
    
    def __init__(self, config: dict):
        """Initialize indexer"""
        self.config = config
        
        # Initialize storage
        storage_cfg = config['storage']
        self.storage = StorageClient(
            endpoint=storage_cfg['endpoint'],
            access_key=storage_cfg['access_key'],
            secret_key=storage_cfg['secret_key'],
            secure=storage_cfg.get('secure', False)
        )
        
        self.embeddings_bucket = storage_cfg['buckets']['embeddings']
        self.indexes_bucket = storage_cfg['buckets']['indexes']
        
        # Initialize FAISS manager
        faiss_cfg = config['faiss']
        self.faiss_manager = FAISSManager(
            dimension=config['embedding']['dimension'],
            index_type=faiss_cfg['index_type'],
            config=faiss_cfg
        )
        
        logger.info("FAISSIndexer initialized")
    
    def load_embeddings_from_storage(self) -> tuple:
        """
        Load all embeddings from storage
        
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        logger.info(f"Loading embeddings from {self.embeddings_bucket}")
        
        # List all embedding files
        embedding_files = self.storage.list_objects(self.embeddings_bucket, "")
        embedding_files = [f for f in embedding_files if f.endswith('.json')]
        
        logger.info(f"Found {len(embedding_files)} embedding files")
        
        all_embeddings = []
        all_metadata = []
        
        for emb_file in embedding_files:
            try:
                # Download embedding data
                data = self.storage.download_json(self.embeddings_bucket, emb_file)
                
                embedding = np.array(data['embedding'], dtype=np.float32)
                metadata = data['metadata']
                
                all_embeddings.append(embedding)
                all_metadata.append(metadata)
            
            except Exception as e:
                logger.error(f"Error loading {emb_file}: {e}")
                continue
        
        embeddings_array = np.vstack(all_embeddings)
        logger.info(f"Loaded {len(embeddings_array)} embeddings")
        
        return embeddings_array, all_metadata
    
    def build_index(self):
        """Build FAISS index from embeddings"""
        logger.info("Starting index build")
        
        # Load embeddings
        embeddings, metadata = self.load_embeddings_from_storage()
        
        if len(embeddings) == 0:
            logger.warning("No embeddings found, skipping index build")
            return
        
        # Add to FAISS index
        self.faiss_manager.add_vectors(embeddings, metadata)
        
        # Save index
        index_path = "/tmp/faiss.index"
        metadata_path = "/tmp/faiss_metadata.pkl"
        
        self.faiss_manager.save(index_path, metadata_path)
        
        # Upload to storage
        self.storage.upload_file(self.indexes_bucket, "faiss.index", index_path)
        self.storage.upload_file(self.indexes_bucket, "faiss_metadata.pkl", metadata_path)
        
        logger.info(f"Index built and uploaded: {self.faiss_manager.get_stats()}")
        
        # Clean up
        os.remove(index_path)
        os.remove(metadata_path)
    
    def update_index(self, new_embeddings_prefix: str = ""):
        """
        Incrementally update existing index
        
        Args:
            new_embeddings_prefix: Prefix for new embeddings to add
        """
        logger.info("Updating index")
        
        # Load existing index
        try:
            index_path = "/tmp/faiss.index"
            metadata_path = "/tmp/faiss_metadata.pkl"
            
            self.storage.download_file(self.indexes_bucket, "faiss.index", index_path)
            self.storage.download_file(self.indexes_bucket, "faiss_metadata.pkl", metadata_path)
            
            self.faiss_manager.load(index_path, metadata_path)
            logger.info("Loaded existing index")
        
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}, building new one")
            return self.build_index()
        
        # Load new embeddings
        new_embeddings, new_metadata = self.load_embeddings_from_storage()
        
        # Filter to only new ones (this is simplified - in production, track which are new)
        # For now, we'll just rebuild
        logger.info("Rebuilding index with all embeddings")
        self.faiss_manager.add_vectors(new_embeddings, new_metadata)
        
        # Save updated index
        self.faiss_manager.save(index_path, metadata_path)
        self.storage.upload_file(self.indexes_bucket, "faiss.index", index_path)
        self.storage.upload_file(self.indexes_bucket, "faiss_metadata.pkl", metadata_path)
        
        logger.info(f"Index updated: {self.faiss_manager.get_stats()}")


def main():
    """Main entry point"""
    # Load configuration
    config_path = Path("/app/config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    def expand_env(obj):
        if isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        elif isinstance(obj, str) and obj.startswith('${'):
            var_name = obj[2:-1].split(':')[0]
            default = obj[2:-1].split(':')[1] if ':' in obj else None
            return os.getenv(var_name, default)
        return obj
    
    config = expand_env(config)
    
    # Create indexer and build
    indexer = FAISSIndexer(config)
    indexer.build_index()


if __name__ == "__main__":
    main()