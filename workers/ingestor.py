"""Document ingestor - chunks PDFs and publishes to Kafka"""
import os
import sys
from pathlib import Path
from typing import List, Dict
import hashlib
import yaml
from pypdf import PdfReader
from loguru import logger
from kafka import KafkaProducer
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.storage import StorageClient
from utils.logger import setup_logger

logger = setup_logger("ingestor")


class DocumentIngestor:
    """Ingest documents, chunk them, and publish to Kafka"""
    
    def __init__(self, config: dict):
        """Initialize ingestor with configuration"""
        self.config = config
        
        # Initialize storage
        storage_cfg = config['storage']
        self.storage = StorageClient(
            endpoint=storage_cfg['endpoint'],
            access_key=storage_cfg['access_key'],
            secret_key=storage_cfg['secret_key'],
            secure=storage_cfg.get('secure', False)
        )
        
        # Initialize Kafka producer
        kafka_cfg = config['kafka']
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_cfg['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        
        self.chunks_topic = kafka_cfg['topics']['chunks']
        self.chunk_size = config['processing']['chunk_size']
        self.chunk_overlap = config['processing']['chunk_overlap']
        
        logger.info("DocumentIngestor initialized")
    
    def chunk_text(self, text: str, metadata: dict) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Full text to chunk
            metadata: Document metadata
        
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        words = text.split()
        
        # Calculate words per chunk
        words_per_chunk = self.chunk_size // 4  # Approximate words
        overlap_words = self.chunk_overlap // 4
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk_words)
            
            # Create chunk ID
            chunk_id = hashlib.md5(
                f"{metadata.get('source', '')}-{i}".encode()
            ).hexdigest()
            
            chunk = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'chunk_index': len(chunks),
                'metadata': {
                    **metadata,
                    'char_start': i * 4,  # Approximate
                    'char_end': (i + len(chunk_words)) * 4
                }
            }
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def process_pdf(self, file_path: str, doc_metadata: dict = None) -> List[Dict]:
        """
        Process a PDF file into chunks
        
        Args:
            file_path: Path to PDF file
            doc_metadata: Additional metadata
        
        Returns:
            List of chunks
        """
        logger.info(f"Processing PDF: {file_path}")
        
        try:
            reader = PdfReader(file_path)
            all_chunks = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                if not text.strip():
                    continue
                
                metadata = {
                    'source': Path(file_path).name,
                    'page': page_num + 1,
                    'total_pages': len(reader.pages),
                    'doc_type': 'pdf',
                    **(doc_metadata or {})
                }
                
                # Chunk page text
                page_chunks = self.chunk_text(text, metadata)
                all_chunks.extend(page_chunks)
            
            logger.info(f"Processed {len(all_chunks)} chunks from {len(reader.pages)} pages")
            return all_chunks
        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []
    
    def ingest_from_storage(self, bucket: str, prefix: str = ""):
        """
        Ingest all documents from storage bucket
        
        Args:
            bucket: Bucket name
            prefix: Object prefix to filter
        """
        logger.info(f"Ingesting from {bucket}/{prefix}")
        
        # List objects
        objects = self.storage.list_objects(bucket, prefix)
        pdf_objects = [obj for obj in objects if obj.endswith('.pdf')]
        
        logger.info(f"Found {len(pdf_objects)} PDF files")
        
        for obj_name in pdf_objects:
            try:
                # Download to temp file
                temp_path = f"/tmp/{Path(obj_name).name}"
                self.storage.download_file(bucket, obj_name, temp_path)
                
                # Process PDF
                chunks = self.process_pdf(temp_path, {'storage_path': obj_name})
                
                # Publish chunks to Kafka
                for chunk in chunks:
                    self.producer.send(self.chunks_topic, value=chunk)
                
                logger.info(f"Published {len(chunks)} chunks from {obj_name}")
                
                # Clean up
                os.remove(temp_path)
            
            except Exception as e:
                logger.error(f"Error ingesting {obj_name}: {e}")
                continue
        
        # Flush producer
        self.producer.flush()
        logger.info("Ingestion complete")
    
    def ingest_local_directory(self, directory: str):
        """
        Ingest PDFs from local directory
        
        Args:
            directory: Path to directory containing PDFs
        """
        logger.info(f"Ingesting from local directory: {directory}")
        
        pdf_files = list(Path(directory).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            try:
                chunks = self.process_pdf(str(pdf_path))
                
                # Publish chunks
                for chunk in chunks:
                    self.producer.send(self.chunks_topic, value=chunk)
                
                logger.info(f"Published {len(chunks)} chunks from {pdf_path.name}")
            
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                continue
        
        self.producer.flush()
        logger.info("Local ingestion complete")
    
    def close(self):
        """Clean up resources"""
        self.producer.close()
        logger.info("Ingestor closed")


def main():
    """Main entry point"""
    # Load configuration
    config_path = Path("/app/config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables in config
    def expand_env(obj):
        if isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        elif isinstance(obj, str) and obj.startswith('${'):
            var_name = obj[2:-1].split(':')[0]
            default = obj[2:-1].split(':')[1] if ':' in obj else None
            return os.getenv(var_name, default)
        return obj
    
    config = expand_env(config)
    
    # Create ingestor
    ingestor = DocumentIngestor(config)
    
    # Ingest from storage
    docs_bucket = config['storage']['buckets']['docs']
    
    # Wait for MinIO to be ready
    import time
    time.sleep(10)
    
    ingestor.ingest_from_storage(docs_bucket)
    
    # Also check local data directory
    local_data_dir = Path("/app/data")
    if local_data_dir.exists():
        ingestor.ingest_local_directory(str(local_data_dir))
    
    ingestor.close()


if __name__ == "__main__":
    main()