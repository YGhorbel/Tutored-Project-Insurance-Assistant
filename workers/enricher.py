"""Text enricher - consumes chunks, enriches with LLM, publishes enriched data"""
import os
import sys
from pathlib import Path
import yaml
import asyncio
from typing import Dict
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

from workers.consumer_base import BaseKafkaConsumer
from utils.llm_client import LLMClient
from utils.storage import StorageClient
from utils.logger import setup_logger

logger = setup_logger("enricher")


class TextEnricher(BaseKafkaConsumer):
    """Enrich chunks with LLM-generated summaries and metadata"""
    
    def __init__(self, config: dict):
        """Initialize enricher"""
        self.config = config
        
        # Initialize Kafka consumer
        kafka_cfg = config['kafka']
        super().__init__(
            bootstrap_servers=kafka_cfg['bootstrap_servers'],
            group_id=kafka_cfg['consumer_group'],
            topics=[kafka_cfg['topics']['chunks']],
            auto_offset_reset=kafka_cfg.get('auto_offset_reset', 'earliest')
        )
        
        # Initialize LLM client
        llm_cfg = config['llm']
        self.llm = LLMClient(
            model_name=llm_cfg['model'],
            api_url=llm_cfg.get('api_url'),
            device=llm_cfg.get('device', 'cuda'),
            load_in_8bit=llm_cfg.get('load_in_8bit', True)
        )
        
        # Initialize storage
        storage_cfg = config['storage']
        self.storage = StorageClient(
            endpoint=storage_cfg['endpoint'],
            access_key=storage_cfg['access_key'],
            secret_key=storage_cfg['secret_key'],
            secure=storage_cfg.get('secure', False)
        )
        
        self.enriched_topic = kafka_cfg['topics']['enriched']
        self.enriched_bucket = storage_cfg['buckets']['enriched']
        self.prompts = config['prompts']
        self.concurrency = llm_cfg.get('concurrency', 4)
        
        # Create producer
        self.create_producer()
        
        # Processing queue
        self.queue = asyncio.Queue(maxsize=100)
        
        logger.info("TextEnricher initialized")
    
    def summarize_chunk(self, text: str) -> str:
        """Generate summary for chunk"""
        prompt = self.prompts['summarization'].format(text=text[:1000])
        summary = self.llm.generate(
            prompt,
            max_tokens=self.config['llm'].get('max_tokens', 256),
            temperature=self.config['llm'].get('temperature', 0.3)
        )
        return summary.strip()
    
    def classify_chunk(self, text: str) -> list:
        """Classify chunk into categories"""
        prompt = self.prompts['classification'].format(text=text[:1000])
        categories_str = self.llm.generate(
            prompt,
            max_tokens=50,
            temperature=0.1
        )
        categories = [c.strip() for c in categories_str.split(',')]
        return categories
    
    def enrich_chunk(self, chunk: Dict) -> Dict:
        """
        Enrich a chunk with LLM-generated content
        
        Args:
            chunk: Chunk dictionary
        
        Returns:
            Enriched chunk
        """
        try:
            text = chunk['text']
            
            # Generate summary
            summary = self.summarize_chunk(text)
            
            # Classify
            categories = self.classify_chunk(text)
            
            # Create enriched chunk
            enriched = {
                **chunk,
                'summary': summary,
                'categories': categories,
                'enrichment_model': self.config['llm']['model'],
                'original_length': len(text),
                'summary_length': len(summary)
            }
            
            logger.debug(f"Enriched chunk {chunk.get('chunk_id', 'unknown')}")
            return enriched
        
        except Exception as e:
            logger.error(f"Error enriching chunk: {e}")
            return chunk  # Return original chunk on error
    
    def process_message(self, message: Dict):
        """Process a single message from Kafka"""
        try:
            # Enrich chunk
            enriched_chunk = self.enrich_chunk(message)
            
            # Publish to enriched topic
            self.publish(self.enriched_topic, enriched_chunk)
            
            # Also save to storage for backup
            chunk_id = enriched_chunk['chunk_id']
            self.storage.upload_json(
                self.enriched_bucket,
                f"{chunk_id}.json",
                enriched_chunk
            )
            
            logger.info(f"Processed and published chunk {chunk_id}")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def run(self):
        """Run the enricher"""
        logger.info("Starting enricher")
        self.consume(self.process_message)


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
    
    # Wait for Kafka
    import time
    time.sleep(15)
    
    # Create and run enricher
    enricher = TextEnricher(config)
    enricher.run()


if __name__ == "__main__":
    main()