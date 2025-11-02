"""Base Kafka consumer with common functionality"""
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from typing import Callable, Optional, Dict, Any
from loguru import logger
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class BaseKafkaConsumer:
    """Base class for Kafka consumers with error handling and retry logic"""
    
    def __init__(self, bootstrap_servers: str, group_id: str, 
                 topics: list, auto_offset_reset: str = 'earliest'):
        """
        Initialize Kafka consumer
        
        Args:
            bootstrap_servers: Kafka broker addresses
            group_id: Consumer group ID
            topics: List of topics to subscribe to
            auto_offset_reset: Offset reset policy
        """
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        
        logger.info(f"Initializing consumer for topics: {topics}")
        self._connect_consumer(auto_offset_reset)
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60))
    def _connect_consumer(self, auto_offset_reset: str):
        """Connect to Kafka with retry logic"""
        try:
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=auto_offset_reset,
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                max_poll_records=10,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000
            )
            logger.info(f"Connected to Kafka: {self.bootstrap_servers}")
        except KafkaError as e:
            logger.error(f"Kafka connection error: {e}")
            raise
    
    def create_producer(self):
        """Create Kafka producer for publishing results"""
        if self.producer is None:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            logger.info("Kafka producer created")
    
    def publish(self, topic: str, message: Dict[Any, Any], key: Optional[str] = None):
        """Publish message to topic"""
        if self.producer is None:
            self.create_producer()
        
        try:
            future = self.producer.send(
                topic,
                value=message,
                key=key.encode('utf-8') if key else None
            )
            future.get(timeout=10)
            logger.debug(f"Published to {topic}")
        except Exception as e:
            logger.error(f"Publish error: {e}")
            raise
    
    def consume(self, process_func: Callable[[Dict], None], 
                max_messages: Optional[int] = None):
        """
        Consume messages and process them
        
        Args:
            process_func: Function to process each message
            max_messages: Maximum messages to process (None for infinite)
        """
        processed = 0
        
        logger.info(f"Starting consumption from {self.topics}")
        
        try:
            for message in self.consumer:
                try:
                    logger.debug(f"Received message from {message.topic}, offset {message.offset}")
                    process_func(message.value)
                    processed += 1
                    
                    if max_messages and processed >= max_messages:
                        logger.info(f"Processed {processed} messages, stopping")
                        break
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Continue processing next message
                    continue
        
        except KeyboardInterrupt:
            logger.info("Consumption interrupted by user")
        
        finally:
            self.close()
    
    def close(self):
        """Close consumer and producer"""
        if self.consumer:
            self.consumer.close()
            logger.info("Consumer closed")
        
        if self.producer:
            self.producer.close()
            logger.info("Producer closed")