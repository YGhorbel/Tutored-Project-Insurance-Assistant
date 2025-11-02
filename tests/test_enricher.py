"""Tests for text enricher"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from workers.enricher import TextEnricher


@pytest.fixture
def mock_config():
    """Mock configuration"""
    return {
        'kafka': {
            'bootstrap_servers': 'localhost:9092',
            'consumer_group': 'test-group',
            'topics': {
                'chunks': 'test-chunks',
                'enriched': 'test-enriched'
            }
        },
        'storage': {
            'endpoint': 'localhost:9000',
            'access_key': 'test',
            'secret_key': 'test',
            'secure': False,
            'buckets': {
                'enriched': 'test-enriched'
            }
        },
        'llm': {
            'model': 'test-model',
            'device': 'cpu',
            'load_in_8bit': False,
            'max_tokens': 256,
            'temperature': 0.3,
            'concurrency': 2
        },
        'prompts': {
            'summarization': 'Summarize: {text}',
            'classification': 'Classify: {text}'
        }
    }


@pytest.fixture
def enricher(mock_config):
    """Create enricher with mocked dependencies"""
    with patch('workers.enricher.BaseKafkaConsumer.__init__', return_value=None), \
         patch('workers.enricher.LLMClient'), \
         patch('workers.enricher.StorageClient'):
        enricher = TextEnricher.__new__(TextEnricher)
        enricher.config = mock_config
        enricher.llm = Mock()
        enricher.storage = Mock()
        enricher.enriched_topic = 'test-enriched'
        enricher.enriched_bucket = 'test-enriched'
        enricher.prompts = mock_config['prompts']
        enricher.producer = Mock()
        return enricher


def test_summarize_chunk(enricher):
    """Test chunk summarization"""
    text = "This is a test insurance regulation document."
    enricher.llm.generate.return_value = "Summary of regulation"
    
    summary = enricher.summarize_chunk(text)
    
    assert summary == "Summary of regulation"
    assert enricher.llm.generate.called


def test_classify_chunk(enricher):
    """Test chunk classification"""
    text = "Life insurance policy requirements"
    enricher.llm.generate.return_value = "Life Insurance, Consumer Protection"
    
    categories = enricher.classify_chunk(text)
    
    assert len(categories) == 2
    assert "Life Insurance" in categories
    assert "Consumer Protection" in categories


def test_enrich_chunk_success(enricher):
    """Test successful chunk enrichment"""
    chunk = {
        'chunk_id': '123',
        'text': 'Test regulation text',
        'metadata': {'source': 'test.pdf'}
    }
    
    enricher.llm.generate.side_effect = [
        "Test summary",
        "Life Insurance, Health Insurance"
    ]
    
    enriched = enricher.enrich_chunk(chunk)
    
    assert enriched['chunk_id'] == '123'
    assert enriched['summary'] == "Test summary"
    assert 'Life Insurance' in enriched['categories']
    assert enriched['original_length'] == len('Test regulation text')


def test_enrich_chunk_error_handling(enricher):
    """Test that enrichment handles errors gracefully"""
    chunk = {
        'chunk_id': '123',
        'text': 'Test text',
        'metadata': {}
    }
    
    enricher.llm.generate.side_effect = Exception("LLM error")
    
    # Should return original chunk on error
    enriched = enricher.enrich_chunk(chunk)
    
    assert enriched == chunk


def test_process_message(enricher):
    """Test message processing"""
    message = {
        'chunk_id': '123',
        'text': 'Test text',
        'metadata': {}
    }
    
    enricher.llm.generate.side_effect = [
        "Summary",
        "Category"
    ]
    enricher.publish = Mock()
    enricher.storage.upload_json = Mock()
    
    enricher.process_message(message)
    
    assert enricher.publish.called
    assert enricher.storage.upload_json.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])