"""Tests for document ingestor"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from workers.ingestor import DocumentIngestor


@pytest.fixture
def mock_config():
    """Mock configuration"""
    return {
        'storage': {
            'endpoint': 'localhost:9000',
            'access_key': 'minioadmin',
            'secret_key': 'minioadmin',
            'secure': False,
            'buckets': {
                'docs': 'test-docs',
                'chunks': 'test-chunks'
            }
        },
        'kafka': {
            'bootstrap_servers': 'localhost:9092',
            'topics': {
                'chunks': 'test-chunks'
            }
        },
        'processing': {
            'chunk_size': 512,
            'chunk_overlap': 50
        }
    }


@pytest.fixture
def ingestor(mock_config):
    """Create ingestor with mocked dependencies"""
    with patch('workers.ingestor.StorageClient'), \
         patch('workers.ingestor.KafkaProducer'):
        return DocumentIngestor(mock_config)


def test_chunk_text(ingestor):
    """Test text chunking"""
    text = "This is a test. " * 100  # Create long text
    metadata = {'source': 'test.pdf', 'page': 1}
    
    chunks = ingestor.chunk_text(text, metadata)
    
    assert len(chunks) > 0
    assert all('chunk_id' in chunk for chunk in chunks)
    assert all('text' in chunk for chunk in chunks)
    assert all('metadata' in chunk for chunk in chunks)
    assert chunks[0]['chunk_index'] == 0


def test_chunk_text_overlap(ingestor):
    """Test that chunks overlap correctly"""
    text = "word " * 200
    metadata = {'source': 'test.pdf'}
    
    chunks = ingestor.chunk_text(text, metadata)
    
    # Check that consecutive chunks have some overlap
    if len(chunks) > 1:
        # Some words from end of first chunk should appear in second
        first_chunk_words = chunks[0]['text'].split()[-10:]
        second_chunk_words = chunks[1]['text'].split()[:10]
        
        # There should be some common words due to overlap
        common_words = set(first_chunk_words) & set(second_chunk_words)
        assert len(common_words) > 0


def test_process_pdf_mock(ingestor):
    """Test PDF processing with mocked PDF reader"""
    with patch('workers.ingestor.PdfReader') as mock_reader:
        # Setup mock
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test content " * 50
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page, mock_page]
        mock_reader.return_value = mock_reader_instance
        
        # Process
        chunks = ingestor.process_pdf('/fake/path.pdf')
        
        assert len(chunks) > 0
        assert all(chunk['metadata']['page'] in [1, 2] for chunk in chunks)
        assert all(chunk['metadata']['total_pages'] == 2 for chunk in chunks)


def test_process_pdf_error_handling(ingestor):
    """Test that PDF processing handles errors gracefully"""
    with patch('workers.ingestor.PdfReader') as mock_reader:
        mock_reader.side_effect = Exception("PDF read error")
        
        chunks = ingestor.process_pdf('/fake/path.pdf')
        
        # Should return empty list on error
        assert chunks == []


def test_ingest_from_storage_mock(ingestor):
    """Test ingesting from storage"""
    # Mock storage client
    ingestor.storage.list_objects = Mock(return_value=['doc1.pdf', 'doc2.pdf'])
    ingestor.storage.download_file = Mock()
    
    # Mock process_pdf
    with patch.object(ingestor, 'process_pdf') as mock_process:
        mock_process.return_value = [
            {'chunk_id': '123', 'text': 'test', 'metadata': {}}
        ]
        
        # Mock producer
        ingestor.producer.send = Mock()
        ingestor.producer.flush = Mock()
        
        # Ingest
        ingestor.ingest_from_storage('test-bucket')
        
        # Verify
        assert ingestor.storage.list_objects.called
        assert mock_process.call_count == 2
        assert ingestor.producer.send.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])