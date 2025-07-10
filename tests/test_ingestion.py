"""
Comprehensive tests for the enhanced document ingestion system.

Tests all file types, chunking strategies, error handling, and integration
with the memory system for complete validation of ingestion capabilities.
"""

import asyncio
import base64
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.ingestion.chunking_strategies import (
    AutoChunkingStrategy,
    ParagraphChunkingStrategy,
    SemanticChunkingStrategy,
    chunk_content,
)
from src.core.ingestion.document_processor import DocumentProcessor
from src.core.ingestion.file_loaders import (
    CSVLoader,
    DOCXLoader,
    HTMLLoader,
    JSONLoader,
    PDFLoader,
    TextLoader,
    get_file_loader,
)
from src.core.ingestion.llm_context_enhancer import LLMContextEnhancer
from src.core.schemas.ingestion import IngestRequest, IngestResponse


@pytest.fixture
def sample_text():
    """Sample text content for testing."""
    return """
    This is a test document with multiple paragraphs.
    
    The first paragraph contains some introductory information about the topic.
    It establishes the context and purpose of the document.
    
    The second paragraph goes into more detail about specific aspects.
    It provides examples and explanations that help clarify the concepts.
    This paragraph is intentionally longer to test chunking behavior.
    
    Finally, the third paragraph summarizes the key points.
    It draws conclusions and provides recommendations for future work.
    """


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing."""
    return {
        "title": "Test Document",
        "author": "Test Author",
        "content": [
            {"section": "Introduction", "text": "This is the introduction."},
            {"section": "Main Content", "text": "This is the main content section."},
            {"section": "Conclusion", "text": "This is the conclusion."}
        ],
        "metadata": {
            "created": "2024-01-01",
            "version": "1.0"
        }
    }


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return """name,age,city
John Doe,30,New York
Jane Smith,25,Los Angeles
Bob Johnson,35,Chicago
Alice Brown,28,Houston"""


@pytest.fixture
def sample_html():
    """Sample HTML content for testing."""
    return """
    <html>
    <head>
        <title>Test Document</title>
        <meta name="description" content="A test HTML document">
    </head>
    <body>
        <h1>Main Title</h1>
        <p>This is the first paragraph with some content.</p>
        <h2>Subsection</h2>
        <p>This is another paragraph in a subsection.</p>
        <ul>
            <li>First item</li>
            <li>Second item</li>
        </ul>
    </body>
    </html>
    """


class TestFileLoaders:
    """Test file loader implementations."""
    
    @pytest.mark.asyncio
    async def test_text_loader(self, sample_text):
        """Test text file loader."""
        loader = TextLoader()
        content_bytes = sample_text.encode('utf-8')
        
        result = await loader.load(content_bytes, "test.txt")
        
        assert result.text == sample_text
        assert len(result.chunks) > 0
        assert result.metadata["format"] == "Text"
        assert result.metadata["file_size"] == len(content_bytes)
    
    @pytest.mark.asyncio
    async def test_markdown_loader(self, sample_text):
        """Test markdown file loader."""
        markdown_content = """# Main Title
        
        ## Subsection
        
        This is a paragraph with **bold** and *italic* text.
        
        - First item
        - Second item
        
        ### Another section
        
        Final paragraph with content.
        """
        
        loader = TextLoader()
        content_bytes = markdown_content.encode('utf-8')
        
        result = await loader.load(content_bytes, "test.md")
        
        assert result.text == markdown_content
        assert result.metadata["format"] == "Markdown"
        assert len(result.chunks) > 0
    
    @pytest.mark.asyncio
    async def test_json_loader(self, sample_json_data):
        """Test JSON file loader."""
        loader = JSONLoader()
        content_bytes = json.dumps(sample_json_data).encode('utf-8')
        
        result = await loader.load(content_bytes, "test.json")
        
        assert "title" in result.text
        assert "Test Document" in result.text
        assert result.metadata["format"] == "JSON"
        assert result.metadata["json_type"] == "dict"
        assert len(result.chunks) > 0
    
    @pytest.mark.asyncio
    async def test_csv_loader(self, sample_csv_data):
        """Test CSV file loader."""
        loader = CSVLoader()
        content_bytes = sample_csv_data.encode('utf-8')
        
        result = await loader.load(content_bytes, "test.csv")
        
        assert "Headers:" in result.text
        assert "name" in result.text
        assert result.metadata["format"] == "CSV"
        assert result.metadata["row_count"] == 4
        assert result.metadata["column_count"] == 3
        assert len(result.chunks) == 4  # One chunk per row
    
    @pytest.mark.asyncio
    async def test_html_loader(self, sample_html):
        """Test HTML file loader."""
        loader = HTMLLoader()
        content_bytes = sample_html.encode('utf-8')
        
        result = await loader.load(content_bytes, "test.html")
        
        assert "Main Title" in result.text
        assert "first paragraph" in result.text
        assert result.metadata["format"] == "HTML"
        assert result.metadata["title"] == "Test Document"
        assert len(result.chunks) > 0
    
    def test_get_file_loader(self):
        """Test file loader registry."""
        # Test supported types
        pdf_loader = get_file_loader("pdf")
        assert isinstance(pdf_loader, PDFLoader)
        
        text_loader = get_file_loader("txt")
        assert isinstance(text_loader, TextLoader)
        
        json_loader = get_file_loader("json")
        assert isinstance(json_loader, JSONLoader)
        
        # Test unsupported type
        with pytest.raises(ValueError, match="No loader available"):
            get_file_loader("unsupported")


class TestChunkingStrategies:
    """Test chunking strategy implementations."""
    
    @pytest.mark.asyncio
    async def test_paragraph_chunking(self, sample_text):
        """Test paragraph-based chunking."""
        strategy = ParagraphChunkingStrategy(chunk_size=200, chunk_overlap=20)
        result = await strategy.chunk(sample_text, {})
        
        assert result.strategy_used == "paragraph"
        assert len(result.chunks) > 1
        assert all(chunk["chunk_type"] in ["paragraph_group", "paragraph_split"] for chunk in result.chunks)
        assert result.total_tokens > 0
    
    @pytest.mark.asyncio
    async def test_semantic_chunking(self, sample_text):
        """Test semantic-based chunking."""
        strategy = SemanticChunkingStrategy(chunk_size=300, chunk_overlap=30)
        result = await strategy.chunk(sample_text, {})
        
        assert result.strategy_used == "semantic"
        assert len(result.chunks) > 0
        assert all(chunk["chunk_type"] == "semantic" for chunk in result.chunks)
    
    @pytest.mark.asyncio
    async def test_auto_chunking_pdf(self):
        """Test auto chunking strategy for PDF."""
        text = "This is a test PDF content with semantic meaning."
        metadata = {"format": "PDF"}
        
        strategy = AutoChunkingStrategy(chunk_size=100)
        result = await strategy.chunk(text, metadata)
        
        assert "auto_semantic" in result.strategy_used
        assert result.metadata["auto_selected_strategy"] == "semantic"
    
    @pytest.mark.asyncio
    async def test_auto_chunking_docx(self, sample_text):
        """Test auto chunking strategy for DOCX."""
        metadata = {"format": "DOCX"}
        
        strategy = AutoChunkingStrategy(chunk_size=200)
        result = await strategy.chunk(sample_text, metadata)
        
        assert "auto_paragraph" in result.strategy_used
        assert result.metadata["auto_selected_strategy"] == "paragraph"
    
    @pytest.mark.asyncio
    async def test_chunk_content_convenience_function(self, sample_text):
        """Test the convenience chunk_content function."""
        result = await chunk_content(
            text=sample_text,
            strategy_name="paragraph",
            chunk_size=150,
            chunk_overlap=15,
            metadata={"format": "text"}
        )
        
        assert result.strategy_used == "paragraph"
        assert len(result.chunks) > 0
    
    @pytest.mark.asyncio
    async def test_chunking_with_existing_chunks(self):
        """Test chunking with pre-existing chunks from file loader."""
        existing_chunks = [
            {"text": "First chunk", "chunk_type": "paragraph"},
            {"text": "Second chunk", "chunk_type": "paragraph"},
        ]
        
        strategy = ParagraphChunkingStrategy(chunk_size=500)
        result = await strategy.chunk("", {}, existing_chunks)
        
        # Should combine small chunks
        assert len(result.chunks) == 1
        assert "First chunk" in result.chunks[0]["text"]
        assert "Second chunk" in result.chunks[0]["text"]


class TestLLMContextEnhancer:
    """Test LLM context enhancement functionality."""
    
    @pytest.mark.asyncio
    async def test_enhance_chunks(self):
        """Test chunk enhancement with context."""
        enhancer = LLMContextEnhancer()
        await enhancer.initialize()
        
        chunks = [
            {"text": "This is a test chunk about machine learning.", "chunk_type": "paragraph"},
            {"text": "Another chunk with technical content about APIs.", "chunk_type": "paragraph"},
        ]
        
        document_context = {
            "file_name": "test_document.pdf",
            "file_type": "pdf",
            "description": "Technical documentation",
        }
        
        enhanced = await enhancer.enhance_chunks(chunks, document_context)
        
        assert len(enhanced) == len(chunks)
        for chunk in enhanced:
            assert "enhanced_context" in chunk
            assert "confidence_score" in chunk
            assert "hallucination_score" in chunk
            assert 0.0 <= chunk["confidence_score"] <= 1.0
            assert 0.0 <= chunk["hallucination_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_enhancement_with_empty_chunks(self):
        """Test enhancement with empty chunk list."""
        enhancer = LLMContextEnhancer()
        await enhancer.initialize()
        
        enhanced = await enhancer.enhance_chunks([], {})
        assert enhanced == []
    
    @pytest.mark.asyncio
    async def test_enhancement_stats(self):
        """Test enhancement statistics tracking."""
        enhancer = LLMContextEnhancer()
        await enhancer.initialize()
        
        chunks = [{"text": "Test chunk", "chunk_type": "test"}]
        await enhancer.enhance_chunks(chunks, {"file_name": "test.txt"})
        
        stats = await enhancer.get_enhancement_stats()
        assert stats["total_chunks_processed"] >= 1
        assert stats["total_enhancement_time"] > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test LLM enhancer health check."""
        enhancer = LLMContextEnhancer()
        await enhancer.initialize()
        
        health = await enhancer.health_check()
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "response_time" in health


class TestDocumentProcessor:
    """Test the main document processor."""
    
    @pytest.fixture
    async def processor(self):
        """Create and initialize document processor."""
        processor = DocumentProcessor()
        await processor.initialize()
        return processor
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager for testing."""
        mock_manager = AsyncMock()
        mock_manager.store_memory.return_value = MagicMock(
            entities_created=["entity1", "entity2"],
            relationships_created=["rel1"],
        )
        return mock_manager
    
    @pytest.mark.asyncio
    async def test_process_text_document(self, processor, mock_memory_manager, sample_text):
        """Test processing a text document."""
        content_bytes = sample_text.encode('utf-8')
        
        result = await processor.process_document(
            content_bytes=content_bytes,
            file_name="test.txt",
            file_type="txt",
            doc_id="test-doc-1",
            source_agent="test-agent",
            description="Test text document",
            chunking_strategy="paragraph",
            chunk_size=200,
            enable_llm_context=True,
            memory_manager=mock_memory_manager,
        )
        
        assert isinstance(result, IngestResponse)
        assert result.status == "success"
        assert result.doc_id == "test-doc-1"
        assert result.chunks_ingested > 0
        assert result.document_metadata.file_name == "test.txt"
        assert result.document_metadata.file_type == "txt"
        assert len(result.chunks_metadata) > 0
        
        # Verify memory manager was called
        assert mock_memory_manager.store_memory.called
    
    @pytest.mark.asyncio
    async def test_process_json_document(self, processor, mock_memory_manager, sample_json_data):
        """Test processing a JSON document."""
        content_bytes = json.dumps(sample_json_data).encode('utf-8')
        
        result = await processor.process_document(
            content_bytes=content_bytes,
            file_name="test.json",
            file_type="json",
            doc_id="test-doc-2",
            chunking_strategy="auto",
            memory_manager=mock_memory_manager,
        )
        
        assert result.status == "success"
        assert result.document_metadata.file_type == "json"
        assert result.chunks_ingested > 0
    
    @pytest.mark.asyncio
    async def test_process_document_without_memory_manager(self, processor, sample_text):
        """Test processing without memory manager."""
        content_bytes = sample_text.encode('utf-8')
        
        result = await processor.process_document(
            content_bytes=content_bytes,
            file_name="test.txt",
            file_type="txt",
            doc_id="test-doc-3",
            memory_manager=None,  # No memory manager
        )
        
        assert result.status == "success"
        assert len(result.entities_created) == 0
        assert len(result.relationships_created) == 0
    
    @pytest.mark.asyncio
    async def test_process_document_with_error(self, processor):
        """Test processing with invalid content."""
        # Invalid content for PDF
        content_bytes = b"This is not a valid PDF file"
        
        result = await processor.process_document(
            content_bytes=content_bytes,
            file_name="invalid.pdf",
            file_type="pdf",
            doc_id="test-doc-4",
        )
        
        assert result.status == "failed"
        assert len(result.warnings) > 0
        assert result.chunks_ingested == 0
    
    @pytest.mark.asyncio
    async def test_processor_stats(self, processor, sample_text):
        """Test processor statistics tracking."""
        content_bytes = sample_text.encode('utf-8')
        
        # Process a document
        await processor.process_document(
            content_bytes=content_bytes,
            file_name="test.txt",
            file_type="txt",
            doc_id="test-doc-5",
        )
        
        stats = await processor.get_processing_stats()
        assert stats["total_documents"] >= 1
        assert stats["total_chunks"] > 0
        assert stats["avg_processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_processor_health_check(self, processor):
        """Test processor health check."""
        health = await processor.health_check()
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "components" in health


class TestAPIEndpoints:
    """Test FastAPI ingestion endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def sample_request_data(self, sample_text):
        """Sample request data for API testing."""
        return {
            "source_type": "base64",
            "file_name": "test.txt",
            "file_type": "txt",
            "content": base64.b64encode(sample_text.encode()).decode(),
            "source_agent": "test-agent",
            "description": "Test document",
            "chunking_strategy": "paragraph",
            "chunk_size": 200,
            "chunk_overlap": 20,
        }
    
    @patch('src.api.routes.ingestion.get_document_processor')
    @patch('src.api.routes.ingestion.get_memory_manager')
    def test_ingest_document_endpoint(self, mock_memory_manager, mock_processor, client, sample_request_data):
        """Test the main document ingestion endpoint."""
        # Mock the processor to return a successful response
        mock_proc_instance = AsyncMock()
        mock_proc_instance.process_document.return_value = IngestResponse(
            status="success",
            doc_id="test-doc-1",
            summary="Test successful",
            chunks_ingested=3,
            total_chunks_attempted=3,
            processing_time=1.0,
            document_metadata=MagicMock(),
            chunks_metadata=[],
            embedding_time=0.1,
            storage_time=0.2,
            graph_time=0.1,
        )
        mock_processor.return_value = mock_proc_instance
        mock_memory_manager.return_value = AsyncMock()
        
        response = client.post("/v1/ingest/document", json=sample_request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_ingested"] > 0
    
    def test_get_capabilities_endpoint(self, client):
        """Test the ingestion capabilities endpoint."""
        response = client.get("/v1/ingest/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        assert "supported_formats" in data
        assert "chunking_strategies" in data
        assert len(data["supported_formats"]) > 0
        
        # Check for specific formats
        format_names = [fmt["format"] for fmt in data["supported_formats"]]
        assert "pdf" in format_names
        assert "docx" in format_names
        assert "txt" in format_names
    
    def test_invalid_request_validation(self, client):
        """Test request validation for invalid data."""
        invalid_data = {
            "source_type": "base64",
            "file_name": "test.txt",
            "file_type": "unsupported_type",  # Invalid type
            "content": "dGVzdA=="
        }
        
        response = client.post("/v1/ingest/document", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_missing_content_validation(self, client):
        """Test validation when required content is missing."""
        invalid_data = {
            "source_type": "base64",
            "file_name": "test.txt",
            "file_type": "txt",
            # Missing required content
        }
        
        response = client.post("/v1/ingest/document", json=invalid_data)
        assert response.status_code == 422


class TestIngestionSchemas:
    """Test Pydantic schemas for ingestion."""
    
    def test_ingest_request_validation(self):
        """Test IngestRequest validation."""
        # Valid request
        valid_data = {
            "source_type": "base64",
            "file_name": "test.txt",
            "file_type": "txt",
            "content": "dGVzdCBjb250ZW50",
            "source_agent": "test-agent",
        }
        
        request = IngestRequest(**valid_data)
        assert request.source_type == "base64"
        assert request.file_name == "test.txt"
        assert request.file_type == "txt"
    
    def test_ingest_request_validation_errors(self):
        """Test IngestRequest validation errors."""
        # Missing content for base64 source
        with pytest.raises(ValueError, match="content is required"):
            IngestRequest(
                source_type="base64",
                file_name="test.txt",
                file_type="txt",
                # Missing content
            )
        
        # Providing content for file source
        with pytest.raises(ValueError, match="content should not be provided"):
            IngestRequest(
                source_type="file",
                file_name="test.txt",
                file_type="txt",
                content="some content",
            )
    
    def test_chunk_size_validation(self):
        """Test chunk size validation."""
        # Too small chunk size
        with pytest.raises(ValueError):
            IngestRequest(
                source_type="base64",
                file_name="test.txt",
                file_type="txt",
                content="dGVzdA==",
                chunk_size=10,  # Too small
            )
        
        # Too large chunk size
        with pytest.raises(ValueError):
            IngestRequest(
                source_type="base64",
                file_name="test.txt",
                file_type="txt",
                content="dGVzdA==",
                chunk_size=5000,  # Too large
            )


class TestErrorHandling:
    """Test error handling in ingestion system."""
    
    @pytest.mark.asyncio
    async def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        with pytest.raises(ValueError, match="No loader available"):
            get_file_loader("unsupported")
    
    @pytest.mark.asyncio
    async def test_invalid_json_content(self):
        """Test handling of invalid JSON content."""
        loader = JSONLoader()
        invalid_json = b"{ invalid json content"
        
        with pytest.raises(ValueError, match="Invalid JSON content"):
            await loader.load(invalid_json, "test.json")
    
    @pytest.mark.asyncio
    async def test_empty_content_handling(self):
        """Test handling of empty content."""
        loader = TextLoader()
        empty_content = b""
        
        result = await loader.load(empty_content, "empty.txt")
        assert result.text == ""
        assert len(result.chunks) == 0
    
    @pytest.mark.asyncio
    async def test_large_content_streaming(self):
        """Test handling of large content (future streaming feature)."""
        # This would test the streaming pipeline for files > 10MB
        # For now, just test that normal processing works with larger content
        large_text = "This is a test paragraph.\n\n" * 1000  # Large text
        
        result = await chunk_content(
            text=large_text,
            strategy_name="paragraph",
            chunk_size=512,
        )
        
        assert len(result.chunks) > 10  # Should create many chunks
        assert result.total_tokens > 1000


@pytest.mark.integration
class TestIntegrationWithMemorySystem:
    """Integration tests with the memory system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_ingestion(self, sample_text):
        """Test complete end-to-end ingestion flow."""
        # This would test the complete flow from API to memory storage
        # For now, test the document processor integration
        
        processor = DocumentProcessor()
        await processor.initialize()
        
        mock_memory_manager = AsyncMock()
        mock_memory_manager.store_memory.return_value = MagicMock(
            entities_created=["entity1"],
            relationships_created=["rel1"],
        )
        
        content_bytes = sample_text.encode('utf-8')
        
        result = await processor.process_document(
            content_bytes=content_bytes,
            file_name="integration_test.txt",
            file_type="txt",
            doc_id="integration-test-1",
            memory_manager=mock_memory_manager,
        )
        
        assert result.status == "success"
        assert len(result.entities_created) > 0
        assert mock_memory_manager.store_memory.called
        
        # Verify chunks were enhanced
        assert any(
            chunk.enhanced_context for chunk in result.chunks_metadata
            if chunk.enhanced_context
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])