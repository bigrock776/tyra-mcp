"""
FastAPI routes for enhanced document ingestion system.

Comprehensive document ingestion API supporting multiple file formats,
dynamic chunking strategies, and LLM-enhanced context injection.
"""

import asyncio
import base64
import io
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from opentelemetry import trace

from ...core.ingestion.document_processor import DocumentProcessor
from ...core.ingestion.file_loaders import get_file_loader
from ...core.memory.manager import MemoryManager
from ...core.observability import get_tracer
from ...core.schemas.ingestion import (
    BatchIngestRequest,
    BatchIngestResponse,
    IngestRequest,
    IngestResponse,
    IngestionCapabilities,
    IngestionProgress,
    IngestionWarning,
    SupportedFormats,
)
from ...core.utils.config import get_settings
from ...core.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
tracer = get_tracer(__name__)
settings = get_settings()

# Global document processor instance
_document_processor: Optional[DocumentProcessor] = None


async def get_document_processor() -> DocumentProcessor:
    """Get or create the document processor instance."""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
        await _document_processor.initialize()
    return _document_processor


async def get_memory_manager() -> MemoryManager:
    """Get the memory manager instance."""
    # This should be injected via dependency injection in production
    from ...core.memory.manager import MemoryManager
    manager = MemoryManager()
    await manager.initialize()
    return manager


@router.post("/document", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    processor: DocumentProcessor = Depends(get_document_processor),
    memory_manager: MemoryManager = Depends(get_memory_manager),
) -> IngestResponse:
    """
    Ingest a single document with comprehensive processing.
    
    Supports multiple file types, dynamic chunking strategies,
    and LLM-enhanced context injection before embedding.
    """
    with tracer.start_as_current_span("ingest_document") as span:
        span.set_attributes({
            "file_name": request.file_name,
            "file_type": request.file_type,
            "source_type": request.source_type,
            "source_agent": request.source_agent,
            "chunking_strategy": request.chunking_strategy,
        })
        
        start_time = time.time()
        doc_id = str(uuid.uuid4())
        
        try:
            logger.info(
                "Starting document ingestion",
                doc_id=doc_id,
                file_name=request.file_name,
                file_type=request.file_type,
                source_type=request.source_type,
                source_agent=request.source_agent,
            )
            
            # Fetch content based on source type
            content_bytes = await _fetch_content(request)
            
            # Process the document
            result = await processor.process_document(
                content_bytes=content_bytes,
                file_name=request.file_name,
                file_type=request.file_type,
                doc_id=doc_id,
                source_agent=request.source_agent or "tyra",
                session_id=request.session_id,
                description=request.description or "",
                chunking_strategy=request.chunking_strategy or "auto",
                chunk_size=request.chunk_size or 512,
                chunk_overlap=request.chunk_overlap or 50,
                enable_llm_context=request.enable_llm_context,
                custom_metadata=request.metadata or {},
                memory_manager=memory_manager,
            )
            
            processing_time = time.time() - start_time
            
            logger.info(
                "Document ingestion completed",
                doc_id=doc_id,
                chunks_ingested=result.chunks_ingested,
                processing_time=processing_time,
                status=result.status,
            )
            
            span.set_attributes({
                "chunks_ingested": result.chunks_ingested,
                "processing_time": processing_time,
                "status": result.status,
            })
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            logger.error(
                "Document ingestion failed",
                doc_id=doc_id,
                file_name=request.file_name,
                error=str(e),
                processing_time=processing_time,
            )
            
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            
            # Return a failed response with error details
            from ...core.schemas.ingestion import DocumentMetadata
            
            error_metadata = DocumentMetadata(
                doc_id=doc_id,
                file_name=request.file_name,
                file_type=request.file_type,
                source_type=request.source_type,
                source_agent=request.source_agent or "tyra",
                session_id=request.session_id,
                description=request.description or "",
                chunking_strategy=request.chunking_strategy or "auto",
                chunk_size=request.chunk_size or 512,
                chunk_overlap=request.chunk_overlap or 50,
                total_chunks=0,
                total_tokens=0,
                processing_time=processing_time,
                embedding_time=0.0,
                storage_time=0.0,
                graph_time=0.0,
                llm_context_enabled=request.enable_llm_context or True,
                parsing_warnings=[f"Failed to process document: {str(e)}"],
            )
            
            return IngestResponse(
                status="failed",
                doc_id=doc_id,
                summary=f"Failed to ingest document: {str(e)}",
                chunks_ingested=0,
                total_chunks_attempted=0,
                processing_time=processing_time,
                document_metadata=error_metadata,
                chunks_metadata=[],
                warnings=[
                    IngestionWarning(
                        type="error",
                        message=str(e),
                        details={"error_type": type(e).__name__}
                    )
                ],
                embedding_time=0.0,
                storage_time=0.0,
                graph_time=0.0,
            )


@router.post("/document/upload", response_model=IngestResponse)
async def ingest_uploaded_file(
    file: UploadFile = File(...),
    source_agent: str = Form("tyra"),
    session_id: Optional[str] = Form(None),
    description: str = Form(""),
    chunking_strategy: str = Form("auto"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    enable_llm_context: bool = Form(True),
    processor: DocumentProcessor = Depends(get_document_processor),
    memory_manager: MemoryManager = Depends(get_memory_manager),
) -> IngestResponse:
    """
    Ingest an uploaded file via multipart form data.
    
    Convenient endpoint for file uploads from web interfaces or tools.
    """
    try:
        # Read file content
        content_bytes = await file.read()
        
        # Determine file type from filename
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        # Map extensions to supported types
        extension_map = {
            'pdf': 'pdf',
            'docx': 'docx',
            'pptx': 'pptx',
            'txt': 'txt',
            'md': 'md',
            'html': 'html',
            'htm': 'html',
            'json': 'json',
            'csv': 'csv',
            'epub': 'epub',
        }
        
        file_type = extension_map.get(file_extension)
        if not file_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Create ingestion request
        request = IngestRequest(
            source_type="file",
            file_name=file.filename,
            file_type=file_type,
            source_agent=source_agent,
            session_id=session_id,
            description=description,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_llm_context=enable_llm_context,
        )
        
        # Process using the same logic but with file content
        doc_id = str(uuid.uuid4())
        
        result = await processor.process_document(
            content_bytes=content_bytes,
            file_name=request.file_name,
            file_type=request.file_type,
            doc_id=doc_id,
            source_agent=request.source_agent,
            session_id=request.session_id,
            description=request.description,
            chunking_strategy=request.chunking_strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            enable_llm_context=request.enable_llm_context,
            custom_metadata=request.metadata or {},
            memory_manager=memory_manager,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"File upload ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded file: {str(e)}"
        )


@router.post("/batch", response_model=BatchIngestResponse)
async def ingest_batch(
    request: BatchIngestRequest,
    background_tasks: BackgroundTasks,
    processor: DocumentProcessor = Depends(get_document_processor),
    memory_manager: MemoryManager = Depends(get_memory_manager),
) -> BatchIngestResponse:
    """
    Ingest multiple documents in a batch with concurrent processing.
    
    Supports up to 100 documents per batch with configurable concurrency.
    """
    start_time = time.time()
    
    logger.info(
        "Starting batch ingestion",
        batch_id=request.batch_id,
        total_documents=len(request.documents),
        max_concurrent=request.max_concurrent,
        source_agent=request.source_agent,
    )
    
    # Process documents concurrently
    semaphore = asyncio.Semaphore(request.max_concurrent)
    results = []
    
    async def process_single_document(doc_request: IngestRequest) -> IngestResponse:
        async with semaphore:
            try:
                # Create a modified request for individual processing
                return await ingest_document(doc_request, processor, memory_manager)
            except Exception as e:
                # Return error response for failed documents
                error_warning = IngestionWarning(
                    type="error",
                    message=f"Failed to process {doc_request.file_name}: {str(e)}",
                    details={"file_name": doc_request.file_name}
                )
                return error_warning
    
    # Execute all ingestions concurrently
    tasks = [process_single_document(doc) for doc in request.documents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successful_ingestions = 0
    failed_ingestions = 0
    total_chunks_ingested = 0
    
    for result in results:
        if isinstance(result, IngestResponse) and result.status == "success":
            successful_ingestions += 1
            total_chunks_ingested += result.chunks_ingested
        else:
            failed_ingestions += 1
    
    processing_time = time.time() - start_time
    
    return BatchIngestResponse(
        batch_id=request.batch_id,
        status="completed" if failed_ingestions == 0 else "partial_success" if successful_ingestions > 0 else "failed",
        total_documents=len(request.documents),
        successful_ingestions=successful_ingestions,
        failed_ingestions=failed_ingestions,
        results=results,
        total_processing_time=processing_time,
        avg_processing_time=processing_time / len(request.documents),
        total_chunks_ingested=total_chunks_ingested,
        progress_percentage=100.0,
    )


@router.get("/capabilities", response_model=IngestionCapabilities)
async def get_ingestion_capabilities() -> IngestionCapabilities:
    """
    Get information about supported file formats and ingestion capabilities.
    """
    supported_formats = [
        SupportedFormats(
            format="pdf",
            extensions=[".pdf"],
            description="Portable Document Format files",
            max_file_size="50MB",
            chunking_strategies=["semantic", "paragraph", "page"],
            features=["Text extraction", "Metadata extraction", "Page-based chunking"],
            limitations=["OCR not supported for image-only PDFs"]
        ),
        SupportedFormats(
            format="docx",
            extensions=[".docx"],
            description="Microsoft Word documents",
            max_file_size="25MB",
            chunking_strategies=["paragraph", "section", "auto"],
            features=["Paragraph detection", "Style preservation", "Table extraction"],
            limitations=["Images not processed"]
        ),
        SupportedFormats(
            format="pptx",
            extensions=[".pptx"],
            description="Microsoft PowerPoint presentations",
            max_file_size="25MB",
            chunking_strategies=["slide", "auto"],
            features=["Slide-based chunking", "Speaker notes extraction"],
            limitations=["Images not processed"]
        ),
        SupportedFormats(
            format="txt",
            extensions=[".txt"],
            description="Plain text files",
            max_file_size="10MB",
            chunking_strategies=["paragraph", "line", "semantic", "auto"],
            features=["Encoding detection", "Line-based processing"],
            limitations=[]
        ),
        SupportedFormats(
            format="md",
            extensions=[".md", ".markdown"],
            description="Markdown files",
            max_file_size="10MB",
            chunking_strategies=["paragraph", "section", "auto"],
            features=["Markdown parsing", "Header detection"],
            limitations=[]
        ),
        SupportedFormats(
            format="html",
            extensions=[".html", ".htm"],
            description="HTML files",
            max_file_size="10MB",
            chunking_strategies=["paragraph", "semantic", "auto"],
            features=["HTML to text conversion", "Structure preservation"],
            limitations=["JavaScript not executed"]
        ),
        SupportedFormats(
            format="json",
            extensions=[".json"],
            description="JSON data files",
            max_file_size="50MB",
            chunking_strategies=["object", "array", "auto"],
            features=["Structured data processing", "Nested object handling"],
            limitations=[]
        ),
        SupportedFormats(
            format="csv",
            extensions=[".csv"],
            description="Comma-separated values files",
            max_file_size="100MB",
            chunking_strategies=["row", "auto"],
            features=["Header detection", "Streaming processing"],
            limitations=[]
        ),
        SupportedFormats(
            format="epub",
            extensions=[".epub"],
            description="Electronic publication format",
            max_file_size="25MB",
            chunking_strategies=["chapter", "semantic", "auto"],
            features=["Chapter extraction", "Metadata extraction"],
            limitations=["DRM-protected files not supported"]
        ),
    ]
    
    return IngestionCapabilities(
        supported_formats=supported_formats,
        chunking_strategies=["auto", "paragraph", "semantic", "slide", "line", "token", "section", "page", "object", "array", "row", "chapter"],
        max_file_size="100MB",
        max_batch_size=100,
        concurrent_limit=20,
        features=[
            "Multi-format support",
            "Dynamic chunking strategies",
            "LLM-enhanced context injection",
            "Batch processing",
            "Streaming for large files",
            "Comprehensive metadata tracking",
            "Hallucination detection",
            "Graph relationship extraction",
            "Concurrent processing",
            "Progress tracking"
        ],
        version="1.0.0"
    )


@router.get("/progress/{job_id}", response_model=IngestionProgress)
async def get_ingestion_progress(job_id: str) -> IngestionProgress:
    """
    Get the progress of a long-running ingestion job.
    
    For future implementation of background job tracking.
    """
    # This would integrate with a job queue system like Celery or RQ
    # For now, return a placeholder
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Progress tracking not yet implemented"
    )


async def _fetch_content(request: IngestRequest) -> bytes:
    """
    Fetch content based on the source type.
    """
    if request.source_type == "base64":
        try:
            return base64.b64decode(request.content)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 content: {str(e)}"
            )
    
    elif request.source_type == "url":
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(str(request.file_url))
                response.raise_for_status()
                return response.content
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to fetch content from URL: {str(e)}"
            )
    
    elif request.source_type == "file":
        # This would be used for server-side file processing
        # For now, this is not supported in the API
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File source type not supported in API. Use upload endpoint instead."
        )
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported source type: {request.source_type}"
        )