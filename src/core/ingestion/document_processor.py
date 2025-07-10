"""
Main document processor orchestrating the complete ingestion pipeline.

Coordinates file loading, chunking, LLM enhancement, embedding, and storage
for comprehensive document ingestion with metadata tracking and quality scoring.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from ..memory.manager import MemoryManager, MemoryStoreRequest
from ..observability import get_tracer
from ..schemas.ingestion import (
    ChunkMetadata,
    DocumentMetadata,
    IngestResponse,
    IngestionWarning,
)
from ..utils.logger import get_logger
from .chunking_strategies import chunk_content
from .file_loaders import get_file_loader
from .llm_context_enhancer import LLMContextEnhancer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class DocumentProcessor:
    """
    Main document processor coordinating the complete ingestion pipeline.
    
    Features:
    - Multi-format file loading
    - Dynamic chunking strategies
    - LLM-enhanced context injection
    - Comprehensive metadata tracking
    - Quality and hallucination scoring
    - Streaming support for large files
    - Performance monitoring
    """
    
    def __init__(self):
        self.llm_enhancer: Optional[LLMContextEnhancer] = None
        self._processing_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
        }
    
    async def initialize(self) -> None:
        """Initialize the document processor."""
        try:
            # Initialize LLM context enhancer
            self.llm_enhancer = LLMContextEnhancer()
            await self.llm_enhancer.initialize()
            
            logger.info("Document processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize document processor: {str(e)}")
            raise
    
    async def process_document(
        self,
        content_bytes: bytes,
        file_name: str,
        file_type: str,
        doc_id: str,
        source_agent: str = "tyra",
        session_id: Optional[str] = None,
        description: str = "",
        chunking_strategy: str = "auto",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        enable_llm_context: bool = True,
        custom_metadata: Optional[Dict[str, Any]] = None,
        memory_manager: Optional[MemoryManager] = None,
    ) -> IngestResponse:
        """
        Process a document through the complete ingestion pipeline.
        """
        with tracer.start_as_current_span("process_document") as span:
            span.set_attributes({
                "doc_id": doc_id,
                "file_name": file_name,
                "file_type": file_type,
                "chunking_strategy": chunking_strategy,
                "enable_llm_context": enable_llm_context,
            })
            
            start_time = time.time()
            warnings = []
            
            try:
                # Step 1: Load and parse the file
                load_start = time.time()
                loader = get_file_loader(file_type)
                parsed_content = await loader.load(content_bytes, file_name)
                load_time = time.time() - load_start
                
                logger.info(
                    "File loaded successfully",
                    doc_id=doc_id,
                    file_type=file_type,
                    text_length=len(parsed_content.text),
                    initial_chunks=len(parsed_content.chunks),
                    load_time=load_time,
                )
                
                # Add any loader warnings
                warnings.extend([
                    IngestionWarning(type="warning", message=w)
                    for w in parsed_content.warnings
                ])
                
                # Step 2: Apply chunking strategy
                chunk_start = time.time()
                chunk_result = await chunk_content(
                    text=parsed_content.text,
                    strategy_name=chunking_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    metadata=parsed_content.metadata,
                    existing_chunks=parsed_content.chunks,
                )
                chunk_time = time.time() - chunk_start
                
                logger.info(
                    "Content chunked successfully",
                    doc_id=doc_id,
                    strategy_used=chunk_result.strategy_used,
                    total_chunks=len(chunk_result.chunks),
                    avg_chunk_size=chunk_result.avg_chunk_size,
                    chunk_time=chunk_time,
                )
                
                # Step 3: LLM context enhancement (if enabled)
                enhancement_start = time.time()
                enhanced_chunks = []
                
                if enable_llm_context and self.llm_enhancer:
                    try:
                        enhanced_chunks = await self.llm_enhancer.enhance_chunks(
                            chunks=chunk_result.chunks,
                            document_context={
                                "file_name": file_name,
                                "file_type": file_type,
                                "description": description,
                                "metadata": parsed_content.metadata,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"LLM enhancement failed, using original chunks: {str(e)}")
                        enhanced_chunks = chunk_result.chunks
                        warnings.append(IngestionWarning(
                            type="warning",
                            message=f"LLM enhancement failed: {str(e)}"
                        ))
                else:
                    enhanced_chunks = chunk_result.chunks
                
                enhancement_time = time.time() - enhancement_start
                
                # Step 4: Create comprehensive metadata
                chunk_metadata_list = []
                for i, chunk in enumerate(enhanced_chunks):
                    chunk_meta = ChunkMetadata(
                        doc_id=doc_id,
                        chunk_index=i,
                        chunk_text=chunk["text"],
                        tokens=chunk.get("tokens", 0),
                        chunk_type=chunk.get("chunk_type", "text"),
                        source_page=chunk.get("source_page"),
                        source_section=chunk.get("source_section"),
                        enhanced_context=chunk.get("enhanced_context"),
                        confidence_score=chunk.get("confidence_score"),
                        hallucination_score=chunk.get("hallucination_score"),
                    )
                    chunk_metadata_list.append(chunk_meta)
                
                # Create document metadata
                doc_metadata = DocumentMetadata(
                    doc_id=doc_id,
                    file_name=file_name,
                    file_type=file_type,
                    file_size=len(content_bytes),
                    source_type="api",  # This could be passed as parameter
                    source_agent=source_agent,
                    session_id=session_id,
                    description=description,
                    chunking_strategy=chunk_result.strategy_used,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    total_chunks=len(enhanced_chunks),
                    total_tokens=chunk_result.total_tokens,
                    processing_time=0.0,  # Will be updated below
                    embedding_time=0.0,  # Will be updated below
                    storage_time=0.0,  # Will be updated below
                    graph_time=0.0,  # Will be updated below
                    llm_context_enabled=enable_llm_context,
                    page_count=parsed_content.metadata.get("page_count"),
                    slide_count=parsed_content.metadata.get("slide_count"),
                    table_count=parsed_content.metadata.get("table_count"),
                    image_count=parsed_content.metadata.get("image_count"),
                    parsing_warnings=[w.message for w in warnings],
                    custom_metadata=custom_metadata or {},
                )
                
                # Step 5: Store in memory system (if memory manager provided)
                storage_start = time.time()
                entities_created = []
                relationships_created = []
                
                if memory_manager:
                    try:
                        # Store each chunk as a memory
                        for chunk_meta in chunk_metadata_list:
                            memory_request = MemoryStoreRequest(
                                content=chunk_meta.chunk_text,
                                agent_id=source_agent,
                                session_id=session_id,
                                metadata={
                                    "doc_id": doc_id,
                                    "chunk_id": chunk_meta.chunk_id,
                                    "chunk_index": chunk_meta.chunk_index,
                                    "chunk_type": chunk_meta.chunk_type,
                                    "file_name": file_name,
                                    "file_type": file_type,
                                    "source_page": chunk_meta.source_page,
                                    "enhanced_context": chunk_meta.enhanced_context,
                                    "confidence_score": chunk_meta.confidence_score,
                                    "hallucination_score": chunk_meta.hallucination_score,
                                    **doc_metadata.custom_metadata,
                                },
                                extract_entities=True,
                                create_relationships=True,
                            )
                            
                            store_result = await memory_manager.store_memory(memory_request)
                            entities_created.extend(store_result.entities_created)
                            relationships_created.extend(store_result.relationships_created)
                        
                        logger.info(
                            "Document stored in memory system",
                            doc_id=doc_id,
                            chunks_stored=len(chunk_metadata_list),
                            entities_created=len(entities_created),
                            relationships_created=len(relationships_created),
                        )
                        
                    except Exception as e:
                        logger.error(f"Failed to store document in memory system: {str(e)}")
                        warnings.append(IngestionWarning(
                            type="error",
                            message=f"Memory storage failed: {str(e)}"
                        ))
                
                storage_time = time.time() - storage_start
                
                # Step 6: Calculate quality metrics
                quality_start = time.time()
                
                confidence_scores = [
                    chunk.get("confidence_score", 0.0)
                    for chunk in enhanced_chunks
                    if chunk.get("confidence_score") is not None
                ]
                
                hallucination_scores = [
                    chunk.get("hallucination_score", 0.0)
                    for chunk in enhanced_chunks
                    if chunk.get("hallucination_score") is not None
                ]
                
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else None
                avg_hallucination = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else None
                
                # Detect safety flags
                safety_flags = []
                if avg_confidence and avg_confidence < 0.5:
                    safety_flags.append("low_confidence")
                if avg_hallucination and avg_hallucination > 0.5:
                    safety_flags.append("high_hallucination_risk")
                
                quality_time = time.time() - quality_start
                
                # Update final metadata with timing
                total_processing_time = time.time() - start_time
                doc_metadata.processing_time = total_processing_time
                doc_metadata.embedding_time = 0.0  # This would be set by memory manager
                doc_metadata.storage_time = storage_time
                doc_metadata.graph_time = 0.0  # This would be set by memory manager
                doc_metadata.avg_confidence_score = avg_confidence
                doc_metadata.avg_hallucination_score = avg_hallucination
                
                # Update processing stats
                self._update_processing_stats(total_processing_time, len(enhanced_chunks))
                
                # Determine final status
                if len(warnings) == 0:
                    status = "success"
                elif any(w.type == "error" for w in warnings):
                    status = "partial_success"
                else:
                    status = "success"
                
                logger.info(
                    "Document processing completed",
                    doc_id=doc_id,
                    status=status,
                    total_time=total_processing_time,
                    chunks_processed=len(enhanced_chunks),
                    warnings_count=len(warnings),
                )
                
                span.set_attributes({
                    "status": status,
                    "chunks_processed": len(enhanced_chunks),
                    "total_time": total_processing_time,
                    "warnings_count": len(warnings),
                })
                
                return IngestResponse(
                    status=status,
                    doc_id=doc_id,
                    summary=f"Successfully processed {file_name} into {len(enhanced_chunks)} chunks",
                    chunks_ingested=len(enhanced_chunks),
                    total_chunks_attempted=len(enhanced_chunks),
                    processing_time=total_processing_time,
                    document_metadata=doc_metadata,
                    chunks_metadata=chunk_metadata_list,
                    avg_confidence_score=avg_confidence,
                    avg_hallucination_score=avg_hallucination,
                    safety_flags=safety_flags,
                    warnings=warnings,
                    entities_created=entities_created,
                    relationships_created=relationships_created,
                    embedding_time=0.0,  # Set by memory manager
                    storage_time=storage_time,
                    graph_time=0.0,  # Set by memory manager
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                
                logger.error(
                    "Document processing failed",
                    doc_id=doc_id,
                    file_name=file_name,
                    error=str(e),
                    processing_time=processing_time,
                )
                
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                
                # Create error metadata
                error_metadata = DocumentMetadata(
                    doc_id=doc_id,
                    file_name=file_name,
                    file_type=file_type,
                    file_size=len(content_bytes),
                    source_type="api",
                    source_agent=source_agent,
                    session_id=session_id,
                    description=description,
                    chunking_strategy=chunking_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    total_chunks=0,
                    total_tokens=0,
                    processing_time=processing_time,
                    embedding_time=0.0,
                    storage_time=0.0,
                    graph_time=0.0,
                    llm_context_enabled=enable_llm_context,
                    parsing_warnings=[f"Processing failed: {str(e)}"],
                    custom_metadata=custom_metadata or {},
                )
                
                return IngestResponse(
                    status="failed",
                    doc_id=doc_id,
                    summary=f"Failed to process {file_name}: {str(e)}",
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
                    entities_created=[],
                    relationships_created=[],
                    embedding_time=0.0,
                    storage_time=0.0,
                    graph_time=0.0,
                )
    
    def _update_processing_stats(self, processing_time: float, chunk_count: int) -> None:
        """Update internal processing statistics."""
        self._processing_stats["total_documents"] += 1
        self._processing_stats["total_chunks"] += chunk_count
        self._processing_stats["total_processing_time"] += processing_time
        
        # Update average
        total_docs = self._processing_stats["total_documents"]
        self._processing_stats["avg_processing_time"] = (
            self._processing_stats["total_processing_time"] / total_docs
        )
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._processing_stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the document processor."""
        try:
            # Check LLM enhancer
            llm_status = "healthy"
            if self.llm_enhancer:
                llm_health = await self.llm_enhancer.health_check()
                llm_status = llm_health.get("status", "unknown")
            
            # Check file loaders
            test_loaders = ["pdf", "docx", "txt", "html", "json", "csv"]
            loader_status = "healthy"
            
            for file_type in test_loaders:
                try:
                    get_file_loader(file_type)
                except Exception:
                    loader_status = "degraded"
                    break
            
            overall_status = "healthy"
            if llm_status != "healthy" or loader_status != "healthy":
                overall_status = "degraded"
            
            return {
                "status": overall_status,
                "components": {
                    "llm_enhancer": llm_status,
                    "file_loaders": loader_status,
                },
                "stats": self._processing_stats,
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    async def close(self) -> None:
        """Close the document processor and cleanup resources."""
        try:
            if self.llm_enhancer:
                await self.llm_enhancer.close()
            
            logger.info("Document processor closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing document processor: {str(e)}")