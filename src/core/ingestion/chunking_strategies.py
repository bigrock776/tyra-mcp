"""
Dynamic chunking strategies for different document types.

Intelligent chunking algorithms that adapt to document structure
and content type for optimal memory segmentation and retrieval.
"""

import abc
import re
import math
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChunkResult:
    """Result from a chunking operation."""
    
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        strategy_used: str,
        total_tokens: int,
        avg_chunk_size: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunks = chunks
        self.strategy_used = strategy_used
        self.total_tokens = total_tokens
        self.avg_chunk_size = avg_chunk_size
        self.metadata = metadata or {}


class BaseChunkingStrategy(abc.ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @property
    @abc.abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        pass
    
    @abc.abstractmethod
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk the text according to this strategy."""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (simple word-based approximation)."""
        return max(1, len(text.split()) * 1.3)  # Rough approximation
    
    def create_chunk(
        self,
        text: str,
        chunk_index: int,
        chunk_type: str = "text",
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a standardized chunk dictionary."""
        chunk = {
            "text": text.strip(),
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "tokens": self.estimate_tokens(text),
            "character_count": len(text),
        }
        
        if source_info:
            chunk.update(source_info)
        
        chunk.update(kwargs)
        return chunk


class ParagraphChunkingStrategy(BaseChunkingStrategy):
    """Chunk text by paragraphs with size constraints."""
    
    @property
    def strategy_name(self) -> str:
        return "paragraph"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk by paragraphs, combining small ones and splitting large ones."""
        
        # If we already have paragraph chunks from file loader, use them
        if existing_chunks and all(chunk.get("chunk_type") == "paragraph" for chunk in existing_chunks):
            paragraphs = [chunk["text"] for chunk in existing_chunks]
        else:
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        if not paragraphs:
            paragraphs = [text]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_tokens = self.estimate_tokens(paragraph)
            current_tokens = self.estimate_tokens(current_chunk)
            
            # If paragraph alone exceeds chunk size, split it
            if paragraph_tokens > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(self.create_chunk(
                        current_chunk,
                        chunk_index,
                        "paragraph_group",
                        {"paragraph_count": current_chunk.count("\n\n") + 1}
                    ))
                    chunk_index += 1
                    current_chunk = ""
                
                # Split large paragraph
                sub_chunks = await self._split_large_paragraph(paragraph, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                
            # If adding this paragraph would exceed chunk size
            elif current_tokens + paragraph_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self.create_chunk(
                    current_chunk,
                    chunk_index,
                    "paragraph_group",
                    {"paragraph_count": current_chunk.count("\n\n") + 1}
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(self.create_chunk(
                current_chunk,
                chunk_index,
                "paragraph_group",
                {"paragraph_count": current_chunk.count("\n\n") + 1}
            ))
        
        total_tokens = sum(chunk["tokens"] for chunk in chunks)
        avg_chunk_size = total_tokens / len(chunks) if chunks else 0
        
        return ChunkResult(
            chunks=chunks,
            strategy_used=self.strategy_name,
            total_tokens=total_tokens,
            avg_chunk_size=avg_chunk_size,
            metadata={
                "original_paragraphs": len(paragraphs),
                "final_chunks": len(chunks),
            }
        )
    
    async def _split_large_paragraph(self, paragraph: str, start_index: int) -> List[Dict[str, Any]]:
        """Split a large paragraph into smaller chunks."""
        sentences = re.split(r'[.!?]+\s+', paragraph)
        chunks = []
        current_chunk = ""
        chunk_index = start_index
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self.estimate_tokens(sentence)
            current_tokens = self.estimate_tokens(current_chunk)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(self.create_chunk(
                    current_chunk,
                    chunk_index,
                    "paragraph_split",
                    {"from_large_paragraph": True}
                ))
                chunk_index += 1
                
                # Add overlap
                if self.chunk_overlap > 0:
                    overlap = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap + " " + sentence if overlap else sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(self.create_chunk(
                current_chunk,
                chunk_index,
                "paragraph_split",
                {"from_large_paragraph": True}
            ))
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        words = text.split()
        if len(words) <= overlap_size:
            return text
        return " ".join(words[-overlap_size:])


class SemanticChunkingStrategy(BaseChunkingStrategy):
    """Chunk text based on semantic boundaries and topic shifts."""
    
    @property
    def strategy_name(self) -> str:
        return "semantic"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk based on semantic boundaries."""
        
        # For now, use a combination of paragraph and sentence boundaries
        # In a full implementation, this would use NLP models to detect topic shifts
        
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return ChunkResult(
                chunks=[self.create_chunk(text, 0, "semantic")],
                strategy_used=self.strategy_name,
                total_tokens=self.estimate_tokens(text),
                avg_chunk_size=self.estimate_tokens(text),
            )
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.estimate_tokens(sentence)
            current_tokens = self.estimate_tokens(current_chunk)
            
            # Check for semantic break indicators
            is_semantic_break = self._is_semantic_break(sentence, i, sentences)
            
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            
            updated_tokens = self.estimate_tokens(current_chunk)
            
            # Create chunk if we hit size limit or semantic break
            if (updated_tokens >= self.chunk_size) or (is_semantic_break and updated_tokens > self.chunk_size * 0.5):
                chunks.append(self.create_chunk(
                    current_chunk,
                    chunk_index,
                    "semantic",
                    {
                        "semantic_break": is_semantic_break,
                        "sentence_count": current_chunk.count(".") + current_chunk.count("!") + current_chunk.count("?")
                    }
                ))
                chunk_index += 1
                
                # Handle overlap
                if self.chunk_overlap > 0 and i < len(sentences) - 1:
                    overlap_sentences = self._get_overlap_sentences(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_sentences
                else:
                    current_chunk = ""
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(self.create_chunk(
                current_chunk,
                chunk_index,
                "semantic",
                {"sentence_count": current_chunk.count(".") + current_chunk.count("!") + current_chunk.count("?")}
            ))
        
        total_tokens = sum(chunk["tokens"] for chunk in chunks)
        avg_chunk_size = total_tokens / len(chunks) if chunks else 0
        
        return ChunkResult(
            chunks=chunks,
            strategy_used=self.strategy_name,
            total_tokens=total_tokens,
            avg_chunk_size=avg_chunk_size,
            metadata={
                "original_sentences": len(sentences),
                "semantic_breaks_detected": sum(1 for chunk in chunks if chunk.get("semantic_break")),
            }
        )
    
    def _is_semantic_break(self, sentence: str, index: int, all_sentences: List[str]) -> bool:
        """Detect potential semantic breaks."""
        # Simple heuristics for semantic breaks
        transition_words = [
            "however", "meanwhile", "furthermore", "moreover", "additionally",
            "consequently", "therefore", "in conclusion", "finally", "next",
            "first", "second", "third", "lastly", "on the other hand"
        ]
        
        sentence_lower = sentence.lower()
        
        # Check for transition words at the beginning
        for word in transition_words:
            if sentence_lower.startswith(word):
                return True
        
        # Check for topic change indicators
        if any(phrase in sentence_lower for phrase in ["new topic", "moving on", "let's discuss", "another"]):
            return True
        
        # Check for paragraph breaks in original text (if available)
        # This is a simplified check
        return False
    
    def _get_overlap_sentences(self, text: str, overlap_words: int) -> str:
        """Get overlap text by sentences."""
        sentences = re.split(r'[.!?]+\s+', text)
        if not sentences:
            return ""
        
        # Get the last sentence for overlap
        last_sentence = sentences[-1].strip()
        words = last_sentence.split()
        
        if len(words) <= overlap_words:
            return last_sentence
        
        return " ".join(words[-overlap_words:])


class SlideChunkingStrategy(BaseChunkingStrategy):
    """Chunk PPTX content by slides."""
    
    @property
    def strategy_name(self) -> str:
        return "slide"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk by slides if slide information is available."""
        
        # If we have existing slide chunks, use them
        if existing_chunks and any(chunk.get("chunk_type") == "slide" for chunk in existing_chunks):
            slide_chunks = [chunk for chunk in existing_chunks if chunk.get("chunk_type") == "slide"]
            
            # Combine slides if they're too small
            combined_chunks = []
            current_chunk = ""
            chunk_index = 0
            
            for slide_chunk in slide_chunks:
                slide_text = slide_chunk["text"]
                slide_tokens = self.estimate_tokens(slide_text)
                current_tokens = self.estimate_tokens(current_chunk)
                
                if current_tokens + slide_tokens <= self.chunk_size or not current_chunk:
                    if current_chunk:
                        current_chunk += "\n\n" + slide_text
                    else:
                        current_chunk = slide_text
                else:
                    # Save current chunk
                    combined_chunks.append(self.create_chunk(
                        current_chunk,
                        chunk_index,
                        "slide_group",
                        {"slide_count": current_chunk.count("\n\n") + 1}
                    ))
                    chunk_index += 1
                    current_chunk = slide_text
            
            # Add final chunk
            if current_chunk.strip():
                combined_chunks.append(self.create_chunk(
                    current_chunk,
                    chunk_index,
                    "slide_group",
                    {"slide_count": current_chunk.count("\n\n") + 1}
                ))
            
            total_tokens = sum(chunk["tokens"] for chunk in combined_chunks)
            avg_chunk_size = total_tokens / len(combined_chunks) if combined_chunks else 0
            
            return ChunkResult(
                chunks=combined_chunks,
                strategy_used=self.strategy_name,
                total_tokens=total_tokens,
                avg_chunk_size=avg_chunk_size,
                metadata={
                    "original_slides": len(slide_chunks),
                    "combined_chunks": len(combined_chunks),
                }
            )
        
        # Fallback to paragraph chunking if no slide info
        paragraph_strategy = ParagraphChunkingStrategy(self.chunk_size, self.chunk_overlap)
        result = await paragraph_strategy.chunk(text, metadata, existing_chunks)
        result.strategy_used = f"{self.strategy_name}_fallback"
        return result


class LineChunkingStrategy(BaseChunkingStrategy):
    """Chunk text line by line (for CSV, logs, etc.)."""
    
    @property
    def strategy_name(self) -> str:
        return "line"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk by lines, combining to meet size constraints."""
        
        lines = text.split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            return ChunkResult(
                chunks=[self.create_chunk(text, 0, "line")],
                strategy_used=self.strategy_name,
                total_tokens=self.estimate_tokens(text),
                avg_chunk_size=self.estimate_tokens(text),
            )
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        line_count = 0
        
        for line in lines:
            line_tokens = self.estimate_tokens(line)
            current_tokens = self.estimate_tokens(current_chunk)
            
            if current_tokens + line_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self.create_chunk(
                    current_chunk,
                    chunk_index,
                    "line_group",
                    {"line_count": line_count}
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_lines = current_chunk.split("\n")[-self.chunk_overlap:]
                    current_chunk = "\n".join(overlap_lines) + "\n" + line
                    line_count = len(overlap_lines) + 1
                else:
                    current_chunk = line
                    line_count = 1
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n" + line
                    line_count += 1
                else:
                    current_chunk = line
                    line_count = 1
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(self.create_chunk(
                current_chunk,
                chunk_index,
                "line_group",
                {"line_count": line_count}
            ))
        
        total_tokens = sum(chunk["tokens"] for chunk in chunks)
        avg_chunk_size = total_tokens / len(chunks) if chunks else 0
        
        return ChunkResult(
            chunks=chunks,
            strategy_used=self.strategy_name,
            total_tokens=total_tokens,
            avg_chunk_size=avg_chunk_size,
            metadata={
                "original_lines": len(lines),
                "final_chunks": len(chunks),
            }
        )


class TokenChunkingStrategy(BaseChunkingStrategy):
    """Chunk text by token count with word boundaries."""
    
    @property
    def strategy_name(self) -> str:
        return "token"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk by token count, respecting word boundaries."""
        
        words = text.split()
        if not words:
            return ChunkResult(
                chunks=[self.create_chunk(text, 0, "token")],
                strategy_used=self.strategy_name,
                total_tokens=self.estimate_tokens(text),
                avg_chunk_size=self.estimate_tokens(text),
            )
        
        chunks = []
        current_words = []
        chunk_index = 0
        
        for i, word in enumerate(words):
            current_words.append(word)
            current_text = " ".join(current_words)
            current_tokens = self.estimate_tokens(current_text)
            
            if current_tokens >= self.chunk_size:
                # Save current chunk
                chunks.append(self.create_chunk(
                    current_text,
                    chunk_index,
                    "token",
                    {"word_count": len(current_words)}
                ))
                chunk_index += 1
                
                # Handle overlap
                if self.chunk_overlap > 0 and i < len(words) - 1:
                    overlap_words = current_words[-self.chunk_overlap:]
                    current_words = overlap_words
                else:
                    current_words = []
        
        # Add final chunk
        if current_words:
            final_text = " ".join(current_words)
            chunks.append(self.create_chunk(
                final_text,
                chunk_index,
                "token",
                {"word_count": len(current_words)}
            ))
        
        total_tokens = sum(chunk["tokens"] for chunk in chunks)
        avg_chunk_size = total_tokens / len(chunks) if chunks else 0
        
        return ChunkResult(
            chunks=chunks,
            strategy_used=self.strategy_name,
            total_tokens=total_tokens,
            avg_chunk_size=avg_chunk_size,
            metadata={
                "original_words": len(words),
                "final_chunks": len(chunks),
            }
        )


class PageChunkingStrategy(BaseChunkingStrategy):
    """Chunk PDF content by pages."""
    
    @property
    def strategy_name(self) -> str:
        return "page"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk by pages if page information is available."""
        
        # If we have existing page chunks, use them
        if existing_chunks and any(chunk.get("chunk_type") == "page" for chunk in existing_chunks):
            page_chunks = [chunk for chunk in existing_chunks if chunk.get("chunk_type") == "page"]
            
            # Combine pages if they're too small
            combined_chunks = []
            current_chunk = ""
            chunk_index = 0
            page_numbers = []
            
            for page_chunk in page_chunks:
                page_text = page_chunk["text"]
                page_tokens = self.estimate_tokens(page_text)
                current_tokens = self.estimate_tokens(current_chunk)
                page_num = page_chunk.get("page_number", len(page_numbers) + 1)
                
                if current_tokens + page_tokens <= self.chunk_size or not current_chunk:
                    if current_chunk:
                        current_chunk += "\n\n" + page_text
                        page_numbers.append(page_num)
                    else:
                        current_chunk = page_text
                        page_numbers = [page_num]
                else:
                    # Save current chunk
                    combined_chunks.append(self.create_chunk(
                        current_chunk,
                        chunk_index,
                        "page_group",
                        {
                            "page_count": len(page_numbers),
                            "page_numbers": page_numbers,
                            "start_page": page_numbers[0],
                            "end_page": page_numbers[-1]
                        }
                    ))
                    chunk_index += 1
                    current_chunk = page_text
                    page_numbers = [page_num]
            
            # Add final chunk
            if current_chunk.strip():
                combined_chunks.append(self.create_chunk(
                    current_chunk,
                    chunk_index,
                    "page_group",
                    {
                        "page_count": len(page_numbers),
                        "page_numbers": page_numbers,
                        "start_page": page_numbers[0] if page_numbers else 1,
                        "end_page": page_numbers[-1] if page_numbers else 1
                    }
                ))
            
            total_tokens = sum(chunk["tokens"] for chunk in combined_chunks)
            avg_chunk_size = total_tokens / len(combined_chunks) if combined_chunks else 0
            
            return ChunkResult(
                chunks=combined_chunks,
                strategy_used=self.strategy_name,
                total_tokens=total_tokens,
                avg_chunk_size=avg_chunk_size,
                metadata={
                    "original_pages": len(page_chunks),
                    "combined_chunks": len(combined_chunks),
                }
            )
        
        # Fallback to paragraph chunking if no page info
        paragraph_strategy = ParagraphChunkingStrategy(self.chunk_size, self.chunk_overlap)
        result = await paragraph_strategy.chunk(text, metadata, existing_chunks)
        result.strategy_used = f"{self.strategy_name}_fallback"
        return result


class SectionChunkingStrategy(BaseChunkingStrategy):
    """Chunk text by sections (headers/headings)."""
    
    @property
    def strategy_name(self) -> str:
        return "section"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk by sections based on headers."""
        
        # Detect sections using headers (markdown-style or numbered)
        sections = self._detect_sections(text)
        
        if len(sections) <= 1:
            # If no clear sections, fall back to paragraph chunking
            paragraph_strategy = ParagraphChunkingStrategy(self.chunk_size, self.chunk_overlap)
            result = await paragraph_strategy.chunk(text, metadata, existing_chunks)
            result.strategy_used = f"{self.strategy_name}_fallback"
            return result
        
        chunks = []
        chunk_index = 0
        
        for i, (section_title, section_content) in enumerate(sections):
            section_tokens = self.estimate_tokens(section_content)
            
            if section_tokens <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(self.create_chunk(
                    section_content,
                    chunk_index,
                    "section",
                    {
                        "section_title": section_title,
                        "section_number": i + 1,
                        "is_complete_section": True
                    }
                ))
                chunk_index += 1
            else:
                # Section too large, split it
                section_chunks = await self._split_large_section(
                    section_content, section_title, chunk_index
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
        
        total_tokens = sum(chunk["tokens"] for chunk in chunks)
        avg_chunk_size = total_tokens / len(chunks) if chunks else 0
        
        return ChunkResult(
            chunks=chunks,
            strategy_used=self.strategy_name,
            total_tokens=total_tokens,
            avg_chunk_size=avg_chunk_size,
            metadata={
                "original_sections": len(sections),
                "final_chunks": len(chunks),
            }
        )
    
    def _detect_sections(self, text: str) -> List[Tuple[str, str]]:
        """Detect sections in text based on headers."""
        sections = []
        
        # Split by markdown headers or numbered sections
        header_pattern = r'^(#{1,6}\s+.+|[0-9]+\.?\s+[A-Z][^.\n]*\.?)\s*$'
        lines = text.split('\n')
        
        current_section_title = "Introduction"
        current_section_content = ""
        
        for line in lines:
            if re.match(header_pattern, line.strip(), re.MULTILINE):
                # Save previous section
                if current_section_content.strip():
                    sections.append((current_section_title, current_section_content.strip()))
                
                # Start new section
                current_section_title = line.strip()
                current_section_content = ""
            else:
                current_section_content += line + "\n"
        
        # Add final section
        if current_section_content.strip():
            sections.append((current_section_title, current_section_content.strip()))
        
        return sections
    
    async def _split_large_section(
        self, section_content: str, section_title: str, start_index: int
    ) -> List[Dict[str, Any]]:
        """Split a large section into smaller chunks."""
        # Use paragraph strategy to split the section
        paragraph_strategy = ParagraphChunkingStrategy(self.chunk_size, self.chunk_overlap)
        result = await paragraph_strategy.chunk(section_content, {})
        
        # Update chunks with section information
        section_chunks = []
        for i, chunk in enumerate(result.chunks):
            chunk.update({
                "chunk_index": start_index + i,
                "chunk_type": "section_part",
                "section_title": section_title,
                "section_part": i + 1,
                "total_section_parts": len(result.chunks)
            })
            section_chunks.append(chunk)
        
        return section_chunks


class ObjectChunkingStrategy(BaseChunkingStrategy):
    """Chunk JSON by objects."""
    
    @property
    def strategy_name(self) -> str:
        return "object"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk JSON by top-level objects."""
        
        # If we have existing object chunks, use them
        if existing_chunks and any(chunk.get("chunk_type") == "json_object" for chunk in existing_chunks):
            object_chunks = [chunk for chunk in existing_chunks if chunk.get("chunk_type") == "json_object"]
            
            # Combine small objects
            combined_chunks = []
            current_objects = []
            chunk_index = 0
            
            for obj_chunk in object_chunks:
                obj_text = obj_chunk["text"]
                obj_tokens = self.estimate_tokens(obj_text)
                current_tokens = sum(self.estimate_tokens(obj["text"]) for obj in current_objects)
                
                if current_tokens + obj_tokens <= self.chunk_size or not current_objects:
                    current_objects.append(obj_chunk)
                else:
                    # Save current chunk
                    combined_text = "\n".join(obj["text"] for obj in current_objects)
                    combined_chunks.append(self.create_chunk(
                        combined_text,
                        chunk_index,
                        "object_group",
                        {
                            "object_count": len(current_objects),
                            "object_keys": [obj.get("object_key", f"object_{i}") for i, obj in enumerate(current_objects)]
                        }
                    ))
                    chunk_index += 1
                    current_objects = [obj_chunk]
            
            # Add final chunk
            if current_objects:
                combined_text = "\n".join(obj["text"] for obj in current_objects)
                combined_chunks.append(self.create_chunk(
                    combined_text,
                    chunk_index,
                    "object_group",
                    {
                        "object_count": len(current_objects),
                        "object_keys": [obj.get("object_key", f"object_{i}") for i, obj in enumerate(current_objects)]
                    }
                ))
            
            total_tokens = sum(chunk["tokens"] for chunk in combined_chunks)
            avg_chunk_size = total_tokens / len(combined_chunks) if combined_chunks else 0
            
            return ChunkResult(
                chunks=combined_chunks,
                strategy_used=self.strategy_name,
                total_tokens=total_tokens,
                avg_chunk_size=avg_chunk_size,
                metadata={
                    "original_objects": len(object_chunks),
                    "combined_chunks": len(combined_chunks),
                }
            )
        
        # Fallback to semantic chunking
        semantic_strategy = SemanticChunkingStrategy(self.chunk_size, self.chunk_overlap)
        result = await semantic_strategy.chunk(text, metadata, existing_chunks)
        result.strategy_used = f"{self.strategy_name}_fallback"
        return result


class ArrayChunkingStrategy(BaseChunkingStrategy):
    """Chunk JSON by array elements."""
    
    @property
    def strategy_name(self) -> str:
        return "array"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk JSON by array elements."""
        
        # If we have existing array chunks, use them
        if existing_chunks and any(chunk.get("chunk_type") == "array_element" for chunk in existing_chunks):
            array_chunks = [chunk for chunk in existing_chunks if chunk.get("chunk_type") == "array_element"]
            
            # Combine array elements to fit chunk size
            combined_chunks = []
            current_elements = []
            chunk_index = 0
            
            for element_chunk in array_chunks:
                element_text = element_chunk["text"]
                element_tokens = self.estimate_tokens(element_text)
                current_tokens = sum(self.estimate_tokens(elem["text"]) for elem in current_elements)
                
                if current_tokens + element_tokens <= self.chunk_size or not current_elements:
                    current_elements.append(element_chunk)
                else:
                    # Save current chunk
                    combined_text = "\n".join(elem["text"] for elem in current_elements)
                    combined_chunks.append(self.create_chunk(
                        combined_text,
                        chunk_index,
                        "array_group",
                        {
                            "element_count": len(current_elements),
                            "start_index": current_elements[0].get("array_index", 0),
                            "end_index": current_elements[-1].get("array_index", len(current_elements) - 1)
                        }
                    ))
                    chunk_index += 1
                    current_elements = [element_chunk]
            
            # Add final chunk
            if current_elements:
                combined_text = "\n".join(elem["text"] for elem in current_elements)
                combined_chunks.append(self.create_chunk(
                    combined_text,
                    chunk_index,
                    "array_group",
                    {
                        "element_count": len(current_elements),
                        "start_index": current_elements[0].get("array_index", 0),
                        "end_index": current_elements[-1].get("array_index", len(current_elements) - 1)
                    }
                ))
            
            total_tokens = sum(chunk["tokens"] for chunk in combined_chunks)
            avg_chunk_size = total_tokens / len(combined_chunks) if combined_chunks else 0
            
            return ChunkResult(
                chunks=combined_chunks,
                strategy_used=self.strategy_name,
                total_tokens=total_tokens,
                avg_chunk_size=avg_chunk_size,
                metadata={
                    "original_elements": len(array_chunks),
                    "combined_chunks": len(combined_chunks),
                }
            )
        
        # Fallback to line chunking for array-like content
        line_strategy = LineChunkingStrategy(self.chunk_size, self.chunk_overlap)
        result = await line_strategy.chunk(text, metadata, existing_chunks)
        result.strategy_used = f"{self.strategy_name}_fallback"
        return result


class RowChunkingStrategy(BaseChunkingStrategy):
    """Chunk CSV by rows (alias for line chunking with CSV-specific handling)."""
    
    @property
    def strategy_name(self) -> str:
        return "row"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk CSV by rows."""
        
        # If we have existing row chunks, use them
        if existing_chunks and any(chunk.get("chunk_type") == "csv_row" for chunk in existing_chunks):
            row_chunks = [chunk for chunk in existing_chunks if chunk.get("chunk_type") == "csv_row"]
            
            # Group rows with header context
            header_row = None
            data_rows = []
            
            for row_chunk in row_chunks:
                if row_chunk.get("is_header", False):
                    header_row = row_chunk
                else:
                    data_rows.append(row_chunk)
            
            # Combine rows into chunks
            combined_chunks = []
            current_rows = []
            chunk_index = 0
            
            for row_chunk in data_rows:
                row_tokens = self.estimate_tokens(row_chunk["text"])
                current_tokens = sum(self.estimate_tokens(row["text"]) for row in current_rows)
                header_tokens = self.estimate_tokens(header_row["text"]) if header_row else 0
                
                if current_tokens + row_tokens + header_tokens <= self.chunk_size or not current_rows:
                    current_rows.append(row_chunk)
                else:
                    # Save current chunk with header
                    chunk_text = ""
                    if header_row:
                        chunk_text = header_row["text"] + "\n"
                    chunk_text += "\n".join(row["text"] for row in current_rows)
                    
                    combined_chunks.append(self.create_chunk(
                        chunk_text,
                        chunk_index,
                        "row_group",
                        {
                            "row_count": len(current_rows),
                            "has_header": header_row is not None,
                            "start_row": current_rows[0].get("row_number", 1),
                            "end_row": current_rows[-1].get("row_number", len(current_rows))
                        }
                    ))
                    chunk_index += 1
                    current_rows = [row_chunk]
            
            # Add final chunk
            if current_rows:
                chunk_text = ""
                if header_row:
                    chunk_text = header_row["text"] + "\n"
                chunk_text += "\n".join(row["text"] for row in current_rows)
                
                combined_chunks.append(self.create_chunk(
                    chunk_text,
                    chunk_index,
                    "row_group",
                    {
                        "row_count": len(current_rows),
                        "has_header": header_row is not None,
                        "start_row": current_rows[0].get("row_number", 1),
                        "end_row": current_rows[-1].get("row_number", len(current_rows))
                    }
                ))
            
            total_tokens = sum(chunk["tokens"] for chunk in combined_chunks)
            avg_chunk_size = total_tokens / len(combined_chunks) if combined_chunks else 0
            
            return ChunkResult(
                chunks=combined_chunks,
                strategy_used=self.strategy_name,
                total_tokens=total_tokens,
                avg_chunk_size=avg_chunk_size,
                metadata={
                    "original_rows": len(data_rows),
                    "combined_chunks": len(combined_chunks),
                    "has_header": header_row is not None,
                }
            )
        
        # Fallback to line chunking
        line_strategy = LineChunkingStrategy(self.chunk_size, self.chunk_overlap)
        result = await line_strategy.chunk(text, metadata, existing_chunks)
        result.strategy_used = self.strategy_name  # Keep row name
        
        # Update chunk types to be row-specific
        for chunk in result.chunks:
            if chunk["chunk_type"] == "line_group":
                chunk["chunk_type"] = "row_group"
                chunk["row_count"] = chunk.pop("line_count", 0)
        
        return result


class ChapterChunkingStrategy(BaseChunkingStrategy):
    """Chunk EPUB content by chapters."""
    
    @property
    def strategy_name(self) -> str:
        return "chapter"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Chunk by chapters if chapter information is available."""
        
        # If we have existing chapter chunks, use them
        if existing_chunks and any(chunk.get("chunk_type") == "chapter" for chunk in existing_chunks):
            chapter_chunks = [chunk for chunk in existing_chunks if chunk.get("chunk_type") == "chapter"]
            
            # Combine short chapters or split long ones
            combined_chunks = []
            current_chapter_content = ""
            current_chapter_titles = []
            chunk_index = 0
            
            for chapter_chunk in chapter_chunks:
                chapter_text = chapter_chunk["text"]
                chapter_tokens = self.estimate_tokens(chapter_text)
                current_tokens = self.estimate_tokens(current_chapter_content)
                chapter_title = chapter_chunk.get("chapter_title", f"Chapter {len(current_chapter_titles) + 1}")
                
                if chapter_tokens > self.chunk_size:
                    # Save current chunk if not empty
                    if current_chapter_content.strip():
                        combined_chunks.append(self.create_chunk(
                            current_chapter_content,
                            chunk_index,
                            "chapter_group",
                            {
                                "chapter_count": len(current_chapter_titles),
                                "chapter_titles": current_chapter_titles
                            }
                        ))
                        chunk_index += 1
                        current_chapter_content = ""
                        current_chapter_titles = []
                    
                    # Split large chapter
                    chapter_parts = await self._split_large_chapter(chapter_text, chapter_title, chunk_index)
                    combined_chunks.extend(chapter_parts)
                    chunk_index += len(chapter_parts)
                    
                elif current_tokens + chapter_tokens <= self.chunk_size or not current_chapter_content:
                    # Add to current chunk
                    if current_chapter_content:
                        current_chapter_content += "\n\n" + chapter_text
                    else:
                        current_chapter_content = chapter_text
                    current_chapter_titles.append(chapter_title)
                else:
                    # Save current chunk
                    combined_chunks.append(self.create_chunk(
                        current_chapter_content,
                        chunk_index,
                        "chapter_group",
                        {
                            "chapter_count": len(current_chapter_titles),
                            "chapter_titles": current_chapter_titles
                        }
                    ))
                    chunk_index += 1
                    current_chapter_content = chapter_text
                    current_chapter_titles = [chapter_title]
            
            # Add final chunk
            if current_chapter_content.strip():
                combined_chunks.append(self.create_chunk(
                    current_chapter_content,
                    chunk_index,
                    "chapter_group",
                    {
                        "chapter_count": len(current_chapter_titles),
                        "chapter_titles": current_chapter_titles
                    }
                ))
            
            total_tokens = sum(chunk["tokens"] for chunk in combined_chunks)
            avg_chunk_size = total_tokens / len(combined_chunks) if combined_chunks else 0
            
            return ChunkResult(
                chunks=combined_chunks,
                strategy_used=self.strategy_name,
                total_tokens=total_tokens,
                avg_chunk_size=avg_chunk_size,
                metadata={
                    "original_chapters": len(chapter_chunks),
                    "combined_chunks": len(combined_chunks),
                }
            )
        
        # Fallback to section-based chunking for chapter-like content
        section_strategy = SectionChunkingStrategy(self.chunk_size, self.chunk_overlap)
        result = await section_strategy.chunk(text, metadata, existing_chunks)
        result.strategy_used = f"{self.strategy_name}_fallback"
        return result
    
    async def _split_large_chapter(
        self, chapter_text: str, chapter_title: str, start_index: int
    ) -> List[Dict[str, Any]]:
        """Split a large chapter into smaller chunks."""
        # Use semantic strategy to split the chapter
        semantic_strategy = SemanticChunkingStrategy(self.chunk_size, self.chunk_overlap)
        result = await semantic_strategy.chunk(chapter_text, {})
        
        # Update chunks with chapter information
        chapter_chunks = []
        for i, chunk in enumerate(result.chunks):
            chunk.update({
                "chunk_index": start_index + i,
                "chunk_type": "chapter_part",
                "chapter_title": chapter_title,
                "chapter_part": i + 1,
                "total_chapter_parts": len(result.chunks)
            })
            chapter_chunks.append(chunk)
        
        return chapter_chunks


class AutoChunkingStrategy(BaseChunkingStrategy):
    """Automatically select the best chunking strategy based on content and file type."""
    
    @property
    def strategy_name(self) -> str:
        return "auto"
    
    async def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ChunkResult:
        """Automatically select and apply the best chunking strategy."""
        
        file_type = metadata.get("format", "").lower()
        text_length = len(text)
        
        # Determine best strategy based on file type and content
        if file_type == "pdf":
            strategy = SemanticChunkingStrategy(self.chunk_size, self.chunk_overlap)
        elif file_type in ["docx", "md", "markdown"]:
            strategy = ParagraphChunkingStrategy(self.chunk_size, self.chunk_overlap)
        elif file_type == "pptx":
            strategy = SlideChunkingStrategy(self.chunk_size, self.chunk_overlap)
        elif file_type == "csv":
            strategy = RowChunkingStrategy(self.chunk_size, self.chunk_overlap)
        elif file_type == "json":
            # For JSON, use object/array chunking if available
            if existing_chunks:
                if any(chunk.get("chunk_type") == "json_object" for chunk in existing_chunks):
                    strategy = ObjectChunkingStrategy(self.chunk_size, self.chunk_overlap)
                elif any(chunk.get("chunk_type") == "array_element" for chunk in existing_chunks):
                    strategy = ArrayChunkingStrategy(self.chunk_size, self.chunk_overlap)
                else:
                    strategy = TokenChunkingStrategy(self.chunk_size, self.chunk_overlap)
            else:
                strategy = SemanticChunkingStrategy(self.chunk_size, self.chunk_overlap)
        elif file_type == "epub":
            strategy = ChapterChunkingStrategy(self.chunk_size, self.chunk_overlap)
        elif text_length < 1000:
            # Small texts - keep as single chunk or minimal splitting
            strategy = TokenChunkingStrategy(self.chunk_size, self.chunk_overlap)
        elif "\n\n" in text and text.count("\n\n") >= 3:
            # Text with clear paragraph structure
            strategy = ParagraphChunkingStrategy(self.chunk_size, self.chunk_overlap)
        else:
            # Default to semantic chunking
            strategy = SemanticChunkingStrategy(self.chunk_size, self.chunk_overlap)
        
        result = await strategy.chunk(text, metadata, existing_chunks)
        result.strategy_used = f"auto_{strategy.strategy_name}"
        result.metadata["auto_selected_strategy"] = strategy.strategy_name
        result.metadata["selection_reason"] = self._get_selection_reason(file_type, text_length, text)
        
        return result
    
    def _get_selection_reason(self, file_type: str, text_length: int, text: str) -> str:
        """Get the reason for strategy selection."""
        if file_type == "pdf":
            return "PDF format - using semantic chunking for better topic coherence"
        elif file_type in ["docx", "md", "markdown"]:
            return "Document format - using paragraph-based chunking"
        elif file_type == "pptx":
            return "Presentation format - using slide-based chunking"
        elif file_type == "csv":
            return "CSV format - using line-based chunking"
        elif file_type == "json":
            return "JSON format - using structured chunking"
        elif text_length < 1000:
            return "Short text - using token-based chunking"
        elif "\n\n" in text and text.count("\n\n") >= 3:
            return "Clear paragraph structure detected - using paragraph chunking"
        else:
            return "Default semantic chunking for optimal coherence"


# Strategy registry
CHUNKING_STRATEGIES = {
    "paragraph": ParagraphChunkingStrategy,
    "semantic": SemanticChunkingStrategy,
    "slide": SlideChunkingStrategy,
    "line": LineChunkingStrategy,
    "token": TokenChunkingStrategy,
    "auto": AutoChunkingStrategy,
    "page": PageChunkingStrategy,
    "section": SectionChunkingStrategy,
    "object": ObjectChunkingStrategy,
    "array": ArrayChunkingStrategy,
    "row": RowChunkingStrategy,
    "chapter": ChapterChunkingStrategy,
}


def get_chunking_strategy(strategy_name: str, chunk_size: int = 512, chunk_overlap: int = 50) -> BaseChunkingStrategy:
    """Get a chunking strategy by name."""
    strategy_class = CHUNKING_STRATEGIES.get(strategy_name.lower())
    if not strategy_class:
        available = ", ".join(CHUNKING_STRATEGIES.keys())
        raise ValueError(f"Unknown chunking strategy: {strategy_name}. Available: {available}")
    
    return strategy_class(chunk_size, chunk_overlap)


async def chunk_content(
    text: str,
    strategy_name: str = "auto",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    metadata: Optional[Dict[str, Any]] = None,
    existing_chunks: Optional[List[Dict[str, Any]]] = None,
) -> ChunkResult:
    """Convenience function to chunk content with specified strategy."""
    strategy = get_chunking_strategy(strategy_name, chunk_size, chunk_overlap)
    return await strategy.chunk(text, metadata or {}, existing_chunks)