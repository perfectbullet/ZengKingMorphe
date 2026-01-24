"""
Semantic document chunking and hierarchical summarization service.

This module provides intelligent document splitting based on semantic meaning
rather than fixed character counts, using embeddings to find natural boundaries.
Also implements hierarchical summarization at chunk, section, and document levels.
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.config import settings
from app.utils.embeddings import get_embedding

logger = get_logger(__name__)

# LLM configuration constants
DEFAULT_LLM_TEMPERATURE = 0.3


@dataclass
class ChunkBoundary:
    """Represents a potential chunk boundary with semantic score."""
    index: int
    position: int
    text: str
    similarity_score: float
    cumulative_size: int


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of text."""
    content: str
    start_pos: int
    end_pos: int
    chunk_index: int
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class SectionSummary:
    """Represents a section-level summary."""
    chunk_indices: List[int]
    summary: str
    embedding: Optional[List[float]] = None


@dataclass
class HierarchicalSummary:
    """Hierarchical summary with multiple levels."""
    chunk_summaries: Dict[int, str]  # chunk_index -> summary
    section_summaries: List[SectionSummary]
    document_summary: Optional[str] = None


class SemanticChunker:
    """
    Splits documents into semantically coherent chunks using embeddings.

    The algorithm:
    1. Split text into sentences
    2. Compute embeddings for sliding windows of sentences
    3. Calculate similarity between consecutive windows
    4. Identify boundaries where similarity drops below threshold
    5. Merge sentences into chunks respecting min/max size constraints
    """

    # Markdown-based formats (.md, .pdf with MinerU) need larger chunks
    # due to short line structure from PDF conversion
    MARKDOWN_FORMATS = {'.md', '.pdf'}

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        window_size: Optional[int] = None,
        file_ext: Optional[str] = None
    ):
        """
        Initialize the semantic chunker.

        Args:
            similarity_threshold: Minimum similarity to merge chunks (default from settings)
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            window_size: Number of sentences to consider for similarity
            file_ext: File extension (.md, .pdf, .txt, etc.) for auto-tuning chunk sizes
        """
        self.similarity_threshold = similarity_threshold or settings.semantic_chunk_similarity_threshold

        # Auto-tune chunk sizes based on file type
        if file_ext and file_ext.lower() in self.MARKDOWN_FORMATS:
            # Markdown/PDF files need larger chunks due to short line structure
            self.min_chunk_size = min_chunk_size or 500
            self.max_chunk_size = max_chunk_size or 2000
        else:
            # Default for plain text and other formats
            self.min_chunk_size = min_chunk_size or settings.semantic_chunk_min_size
            self.max_chunk_size = max_chunk_size or settings.semantic_chunk_max_size

        self.window_size = window_size or settings.semantic_chunk_window_size

        # Chinese and English sentence separators
        self.sentence_pattern = re.compile(
            r'([。！？\.!?]+[\n\t ]*|[\n]+)'
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving delimiters.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split by sentence boundaries
        parts = self.sentence_pattern.split(text)

        sentences = []
        current = ""

        for i, part in enumerate(parts):
            if self.sentence_pattern.match(part):
                # This is a delimiter
                if current:
                    sentences.append(current + part)
                    current = ""
            else:
                # This is content
                if current:
                    current += part
                else:
                    current = part

        # Add remaining content
        if current:
            sentences.append(current)

        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def _create_text_windows(
        self,
        sentences: List[str]
    ) -> List[str]:
        """
        Create sliding windows of sentences for embedding.

        Args:
            sentences: List of sentences

        Returns:
            List of text windows
        """
        windows = []

        for i in range(len(sentences)):
            window_sentences = sentences[i:i + self.window_size]
            window_text = " ".join(window_sentences)
            windows.append(window_text)

        return windows

    async def _compute_similarities(
        self,
        windows: List[str]
    ) -> List[float]:
        """
        Compute cosine similarity between consecutive windows.

        Args:
            windows: List of text windows

        Returns:
            List of similarity scores (length = len(windows) - 1)
        """
        if len(windows) < 2:
            return []

        similarities = []

        # Get embedding instance once
        embedder = get_embedding()

        for i in range(len(windows) - 1):
            try:
                # Use embed_query for single text embedding (synchronous)
                emb1 = embedder.embed_query(windows[i])
                emb2 = embedder.embed_query(windows[i + 1])

                # Cosine similarity
                similarity = self._cosine_similarity(emb1, emb2)
                similarities.append(similarity)

            except Exception as e:
                logger.error(
                    "Failed to compute similarity",
                    window_index=i,
                    error=str(e),
                    exc_info=True
                )
                # Default to high similarity (no split) on error
                similarities.append(1.0)

        return similarities

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def _find_chunk_boundaries(
        self,
        sentences: List[str],
        similarities: List[float]
    ) -> List[ChunkBoundary]:
        """
        Identify optimal chunk boundaries based on similarity drops.

        Args:
            sentences: List of sentences
            similarities: Similarity scores between consecutive windows

        Returns:
            List of potential chunk boundaries
        """
        boundaries = []
        current_size = 0

        for i, similarity in enumerate(similarities):
            current_size += len(sentences[i])

            # Check if this is a good boundary point
            is_low_similarity = similarity < self.similarity_threshold
            exceeds_max = current_size >= self.max_chunk_size
            is_end = i == len(similarities) - 1

            if is_low_similarity or exceeds_max or is_end:
                # Ensure minimum size is met
                if current_size >= self.min_chunk_size or is_end:
                    boundary = ChunkBoundary(
                        index=i,
                        position=sum(len(s) for s in sentences[:i + 1]),
                        text=" ".join(sentences[:i + 1]),
                        similarity_score=similarity,
                        cumulative_size=current_size
                    )
                    boundaries.append(boundary)
                    current_size = 0

        return boundaries

    async def chunk_text(self, text: str) -> List[SemanticChunk]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of semantic chunks
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return [SemanticChunk(
                content=text,
                start_pos=0,
                end_pos=len(text),
                chunk_index=0
            )]

        logger.info(
            "Starting semantic chunking",
            text_length=len(text),
            threshold=self.similarity_threshold
        )

        # Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [SemanticChunk(
                content=text,
                start_pos=0,
                end_pos=len(text),
                chunk_index=0
            )]

        # Create windows and compute similarities
        windows = self._create_text_windows(sentences)
        similarities = await self._compute_similarities(windows)

        # Find boundaries
        boundaries = await self._find_chunk_boundaries(sentences, similarities)

        if not boundaries:
            # Fallback: return entire text as single chunk
            return [SemanticChunk(
                content=text,
                start_pos=0,
                end_pos=len(text),
                chunk_index=0
            )]

        # Create chunks from boundaries
        chunks = []
        start_idx = 0
        chunk_index = 0

        for boundary in boundaries:
            end_idx = boundary.index + 1
            chunk_text = " ".join(sentences[start_idx:end_idx])

            chunks.append(SemanticChunk(
                content=chunk_text,
                start_pos=start_idx,
                end_pos=end_idx,
                chunk_index=chunk_index
            ))

            start_idx = end_idx
            chunk_index += 1

        logger.info(
            "Semantic chunking completed",
            chunks_count=len(chunks),
            avg_chunk_size=sum(len(c.content) for c in chunks) // len(chunks)
        )

        return chunks


class HierarchicalSummarizer:
    """
    Creates summaries at multiple levels: chunk, section, and document.

    Hierarchical structure:
    - Chunk level: Summary of individual semantic chunks
    - Section level: Summary of related chunks (grouped by topic)
    - Document level: Summary of entire document
    """

    def __init__(self):
        """Initialize the hierarchical summarizer."""
        self.chunk_enabled = settings.summary_chunk_level
        self.section_enabled = settings.summary_section_level
        self.document_enabled = settings.summary_document_level
        self.section_threshold = settings.summary_section_threshold
        self.document_threshold = settings.summary_document_threshold

    async def summarize(
        self,
        chunks: List[SemanticChunk],
        doc_id: str
    ) -> HierarchicalSummary:
        """
        Generate hierarchical summaries for document chunks.

        Args:
            chunks: List of semantic chunks
            doc_id: Document ID for logging

        Returns:
            Hierarchical summary object
        """
        logger.info(
            "Starting hierarchical summarization",
            doc_id=doc_id,
            chunks_count=len(chunks),
            chunk_level=self.chunk_enabled,
            section_level=self.section_enabled,
            document_level=self.document_enabled
        )

        summary = HierarchicalSummary(
            chunk_summaries={},
            section_summaries=[]
        )

        # Chunk-level summaries
        if self.chunk_enabled:
            await self._summarize_chunks(chunks, summary, doc_id)

        # Section-level summaries
        if self.section_enabled and len(chunks) >= self.section_threshold:
            await self._summarize_sections(chunks, summary, doc_id)

        # Document-level summary
        if self.document_enabled and len(chunks) >= self.document_threshold:
            await self._summarize_document(chunks, summary, doc_id)

        logger.info(
            "Hierarchical summarization completed",
            doc_id=doc_id,
            chunk_summaries=len(summary.chunk_summaries),
            section_summaries=len(summary.section_summaries),
            has_document_summary=summary.document_summary is not None
        )

        return summary

    def _create_llm(self, max_tokens_multiplier: int = 1):
        """
        Create LLM instance for summarization.

        Args:
            max_tokens_multiplier: Multiplier for max_tokens setting

        Returns:
            Configured LLM instance
        """
        if settings.use_ollama:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=DEFAULT_LLM_TEMPERATURE
            )
        else:
            from langchain_openai import ChatOpenAI
            api_key = settings.siliconflow_api_key or settings.openai_api_key
            return ChatOpenAI(
                model=settings.openai_model,
                temperature=DEFAULT_LLM_TEMPERATURE,
                max_tokens=settings.summary_max_tokens * max_tokens_multiplier,
                api_key=api_key,
                base_url=settings.openai_api_base
            )

    async def _summarize_chunks(
        self,
        chunks: List[SemanticChunk],
        summary: HierarchicalSummary,
        doc_id: str
    ):
        """Generate summaries for individual chunks."""
        llm = self._create_llm(max_tokens_multiplier=1)

        prompts = []

        for chunk in chunks:
            prompt = self._create_chunk_summary_prompt(chunk.content)
            prompts.append((chunk.chunk_index, prompt))

        # Process in batches
        batch_size = settings.summary_batch_size

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            tasks = [
                self._generate_summary(llm, prompt, f"{doc_id}_chunk_{idx}")
                for idx, prompt in batch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (idx, _), result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(
                        "Failed to summarize chunk",
                        doc_id=doc_id,
                        chunk_index=idx,
                        error=str(result),
                        exc_info=True
                    )
                elif result:
                    summary.chunk_summaries[idx] = result
                    # Update chunk with summary
                    chunks[idx].summary = result
                    logger.debug(f"Generated summary for chunk {idx}: {result[:50]}...")

    async def _summarize_sections(
        self,
        chunks: List[SemanticChunk],
        summary: HierarchicalSummary,
        doc_id: str
    ):
        """Generate summaries for sections (groups of related chunks)."""
        sections = self._group_chunks_into_sections(chunks)
        if not sections:
            return

        llm = self._create_llm(max_tokens_multiplier=2)

        for section in sections:
            section_text = "\n\n".join(
                chunks[idx].content for idx in section.chunk_indices
            )

            prompt = self._create_section_summary_prompt(section_text)
            result = await self._generate_summary(
                llm,
                prompt,
                f"{doc_id}_section_{section.chunk_indices[0]}"
            )

            if result:
                section.summary = result

        summary.section_summaries = sections

    async def _summarize_document(
        self,
        chunks: List[SemanticChunk],
        summary: HierarchicalSummary,
        doc_id: str
    ):
        """Generate document-level summary."""
        if summary.section_summaries:
            source_text = "\n\n".join(
                s.summary for s in summary.section_summaries
            )
        else:
            source_text = "\n\n".join(
                chunks[idx].content for idx in list(summary.chunk_summaries.keys())[:20]
            )

        prompt = self._create_document_summary_prompt(source_text)
        llm = self._create_llm(max_tokens_multiplier=3)

        result = await self._generate_summary(llm, prompt, f"{doc_id}_document")

        if result:
            summary.document_summary = result

    def _group_chunks_into_sections(
        self,
        chunks: List[SemanticChunk]
    ) -> List[SectionSummary]:
        """
        Group chunks into sections based on semantic coherence.

        Args:
            chunks: List of semantic chunks

        Returns:
            List of section summaries (without summaries yet)
        """
        sections = []
        current_section = []
        chunks_per_section = max(3, self.section_threshold // 2)

        for i, chunk in enumerate(chunks):
            current_section.append(i)

            if len(current_section) >= chunks_per_section:
                sections.append(SectionSummary(
                    chunk_indices=current_section.copy(),
                    summary=""
                ))
                current_section = []

        # Add remaining chunks as last section
        if current_section:
            sections.append(SectionSummary(
                chunk_indices=current_section,
                summary=""
            ))

        return sections

    def _create_chunk_summary_prompt(self, content: str) -> str:
        """Create prompt for chunk-level summary."""
        return f"""请用简洁的中文总结以下文本的核心内容，不超过50字：

{content}

总结："""

    def _create_section_summary_prompt(self, content: str) -> str:
        """Create prompt for section-level summary."""
        return f"""请用简洁的中文总结以下章节内容的主要观点，不超过100字：

{content}

总结："""

    def _create_document_summary_prompt(self, content: str) -> str:
        """Create prompt for document-level summary."""
        return f"""请用简洁的中文总结以下文档的整体内容和主要要点，不超过200字：

{content}

总结："""

    async def _generate_summary(
        self,
        llm,
        prompt: str,
        log_id: str
    ) -> Optional[str]:
        """
        Generate summary using LLM.

        Args:
            llm: LLM instance
            prompt: Summary prompt
            log_id: ID for logging

        Returns:
            Generated summary text
        """
        try:
            from langchain_core.messages import HumanMessage

            logger.debug(f"Calling LLM for {log_id}, prompt length: {len(prompt)}")

            # Use ainvoke instead of agenerate for simpler async call
            response = await llm.ainvoke([HumanMessage(content=prompt)])

            logger.debug(f"LLM response for {log_id}, response type: {type(response)}")

            if response and hasattr(response, 'content'):
                summary = response.content.strip()
                if summary:
                    logger.info(f"Generated summary for {log_id}, length: {len(summary)}")
                    return summary
                else:
                    logger.warning(f"Empty summary content for {log_id}")
            else:
                logger.warning(f"Unexpected response format for {log_id}: {response}")

        except Exception as e:
            import traceback
            logger.error(
                f"Failed to generate summary for {log_id}: {str(e)}\n{traceback.format_exc()}",
                log_id=log_id,
                error=str(e)
            )

        return None


# Global instances with cache for different file_ext configurations
_semantic_chunker_cache: Dict[Optional[str], SemanticChunker] = {}
_hierarchical_summarizer: Optional[HierarchicalSummarizer] = None


def get_semantic_chunker(file_ext: Optional[str] = None) -> SemanticChunker:
    """
    Get or create a semantic chunker instance, cached by file_ext.

    Args:
        file_ext: File extension for auto-tuning chunk sizes.
            When provided, creates/retrieves a chunker optimized for that file type.
            When None, returns the default chunker.

    Returns:
        SemanticChunker instance (cached per file_ext)
    """
    cache_key = file_ext
    if cache_key not in _semantic_chunker_cache:
        _semantic_chunker_cache[cache_key] = SemanticChunker(file_ext=file_ext)
    return _semantic_chunker_cache[cache_key]


def get_hierarchical_summarizer() -> HierarchicalSummarizer:
    """Get or create the global hierarchical summarizer instance."""
    global _hierarchical_summarizer
    if _hierarchical_summarizer is None:
        _hierarchical_summarizer = HierarchicalSummarizer()
    return _hierarchical_summarizer


async def semantic_chunk_text(text: str, file_ext: Optional[str] = None) -> List[SemanticChunk]:
    """
    Convenience function to chunk text semantically.

    Args:
        text: Input text
        file_ext: File extension for auto-tuning chunk sizes

    Returns:
        List of semantic chunks
    """
    if not settings.enable_semantic_chunking:
        # Fallback to simple splitting if semantic chunking is disabled
        return [SemanticChunk(
            content=text,
            start_pos=0,
            end_pos=len(text),
            chunk_index=0
        )]

    chunker = get_semantic_chunker(file_ext=file_ext)
    return await chunker.chunk_text(text)


async def create_hierarchical_summary(
    chunks: List[SemanticChunk],
    doc_id: str
) -> HierarchicalSummary:
    """
    Convenience function to create hierarchical summaries.

    Args:
        chunks: List of semantic chunks
        doc_id: Document ID

    Returns:
        Hierarchical summary
    """
    if not settings.enable_hierarchical_summary:
        return HierarchicalSummary(
            chunk_summaries={},
            section_summaries=[]
        )

    summarizer = get_hierarchical_summarizer()
    return await summarizer.summarize(chunks, doc_id)
