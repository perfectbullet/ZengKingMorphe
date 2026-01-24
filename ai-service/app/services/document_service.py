"""
Document processing service.
"""
import os
import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiofiles
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from app.core.logging import get_logger
from app.core.config import settings
from app.core.database import get_database
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.models.database import DocumentModel, DocumentChunkModel

# Import semantic chunking and hierarchical summarization
from app.services.semantic_chunking import (
    semantic_chunk_text,
    create_hierarchical_summary,
    SemanticChunk
)

logger = get_logger(__name__)


class DocumentProcessor:
    """Document processing service for RAG."""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._extract_pdf,
            '.docx': self._extract_docx,
            '.txt': self._extract_txt,
            '.md': self._extract_markdown,
            '.html': self._extract_html,
        }
    
    async def process_document(
        self,
        file_path: str,
        filename: str,
        kb_id: str,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,  # Add task_id for progress tracking
        chunk_config: Optional[Dict[str, Any]] = None,  # Custom chunk configuration
        doc_id: Optional[str] = None,  # Pre-generated doc_id (for immediate return)
        resource_id: Optional[int] = None  # External system resource ID
    ) -> str:
        """
        Process a document: extract text, chunk, vectorize, and store.

        Args:
            file_path: Path to the document file
            filename: Original filename
            kb_id: Knowledge base ID
            category: Document category (optional)
            metadata: Additional metadata (optional)
            task_id: Task ID for progress tracking (optional)
            chunk_config: Custom chunking configuration (optional)
            doc_id: Pre-generated document ID (optional, will be generated if not provided)
            resource_id: External system resource ID (optional)

        Returns:
            Document ID
        """
        try:
            # Use provided doc_id or generate a new one
            if not doc_id:
                doc_id = self._generate_doc_id(filename, kb_id)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(filename)[1].lower()

            # Merge metadata with resource_id
            doc_metadata = metadata.copy() if metadata else {}
            if resource_id:
                doc_metadata["resource_id"] = resource_id

            # Create document record
            db = await get_database()
            doc_model = DocumentModel(
                doc_id=doc_id,
                filename=filename,
                kb_id=kb_id,
                category=category,
                size=file_size,
                format=file_ext[1:].upper(),
                status="processing",
                segment_config=chunk_config,  # Save segment configuration
                metadata=doc_metadata
            )

            await db.documents.insert_one(doc_model.model_dump())

            logger.info(
                "Started processing document",
                doc_id=doc_id,
                filename=filename,
                kb_id=kb_id,
                custom_chunking=bool(chunk_config),
                resource_id=resource_id
            )

            # Extract text (check if we should use MinerU for PDFs)
            use_mineru = metadata.get('use_mineru', False) if metadata else False
            text_content = await self._extract_text(file_path, file_ext, use_mineru=use_mineru)

            # Preprocess text if chunk_config specifies
            if chunk_config:
                text_content = self._preprocess_text(text_content, chunk_config)

            # Chunk document (async for semantic chunking)
            # Returns (chunks, hierarchical_summary_data) tuple
            chunk_result = await self._chunk_text(text_content, doc_id, kb_id, file_ext=file_ext, chunk_config=chunk_config)

            # Handle different return types: tuple (semantic) or list (traditional)
            if isinstance(chunk_result, tuple):
                chunks, hierarchical_summary_data = chunk_result
            else:
                chunks = chunk_result
                hierarchical_summary_data = None

            # Update total_chunks if task_id provided
            if task_id:
                await db.document_tasks.update_one(
                    {"task_id": task_id},
                    {
                        "$set": {
                            "total_chunks": len(chunks),
                            "processed_chunks": 0
                        }
                    }
                )

            # Process chunks (vectorize and store)
            await self._process_chunks(chunks, doc_id, kb_id, task_id=task_id)

            # Build hierarchical summary data for storage
            hierarchical_summary_to_save = {}
            if hierarchical_summary_data:
                # Convert HierarchicalSummary object to dict for storage
                hierarchical_summary_to_save = {
                    "document_summary": hierarchical_summary_data.document_summary or "",
                    "section_summaries": [
                        {
                            "chunk_indices": s.chunk_indices,
                            "summary": s.summary
                        }
                        for s in (hierarchical_summary_data.section_summaries or [])
                    ]
                }

            # Update document status
            update_data = {
                "status": "completed",
                "chunks_count": len(chunks),
                "vectors_count": len(chunks),
                "segment_config": chunk_config,  # Persist segment configuration
                "processed_at": datetime.utcnow()
            }

            # Add hierarchical summary if available
            if hierarchical_summary_to_save:
                update_data["hierarchical_summary"] = hierarchical_summary_to_save

            await db.documents.update_one(
                {"doc_id": doc_id},
                {"$set": update_data}
            )

            logger.info(
                "Completed processing document",
                doc_id=doc_id,
                chunks_count=len(chunks),
                has_hierarchical_summary=bool(hierarchical_summary_to_save)
            )
            return doc_id
            
        except Exception as e:
            logger.error(
                "Failed to process document",
                filename=filename,
                error=str(e),
                exc_info=True
            )

            # Update document status to failed
            try:
                db = await get_database()
                await db.documents.update_one(
                    {"doc_id": doc_id},
                    {
                        "$set": {
                            "status": "failed",
                            "error_message": str(e)
                        }
                    }
                )
            except Exception as update_error:
                logger.warning(
                    "Failed to update document status to failed",
                    doc_id=doc_id,
                    error=str(update_error)
                )
            
            raise

    async def _extract_text(
        self,
        file_path: str,
        file_ext: str,
        use_mineru: bool = False
    ) -> str:
        """
        Extract text from document.

        Args:
            file_path: Path to file
            file_ext: File extension
            use_mineru: Whether to use MinerU API for PDF parsing (only for .pdf files)

        Returns:
            Extracted text content
        """
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Call extractor with use_mineru parameter for PDF files
        if file_ext == '.pdf' and use_mineru:
            return await self._extract_pdf(file_path, use_mineru=True)
        else:
            extractor = self.supported_formats[file_ext]
            return await extractor(file_path)

    async def _extract_pdf(self, file_path: str, use_mineru: bool = False) -> str:
        """
        Extract text from PDF.

        Args:
            file_path: Path to PDF file
            use_mineru: Whether to use MinerU API for enhanced PDF parsing

        Returns:
            Extracted text content
        """
        if use_mineru:
            return await self._extract_pdf_with_mineru(file_path)
        else:
            return await self._extract_pdf_basic(file_path)

    async def _extract_pdf_basic(self, file_path: str) -> str:
        """Extract text from PDF using basic PyPDF extraction."""
        try:
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error("Failed to extract PDF", file=file_path, error=str(e), exc_info=True)
            raise

    def _extract_content_from_result(self, content_items: list) -> list:
        """Extract text content from MinerU result items."""
        text_parts = []
        if isinstance(content_items, list):
            for item in content_items:
                if isinstance(item, dict):
                    text = item.get("text", "") or item.get("content", "")
                    if text:
                        text_parts.append(text)
                elif isinstance(item, str):
                    text_parts.append(item)
        return text_parts

    async def _extract_pdf_with_mineru(self, file_path: str) -> str:
        """
        Extract text from PDF using MinerU API with enhanced parsing.

        This method uses the MinerU client which provides:
        - Better table extraction
        - Formula parsing
        - Multi-column layout handling
        - Automatic caching for faster reprocessing

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        try:
            # Import here to avoid circular dependencies
            from app.services.mineru_client import get_mineru_client

            logger.info(f"Using MinerU API for PDF extraction: {file_path}")

            # Get client and process PDF with MinerU
            client = get_mineru_client()
            async with client:
                result = await client.process_pdf(file_path, use_cache=True)

            # Extract text content from result
            if result.get("status") == "success" and "content" in result:
                content_items = result["content"]
                text_parts = self._extract_content_from_result(content_items)
                extracted_text = "\n\n".join(text_parts)

                logger.info(
                    "Successfully extracted text with MinerU",
                    file=file_path,
                    content_length=len(extracted_text)
                )

                return extracted_text
            else:
                # Fallback to basic extraction if MinerU fails
                logger.warning(
                    "MinerU extraction failed or returned no content, falling back to basic extraction",
                    file=file_path,
                    status=result.get("status")
                )
                return await self._extract_pdf_basic(file_path)

        except ImportError:
            logger.warning("MinerU client not available, using basic PDF extraction")
            return await self._extract_pdf_basic(file_path)
        except Exception as e:
            logger.error(
                f"Failed to extract PDF with MinerU: {file_path}",
                error=str(e),
                exc_info=True
            )
            # Fallback to basic extraction
            logger.info(f"Falling back to basic PDF extraction: {file_path}")
            return await self._extract_pdf_basic(file_path)

    async def _extract_docx(self, file_path: str) -> str:
        """Extract text from Word document."""
        try:
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.error("Failed to extract DOCX", file=file_path, error=str(e), exc_info=True)
            raise
    
    async def _extract_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error("Failed to extract TXT", file=file_path, error=str(e), exc_info=True)
            raise
    
    async def _extract_markdown(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                md_content = await f.read()
            # Convert to HTML then extract text
            # html = markdown.markdown(md_content)
            # soup = BeautifulSoup(html, 'html.parser')
            return md_content
        except Exception as e:
            logger.error("Failed to extract Markdown", file=file_path, error=str(e), exc_info=True)
            raise
    
    async def _extract_html(self, file_path: str) -> str:
        """Extract text from HTML file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                html_content = await f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()
        except Exception as e:
            logger.error("Failed to extract HTML", file=file_path, error=str(e), exc_info=True)
            raise
    
    def _preprocess_text(self, text: str, chunk_config: Dict[str, Any]) -> str:
        """
        Preprocess text based on chunk configuration.
        
        Args:
            text: Raw text content
            chunk_config: Chunk configuration with preprocessing flags
            
        Returns:
            Preprocessed text
        """
        # Remove consecutive spaces, newlines, tabs
        if chunk_config.get('is_space_flag', 0) == 1:
            # Replace multiple spaces with single space
            text = re.sub(r' {2,}', ' ', text)
            # Replace multiple newlines with double newline (preserve paragraphs)
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Replace tabs with space
            text = re.sub(r'\t+', ' ', text)
            logger.info("Applied space/newline/tab preprocessing")
        
        # Remove table of contents, headers, footers (basic heuristic)
        if chunk_config.get('is_menu_flag', 0) == 1:
            # Remove common TOC patterns
            toc_patterns = [
                r'目录.*?(?=\n\n|\Z)',  # Chinese TOC
                r'Table of Contents.*?(?=\n\n|\Z)',  # English TOC
                r'^第[一二三四五六七八九十\d]+章.*$',  # Chapter titles
                r'^Chapter \d+.*$',  # English chapters
                r'页眉|页脚|Page \d+',  # Headers/footers
            ]
            for pattern in toc_patterns:
                text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
            logger.info("Applied TOC/header/footer removal")
        
        return text.strip()
    
    def _calculate_dynamic_chunk_params(self, total_chars: int, file_ext: str = '.txt') -> tuple[int, int]:
        """
        根据文档长度动态计算chunk_size和overlap。

        策略:
        - 短文档 (<1000字): 减少分段,保持完整性
        - 中等文档 (1000-5000字): 标准分段
        - 长文档 (5000-20000字): 加大overlap保证上下文连贯
        - 超长文档 (>20000字): 更小的chunk,更大的overlap

        Args:
            total_chars: 文档总字符数
            file_ext: 文件扩展名

        Returns:
            (chunk_size, chunk_overlap) 元组
        """
        # PDF使用MinerU时,由于markdown格式需要更大的chunk
        is_markdown_based = file_ext in ['.md', '.pdf']

        if total_chars < 1000:
            # 短文档: 保持完整性,减少分段
            if is_markdown_based:
                return 600, 60
            return 500, 50
        elif total_chars < 5000:
            # 中等文档: 标准分段
            if is_markdown_based:
                return 500, 80
            return 400, 60
        elif total_chars < 20000:
            # 长文档: 加大overlap保证上下文连贯
            if is_markdown_based:
                return 400, 100
            return 350, 80
        else:
            # 超长文档: 更小的chunk,更大的overlap
            if is_markdown_based:
                return 300, 120
            return 256, 100

    async def _chunk_text(
        self,
        text: str,
        doc_id: str,
        kb_id: str,
        file_ext: Optional[str] = None,
        chunk_config: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunkModel]:
        """
        Split text into chunks using semantic chunking.

        Note: chunk_config is saved for reference but semantic chunking is always used.
        Hierarchical summary is generated separately after semantic chunking,
        controlled by enable_hierarchical_summary setting.

        Args:
            text: Text content
            doc_id: Document ID
            kb_id: Knowledge base ID
            file_ext: File extension (e.g., '.md', '.html', '.txt', '.pdf')
            chunk_config: Custom chunking configuration (saved for reference only, not used)

        Returns:
            List of document chunks
        """
        # Always use semantic chunking (chunk_config is saved but not used for splitting)
        if settings.enable_semantic_chunking:
            return await self._semantic_chunk_text(text, doc_id, kb_id, file_ext)

        # Fallback to traditional chunking only if semantic chunking is disabled
        # Determine chunk size and overlap for traditional chunking
        if chunk_config:
            chunk_size = chunk_config.get('segment_union_max_length', settings.chunk_size)
            chunk_overlap = min(chunk_size // 10, 50)  # 10% overlap, max 50
            segment_type = chunk_config.get('segment_type', 0)
        else:
            # 使用动态分段策略
            chunk_size, chunk_overlap = self._calculate_dynamic_chunk_params(len(text), file_ext or '.txt')
            segment_type = -1  # Use default logic

            logger.info(
                "Using dynamic chunking params",
                doc_id=doc_id,
                total_chars=len(text),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        # Determine separators
        separators = None

        # Custom identifier-based splitting (segment_type=1)
        if segment_type == 1:
            identifier_type = chunk_config.get('segment_identifier_type', 0)

            if identifier_type == 0:  # System default identifiers
                # Parse identifier_default bitmap: "1111111" = [......, 。, ., ！, !, ？, ?]
                bitmap = chunk_config.get('identifier_default', '1111111')
                default_identifiers = ['......', '。', '.', '！', '!', '？', '?']
                separators = [default_identifiers[i] for i, bit in enumerate(bitmap) if bit == '1' and i < len(default_identifiers)]
                # Add fallback separators
                separators.extend(['\n\n', '\n', ' ', ''])
                logger.info(f"Using system default identifiers: separators={separators[:7]}")

            elif identifier_type == 1:  # Custom identifiers
                custom_str = chunk_config.get('identifier_customize', '')
                if custom_str:
                    # Parse custom identifiers (comma-separated or direct list)
                    separators = [s.strip() for s in custom_str.split(',') if s.strip()]
                    separators.extend(['\n\n', '\n', ' ', ''])  # Add fallbacks
                    logger.info(f"Using custom identifiers: separators={separators}")

        # Newline splitting (segment_type=0) or default logic
        if separators is None:
            if file_ext in ['.md', '.pdf']:  # PDF converted to markdown
                separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN)
            elif file_ext == '.html':
                separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.HTML)
            else:
                # Default separators for plain text
                separators = [
                    "\n\n",  # Paragraph boundary
                    "\n",    # Line break
                    "。",    # Chinese period
                    "！",    # Chinese exclamation
                    "？",    # Chinese question
                    ".",     # English period
                    "!",     # English exclamation
                    "?",     # English question
                    ";",     # Semicolon
                    ":",     # Colon
                    " ",     # Space
                    "",      # Character-level split (fallback)
                ]

        # Create text splitter with appropriate separators
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

        # Split text into chunks
        chunk_texts = text_splitter.split_text(text)

        # Convert to DocumentChunkModel objects
        chunks = []
        for chunk_index, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{doc_id}_chunk_{chunk_index}"

            chunk_model = DocumentChunkModel(
                chunk_id=chunk_id,
                doc_id=doc_id,
                kb_id=kb_id,
                content=chunk_text,
                chunk_index=chunk_index
            )
            chunks.append(chunk_model)

        logger.info(
            "Chunked document with RecursiveCharacterTextSplitter",
            doc_id=doc_id,
            file_ext=file_ext,
            chunks_count=len(chunks),
            avg_chunk_size=sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0
        )

        return chunks

    async def _semantic_chunk_text(
        self,
        text: str,
        doc_id: str,
        kb_id: str,
        file_ext: Optional[str] = None
    ) -> List[DocumentChunkModel]:
        """
        Split text into semantically coherent chunks using embeddings.

        Args:
            text: Text content
            doc_id: Document ID
            kb_id: Knowledge base ID
            file_ext: File extension

        Returns:
            List of document chunks with summaries
        """
        logger.info(
            "Using semantic chunking with hierarchical summarization",
            doc_id=doc_id,
            text_length=len(text),
            file_ext=file_ext
        )

        try:
            # Use semantic chunking service with file_ext for auto-tuning chunk sizes
            semantic_chunks = await semantic_chunk_text(text, file_ext=file_ext)

            # Generate hierarchical summaries and capture the result
            hierarchical_summary_data = None
            if settings.enable_hierarchical_summary:
                from app.services.semantic_chunking import HierarchicalSummary
                hierarchical_summary_data = await create_hierarchical_summary(semantic_chunks, doc_id)

            # Convert to DocumentChunkModel objects
            chunks = []
            for semantic_chunk in semantic_chunks:
                chunk_id = f"{doc_id}_chunk_{semantic_chunk.chunk_index}"

                # Build metadata with summary
                metadata = {
                    "chunk_type": "semantic",
                    "summary": semantic_chunk.summary or ""
                }

                chunk_model = DocumentChunkModel(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    kb_id=kb_id,
                    content=semantic_chunk.content,
                    chunk_index=semantic_chunk.chunk_index,
                    metadata=metadata
                )
                chunks.append(chunk_model)

            logger.info(
                "Semantic chunking completed",
                doc_id=doc_id,
                chunks_count=len(chunks),
                with_summaries=settings.enable_hierarchical_summary
            )

            # Return chunks with hierarchical summary data
            # Store it as an attribute for later use in process_document
            return chunks, hierarchical_summary_data

        except Exception as e:
            logger.error(
                "Semantic chunking failed, falling back to traditional chunking",
                doc_id=doc_id,
                error=str(e),
                exc_info=True
            )
            # Fall back to traditional chunking
            return await self._fallback_traditional_chunking(text, doc_id, kb_id, file_ext), None

    async def _fallback_traditional_chunking(
        self,
        text: str,
        doc_id: str,
        kb_id: str,
        file_ext: Optional[str] = None
    ) -> List[DocumentChunkModel]:
        """
        Fallback to traditional character-based chunking.

        Args:
            text: Text content
            doc_id: Document ID
            kb_id: Knowledge base ID
            file_ext: File extension

        Returns:
            List of document chunks
        """
        chunk_size, chunk_overlap = self._calculate_dynamic_chunk_params(len(text), file_ext or '.txt')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", ";", ":", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        chunk_texts = text_splitter.split_text(text)

        chunks = []
        for chunk_index, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{doc_id}_chunk_{chunk_index}"

            chunk_model = DocumentChunkModel(
                chunk_id=chunk_id,
                doc_id=doc_id,
                kb_id=kb_id,
                content=chunk_text,
                chunk_index=chunk_index
            )
            chunks.append(chunk_model)

        logger.info(
            "Fallback traditional chunking completed",
            doc_id=doc_id,
            chunks_count=len(chunks)
        )

        return chunks
    
    async def _process_chunks(
        self,
        chunks: List[DocumentChunkModel],
        doc_id: str,
        kb_id: str,
        task_id: Optional[str] = None  # Add task_id for progress tracking
    ) -> None:
        """
        Process chunks: vectorize and store in vector DB and ElasticSearch.
        
        Args:
            chunks: List of document chunks
            doc_id: Document ID
            kb_id: Knowledge base ID
            task_id: Task ID for progress tracking (optional)
        """
        db = await get_database()

        # Filter out empty chunks to avoid embedding dimension errors
        valid_chunks = [chunk for chunk in chunks if chunk.content and chunk.content.strip()]

        if not valid_chunks:
            logger.warning("No valid chunks to store after filtering empty content")
            return

        if len(valid_chunks) < len(chunks):
            logger.warning(
                f"Filtered out {len(chunks) - len(valid_chunks)} empty chunks, "
                f"remaining: {len(valid_chunks)}"
            )

        # Prepare data for batch operations
        chunk_texts = [chunk.content for chunk in valid_chunks]
        chunk_ids = [chunk.chunk_id for chunk in valid_chunks]
        chunk_metadatas = [
            {
                "doc_id": chunk.doc_id,
                "kb_id": chunk.kb_id,
                "chunk_index": chunk.chunk_index,
                "summary": chunk.metadata.get("summary", "") if chunk.metadata else ""
            }
            for chunk in valid_chunks
        ]

        # Store in Chroma (with vectorization)
        await chroma_db.add_documents(
            collection_name="doc",
            documents=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )

        # Store in ElasticSearch
        for i, chunk in enumerate(valid_chunks):
            # 获取 summary：优先从 chunk.summary，其次从 chunk.metadata.summary
            chunk_summary = chunk.summary if chunk.summary else ""
            if not chunk_summary and chunk.metadata:
                chunk_summary = chunk.metadata.get("summary", "")

            await es_db.index_document(
                index="doc",
                doc_id=chunk.chunk_id,
                document={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                    "content": chunk.content,
                    "summary": chunk_summary,
                    "chunk_index": chunk.chunk_index,
                    "created_at": datetime.utcnow().isoformat()
                }
            )

            # Update progress if task_id provided
            if task_id:
                processed = i + 1
                progress = (processed / len(valid_chunks)) * 100.0

                await db.document_tasks.update_one(
                    {"task_id": task_id},
                    {
                        "$set": {
                            "processed_chunks": processed,
                            "progress": round(progress, 2)
                        }
                    }
                )

            # Update chunk with vector_id
            valid_chunks[i].vector_id = chunk.chunk_id

        # Store chunks in MongoDB
        chunk_docs = [chunk.model_dump() for chunk in valid_chunks]
        await db.document_chunks.insert_many(chunk_docs)

        logger.info(f"Stored chunks: doc_id={doc_id}, chunks_count={len(valid_chunks)}")
    
    def _generate_doc_id(self, filename: str, kb_id: str) -> str:
        """Generate unique document ID."""
        return generate_doc_id(filename, kb_id)


def generate_doc_id(filename: str, kb_id: str) -> str:
    """
    Generate unique document ID.

    Args:
        filename: Document filename
        kb_id: Knowledge base ID

    Returns:
        Unique document ID
    """
    timestamp = datetime.utcnow().timestamp()
    content = f"{filename}_{kb_id}_{timestamp}"
    hash_obj = hashlib.md5(content.encode())
    return f"doc_{hash_obj.hexdigest()[:12]}"


# Global document processor instance
document_processor = DocumentProcessor()
