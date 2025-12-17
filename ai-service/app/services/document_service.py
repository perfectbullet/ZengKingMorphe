"""
Document processing service.
"""
import os
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiofiles
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import markdown

from app.core.logging import get_logger
from app.core.config import settings
from app.core.database import get_database
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.models.database import DocumentModel, DocumentChunkModel

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
        task_id: Optional[str] = None  # Add task_id for progress tracking
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
            
        Returns:
            Document ID
        """
        try:
            # Generate document ID
            doc_id = self._generate_doc_id(filename, kb_id)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            
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
                metadata=metadata or {}
            )
            
            await db.documents.insert_one(doc_model.model_dump())
            
            logger.info(
                "Started processing document",
                doc_id=doc_id,
                filename=filename,
                kb_id=kb_id
            )
            
            # Extract text
            text_content = await self._extract_text(file_path, file_ext)
            
            # Chunk document
            chunks = self._chunk_text(text_content, doc_id, kb_id)
            
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
            
            # Update document status
            await db.documents.update_one(
                {"doc_id": doc_id},
                {
                    "$set": {
                        "status": "completed",
                        "chunks_count": len(chunks),
                        "vectors_count": len(chunks),
                        "processed_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(
                "Completed processing document",
                doc_id=doc_id,
                chunks_count=len(chunks)
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
            except Exception:
                pass
            
            raise
    
    async def _extract_text(self, file_path: str, file_ext: str) -> str:
        """
        Extract text from document.
        
        Args:
            file_path: Path to file
            file_ext: File extension
            
        Returns:
            Extracted text content
        """
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        extractor = self.supported_formats[file_ext]
        return await extractor(file_path)
    
    async def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF."""
        try:
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error("Failed to extract PDF", file=file_path, error=str(e))
            raise
    
    async def _extract_docx(self, file_path: str) -> str:
        """Extract text from Word document."""
        try:
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.error("Failed to extract DOCX", file=file_path, error=str(e))
            raise
    
    async def _extract_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            logger.error("Failed to extract TXT", file=file_path, error=str(e))
            raise
    
    async def _extract_markdown(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                md_content = await f.read()
            # Convert to HTML then extract text
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
        except Exception as e:
            logger.error("Failed to extract Markdown", file=file_path, error=str(e))
            raise
    
    async def _extract_html(self, file_path: str) -> str:
        """Extract text from HTML file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                html_content = await f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()
        except Exception as e:
            logger.error("Failed to extract HTML", file=file_path, error=str(e))
            raise
    
    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        kb_id: str
    ) -> List[DocumentChunkModel]:
        """
        Split text into chunks.
        
        Args:
            text: Text content
            doc_id: Document ID
            kb_id: Knowledge base ID
            
        Returns:
            List of document chunks
        """
        chunks = []
        chunk_size = settings.chunk_size
        chunk_overlap = settings.chunk_overlap
        
        # Simple paragraph-based chunking with overlap
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            # If adding this paragraph exceeds chunk size, save current chunk
            if current_length + para_length > chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                
                chunk_model = DocumentChunkModel(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    kb_id=kb_id,
                    content=chunk_text,
                    chunk_index=chunk_index
                )
                chunks.append(chunk_model)
                
                chunk_index += 1
                
                # Keep last paragraph for overlap
                if chunk_overlap > 0:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(para)
            current_length += para_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
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
            "Chunked document",
            doc_id=doc_id,
            chunks_count=len(chunks),
            avg_chunk_size=sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0
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
        
        # Prepare data for batch operations
        chunk_texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        chunk_metadatas = [
            {
                "doc_id": chunk.doc_id,
                "kb_id": chunk.kb_id,
                "chunk_index": chunk.chunk_index
            }
            for chunk in chunks
        ]
        
        # Store in Chroma (with vectorization)
        await chroma_db.add_documents(
            collection_name="doc",
            documents=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        
        # Store in ElasticSearch
        for i, chunk in enumerate(chunks):
            await es_db.index_document(
                index="doc",
                doc_id=chunk.chunk_id,
                document={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            # Update progress if task_id provided
            if task_id:
                processed = i + 1
                progress = (processed / len(chunks)) * 100.0
                
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
            chunks[i].vector_id = chunk.chunk_id
        
        # Store chunks in MongoDB
        chunk_docs = [chunk.model_dump() for chunk in chunks]
        await db.document_chunks.insert_many(chunk_docs)
        
        logger.info(
            "Stored chunks",
            doc_id=doc_id,
            chunks_count=len(chunks)
        )
    
    def _generate_doc_id(self, filename: str, kb_id: str) -> str:
        """Generate unique document ID."""
        timestamp = datetime.utcnow().timestamp()
        content = f"{filename}_{kb_id}_{timestamp}"
        hash_obj = hashlib.md5(content.encode())
        return f"doc_{hash_obj.hexdigest()[:12]}"


# Global document processor instance
document_processor = DocumentProcessor()
