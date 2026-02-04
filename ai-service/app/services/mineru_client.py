"""
MinerU API client for PDF parsing with intelligent caching and chunking.

This client provides:
- PDF splitting into smaller chunks (8 pages per chunk)
- MD5-based caching to avoid reprocessing identical content
- MongoDB-based status tracking with mineru_ prefixed collections
- Automatic merging of processed chunks
"""
import os
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from httpx import AsyncClient
from pypdf import PdfReader, PdfWriter

from app.core.logging import get_logger
from app.core.config import settings
from app.core.database import get_database

logger = get_logger(__name__)


class MineruClient:
    """
    Client for MinerU PDF parsing API with intelligent caching and chunking.

    Features:
    - Splits large PDFs into 8-page chunks for faster processing
    - Uses MD5 hash for cache keys to avoid reprocessing
    - Tracks processing status in MongoDB (mineru_ prefixed collections)
    - Merges processed chunks back into complete results
    - Supports parallel processing of chunks
    """

    # Collection names with mineru_ prefix
    COLLECTION_CACHE = "mineru_cache"
    COLLECTION_JOBS = "mineru_jobs"

    # Default chunk size (pages per chunk)
    DEFAULT_PAGES_PER_CHUNK = 8

    def __init__(
        self,
        api_url: Optional[str] = None,
        pages_per_chunk: Optional[int] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize MinerU client.

        Args:
            api_url: MinerU API endpoint URL (default: from settings)
            pages_per_chunk: Number of pages to process per chunk (default: from settings)
            timeout: API request timeout in seconds (default: from settings)
        """
        self.api_url = api_url or settings.mineru_api_url
        self.pages_per_chunk = pages_per_chunk or settings.mineru_pages_per_chunk
        self.timeout = timeout or settings.mineru_timeout
        self.client: Optional[AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    def _clean_mineru_markdown(self, markdown_text: str) -> str:
        """
        清理MinerU输出的markdown噪音,提升embedding质量。

        清理规则:
        - 移除表格竖线 `|`
        - 移除markdown加粗标记 `**`
        - 移除markdown标题标记 `#`
        - 规范化连续空格和换行

        Args:
            markdown_text: 原始markdown文本

        Returns:
            清理后的文本
        """
        import re

        if not markdown_text:
            return ""

        # 移除表格竖线 (保留表格内容,只移除分隔符)
        text = re.sub(r'\|', ' ', markdown_text)

        # 移除markdown加粗标记
        text = re.sub(r'\*\*', '', text)

        # 移除markdown标题(但保留标题文本)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

        # 规范化连续空格为单个空格
        text = re.sub(r' {2,}', ' ', text)

        # 规范化连续换行(最多保留2个)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 移除制表符
        text = re.sub(r'\t+', ' ', text)

        return text.strip()

    def _calculate_md5(self, file_path: str, start_page: int, end_page: int) -> str:
        """
        Calculate MD5 hash for a PDF chunk.

        Uses normalized file path + page range + file size for uniqueness.
        This ensures the same file and page range always produces the same cache key.

        Note: We use file size instead of mtime to avoid cache invalidation
        when files are simply copied or touched. If the file content actually
        changes, the size will likely change too.

        Args:
            file_path: Path to PDF file
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (0-indexed)

        Returns:
            MD5 hash string
        """
        # Normalize to absolute path
        abs_path = os.path.abspath(file_path)

        # Use normalized file path, page range, and file size for hash
        # Using size instead of mtime ensures stability across runs
        file_stat = os.stat(abs_path)
        hash_input = f"{abs_path}_{start_page}_{end_page}_{file_stat.st_size}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    async def _split_pdf(self, file_path: str) -> List[Tuple[Path, int, int]]:
        """
        Split PDF into smaller chunks.

        Args:
            file_path: Path to PDF file

        Returns:
            List of (temp_file_path, start_page, end_page) tuples
        """
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            chunks = []

            # Create temp directory for chunks
            temp_dir = Path("temp_pdf_chunks")
            temp_dir.mkdir(exist_ok=True)

            logger.info(
                f"Splitting PDF into chunks",
                file=file_path,
                total_pages=total_pages,
                pages_per_chunk=self.pages_per_chunk
            )

            # Split into chunks
            for start_page in range(0, total_pages, self.pages_per_chunk):
                end_page = min(start_page + self.pages_per_chunk, total_pages)

                # Create new PDF for this chunk
                writer = PdfWriter()
                for page_num in range(start_page, end_page):
                    writer.add_page(reader.pages[page_num])

                # Save chunk to temp file
                chunk_filename = f"chunk_{start_page}_{end_page}.pdf"
                chunk_path = temp_dir / chunk_filename

                with open(chunk_path, "wb") as f:
                    writer.write(f)

                chunks.append((chunk_path, start_page, end_page))

                logger.debug(
                    f"Created PDF chunk",
                    chunk_file=chunk_filename,
                    start_page=start_page,
                    end_page=end_page
                )

            logger.info(f"Split PDF into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to split PDF: file_path={file_path}, error={str(e)}", exc_info=True)
            raise

    async def _call_mineru_api(
        self,
        file_path: str,
        start_page: int,
        end_page: int
    ) -> Dict[str, Any]:
        """
        Call MinerU API for a single PDF chunk.

        Args:
            file_path: Path to PDF file
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (0-indexed)

        Returns:
            API response JSON
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        # Calculate MD5 for cache key
        cache_key = self._calculate_md5(file_path, start_page, end_page)

        # Check cache first
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            logger.info(
                f"Using cached result for chunk",
                cache_key=cache_key,
                start_page=start_page,
                end_page=end_page
            )
            return cached_result

        # Prepare API request
        files = {
            'files': (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')
        }

        data = {
            'return_middle_json': 'true',
            'return_model_output': 'true',
            'return_md': 'true',
            'return_images': 'false',
            'start_page_id': str(start_page),
            'end_page_id': str(end_page),
            'parse_method': 'auto',
            'lang_list': 'ch',
            'output_dir': './output',
            'server_url': 'string',
            'return_content_list': 'true',
            'backend': 'pipeline',
            'table_enable': 'true',
            'response_format_zip': 'false',
            'formula_enable': 'true'
        }

        try:
            logger.info(
                f"Calling MinerU API for chunk",
                cache_key=cache_key,
                start_page=start_page,
                end_page=end_page
            )

            response = await self.client.post(
                self.api_url,
                files=files,
                data=data
            )
            response.raise_for_status()

            result = response.json()

            # Cache the result
            await self._save_cached_result(cache_key, result, start_page, end_page)

            logger.info(
                f"Successfully processed chunk",
                cache_key=cache_key,
                start_page=start_page,
                end_page=end_page
            )

            return result

        except Exception as e:
            logger.error(
                f"Failed to call MinerU API",
                error=str(e),
                start_page=start_page,
                end_page=end_page,
                exc_info=True
            )
            raise

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result from MongoDB.

        Args:
            cache_key: MD5 cache key

        Returns:
            Cached result or None
        """
        try:
            db = await get_database()
            collection = db[self.COLLECTION_CACHE]

            record = await collection.find_one({"_id": cache_key})
            if record:
                return record.get("result")
            return None

        except Exception as e:
            logger.warning(f"Failed to get cached result: cache_key={cache_key}, error={str(e)}")
            return None

    async def _save_cached_result(
        self,
        cache_key: str,
        result: Dict[str, Any],
        start_page: int,
        end_page: int
    ) -> None:
        """
        Save result to cache in MongoDB.

        Args:
            cache_key: MD5 cache key
            result: API result to cache
            start_page: Starting page number
            end_page: Ending page number
        """
        try:
            db = await get_database()
            collection = db[self.COLLECTION_CACHE]

            cache_record = {
                "_id": cache_key,
                "result": result,
                "start_page": start_page,
                "end_page": end_page,
                "created_at": datetime.utcnow(),
                "accessed_at": datetime.utcnow()
            }

            await collection.update_one(
                {"_id": cache_key},
                {"$set": cache_record},
                upsert=True
            )

            logger.debug(f"Saved result to cache: {cache_key}")

        except Exception as e:
            logger.warning(f"Failed to save cached result: cache_key={cache_key}, error={str(e)}")

    async def _create_job_record(self, file_path: str, total_chunks: int) -> str:
        """
        Create a job record in MongoDB for tracking progress.

        Args:
            file_path: Path to PDF file
            total_chunks: Total number of chunks to process

        Returns:
            Job ID
        """
        try:
            db = await get_database()
            collection = db[self.COLLECTION_JOBS]

            job_id = hashlib.md5(f"{file_path}_{datetime.utcnow().timestamp()}".encode()).hexdigest()

            job_record = {
                "_id": job_id,
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "status": "processing",
                "total_chunks": total_chunks,
                "processed_chunks": 0,
                "failed_chunks": 0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "results": []
            }

            await collection.insert_one(job_record)

            logger.info(
                f"Created job record",
                job_id=job_id,
                file_path=file_path,
                total_chunks=total_chunks
            )

            return job_id

        except Exception as e:
            logger.error(f"Failed to create job record: file_path={file_path}, error={str(e)}", exc_info=True)
            raise

    async def _update_job_progress(
        self,
        job_id: str,
        processed_result: Dict[str, Any],
        success: bool = True
    ) -> None:
        """
        Update job progress in MongoDB.

        Args:
            job_id: Job ID
            processed_result: Result from processing a chunk
            success: Whether the chunk was processed successfully
        """
        try:
            db = await get_database()
            collection = db[self.COLLECTION_JOBS]

            update_data = {
                "$set": {"updated_at": datetime.utcnow()}
            }

            if success:
                update_data["$inc"] = {"processed_chunks": 1}
                update_data["$push"] = {"results": processed_result}
            else:
                update_data["$inc"] = {"failed_chunks": 1}

            await collection.update_one({"_id": job_id}, update_data)

        except Exception as e:
            logger.warning(f"Failed to update job progress: job_id={job_id}, error={str(e)}")

    async def _finalize_job(self, job_id: str, final_result: Dict[str, Any]) -> None:
        """
        Mark job as completed and save final merged result.

        Args:
            job_id: Job ID
            final_result: Merged result from all chunks
        """
        try:
            db = await get_database()
            collection = db[self.COLLECTION_JOBS]

            await collection.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "completed",
                        "final_result": final_result,
                        "completed_at": datetime.utcnow()
                    }
                }
            )

            logger.info(f"Job completed: {job_id}")

        except Exception as e:
            logger.error(f"Failed to finalize job: job_id={job_id}, error={str(e)}", exc_info=True)

    async def _merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from multiple chunks into a single result.

        Args:
            results: List of chunk results

        Returns:
            Merged result
        """
        try:
            # Sort results by page range
            sorted_results = sorted(
                results,
                key=lambda x: (x.get("start_page", 0), x.get("end_page", 0))
            )

            # Initialize merged structure
            merged = {
                "status": "success",
                "chunks": len(sorted_results),
                "content": [],
                "metadata": {
                    "total_pages": sum(
                        r.get("result", {}).get("metadata", {}).get("pages_processed", 0)
                        for r in sorted_results
                    ),
                    "merged_at": datetime.utcnow().isoformat()
                }
            }

            # Merge content from each chunk
            # MinerU API returns: {backend, version, results: {chunk_X_Y: {md_content, ...}}}
            all_md_content = []
            total_pages = 0

            for result in sorted_results:
                chunk_data = result.get("result", {})

                # Check for MinerU API format: {results: {chunk_id: {md_content}}}
                if "results" in chunk_data:
                    chunk_results = chunk_data.get("results", {})
                    for chunk_id, chunk_content in chunk_results.items():
                        if isinstance(chunk_content, dict):
                            md_content = chunk_content.get("md_content", "")
                            if md_content:
                                all_md_content.append(md_content)
                            # Get pages processed if available
                            total_pages += chunk_content.get("pages_processed", 0)

                # Check for legacy format: {content: [...]}
                elif "content" in chunk_data:
                    if isinstance(chunk_data["content"], list):
                        merged["content"].extend(chunk_data["content"])
                    else:
                        merged["content"].append(chunk_data["content"])

            # If we found md_content, create content items for document_service
            if all_md_content:
                # Combine all markdown content and create content items
                combined_md = "\n\n".join(all_md_content)

                # 清理markdown噪音,提升embedding质量
                cleaned_md = self._clean_mineru_markdown(combined_md)

                # 同时返回原始内容和清理后的内容
                merged["content"] = [{"text": cleaned_md}]
                merged["metadata"]["original_markdown"] = combined_md  # 保留原始内容供调试
                merged["metadata"]["cleaned_markdown"] = cleaned_md  # 清理后的内容
                merged["metadata"]["total_pages"] = total_pages or len(sorted_results)

                logger.info(
                    f"Merged {len(sorted_results)} chunk results with md_content",
                    total_chars_original=len(combined_md),
                    total_chars_cleaned=len(cleaned_md),
                    chars_removed=len(combined_md) - len(cleaned_md)
                )
            else:
                logger.info(
                    f"Merged {len(sorted_results)} chunk results",
                    total_content_items=len(merged["content"])
                )

            return merged

        except Exception as e:
            logger.error(f"Failed to merge results: error={str(e)}", exc_info=True)
            raise

    async def process_pdf(
        self,
        file_path: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Process a PDF file through MinerU API with automatic chunking and caching.

        Args:
            file_path: Path to PDF file
            use_cache: Whether to use cached results

        Returns:
            Merged result from all chunks
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Starting PDF processing: file={file_path}, use_cache={use_cache}")

        # Split PDF into chunks
        chunks = await self._split_pdf(file_path)

        # Create job record
        job_id = await self._create_job_record(file_path, len(chunks))

        try:
            # Process chunks (in parallel with semaphore to limit concurrency)
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent API calls
            results = []

            async def process_chunk(chunk_info: Tuple[Path, int, int]):
                chunk_path, start_page, end_page = chunk_info

                async with semaphore:
                    try:
                        result = await self._call_mineru_api(str(chunk_path), start_page, end_page)

                        # Add metadata
                        result_with_meta = {
                            "result": result,
                            "start_page": start_page,
                            "end_page": end_page,
                            "processed_at": datetime.utcnow().isoformat()
                        }

                        await self._update_job_progress(job_id, result_with_meta, success=True)
                        return result_with_meta

                    except Exception as e:
                        logger.error(
                            f"Failed to process chunk",
                            start_page=start_page,
                            end_page=end_page,
                            error=str(e)
                        )
                        await self._update_job_progress(job_id, {}, success=False)
                        raise

            # Process all chunks
            tasks = [process_chunk(chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks)

            # Merge results
            final_result = await self._merge_results(results)

            # Finalize job
            await self._finalize_job(job_id, final_result)

            logger.info(
                f"Successfully processed PDF",
                file=file_path,
                job_id=job_id,
                total_chunks=len(chunks)
            )

            return final_result

        except Exception as e:
            logger.error(f"Failed to process PDF: file_path={file_path}, error={str(e)}", exc_info=True)

            # Update job status to failed
            try:
                db = await get_database()
                await db[self.COLLECTION_JOBS].update_one(
                    {"_id": job_id},
                    {
                        "$set": {
                            "status": "failed",
                            "error_message": str(e),
                            "failed_at": datetime.utcnow()
                        }
                    }
                )
            except Exception:
                pass

            raise

        finally:
            # Clean up temp files
            try:
                temp_dir = Path("temp_pdf_chunks")
                if temp_dir.exists():
                    for chunk_file in temp_dir.glob("*.pdf"):
                        chunk_file.unlink()
                    temp_dir.rmdir()
                    logger.debug("Cleaned up temp PDF chunks")
            except Exception as e:
                logger.warning(f"Failed to clean up temp files: error={str(e)}")

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a processing job.

        Args:
            job_id: Job ID

        Returns:
            Job status or None
        """
        try:
            db = await get_database()
            collection = db[self.COLLECTION_JOBS]

            job = await collection.find_one({"_id": job_id})
            if job:
                job.pop("_id", None)  # Remove MongoDB _id
            return job

        except Exception as e:
            logger.error(f"Failed to get job status: job_id={job_id}, error={str(e)}", exc_info=True)
            return None

    async def clear_cache(self, older_than_days: int = 30) -> int:
        """
        Clear old cache entries.

        Args:
            older_than_days: Delete cache entries older than this many days

        Returns:
            Number of entries deleted
        """
        try:
            db = await get_database()
            collection = db[self.COLLECTION_CACHE]

            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

            result = await collection.delete_many({
                "created_at": {"$lt": cutoff_date}
            })

            logger.info(f"Cleared {result.deleted_count} cache entries older than {older_than_days} days")
            return result.deleted_count

        except Exception as e:
            logger.error(f"Failed to clear cache: error={str(e)}", exc_info=True)
            return 0


def create_mineru_client() -> MineruClient:
    """
    Factory function to create MinerU client with settings from configuration.

    Returns:
        Configured MineruClient instance
    """
    return MineruClient(
        api_url=settings.mineru_api_url,
        pages_per_chunk=settings.mineru_pages_per_chunk,
        timeout=settings.mineru_timeout
    )


# Global client instance (lazy initialization)
mineru_client: Optional[MineruClient] = None


def get_mineru_client() -> MineruClient:
    """
    Get or create global MinerU client instance.

    Returns:
        MineruClient instance
    """
    global mineru_client
    if mineru_client is None:
        mineru_client = create_mineru_client()
    return mineru_client
