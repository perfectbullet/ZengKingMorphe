"""
Mathematics Textbook Retrieval Service.

Specialized RAG retrieval for math textbook data with Q&A structure.
Uses embedding_text (questions only) for vector search but returns
context_text (answers) as the content for LLM context.
"""
from typing import List, Dict, Any, Optional
from app.core.logging import get_logger
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.core.config import settings
from app.services.rag_service import RAGRetrieval

logger = get_logger(__name__)

# Math textbook knowledge base ID
MATH_KB_ID = "kb_9abcbe4aa557"


class MathTextbookRetrieval(RAGRetrieval):
    """
    Specialized retrieval service for mathematics textbook knowledge base.

    Key differences from standard RAGRetrieval:
    1. Uses embedding_text (questions only) for vector search
    2. Returns context_text (answers) as the content field for LLM
    3. Preserves math-specific metadata (book_title, chapter_title, section_title)
    """

    async def _vector_search(
        self,
        query: str,
        kb_ids: Optional[List[str]],
        top_k: int,
        enable_rerank: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Vector-based semantic search using Chroma.

        For math textbook, uses context_text from metadata instead of
        the indexed document text (which is embedding_text/questions only).

        Args:
            query: Search query
            kb_ids: Knowledge base IDs filter (should only contain MATH_KB_ID)
            top_k: Number of results
            enable_rerank: Whether to rerank results

        Returns:
            Search results with context_text as content
        """

        # Early return for empty query to avoid BGE API errors
        if not query or not query.strip():
            logger.warning(
                "Math textbook vector search skipped: empty query",
                query_len=len(query) if query else 0
            )
            return []

        try:
            # Build filter
            where_filter = None
            if kb_ids:
                where_filter = {"kb_id": {"$in": kb_ids}}

            # Query Chroma
            results = await chroma_db.query_documents(
                collection_name="doc",
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )

            # Format results - use context_text from metadata
            documents = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                result_ids = results.get("ids", [[]])[0]  # Get chunk_ids from ChromaDB
                for i, doc_text in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 1.0
                    # Convert distance to similarity score (0-1)
                    similarity = max(0.0, 1.0 - distance)

                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    chunk_id = result_ids[i] if i < len(result_ids) else None

                    # For math textbook, use context_text (answer) as content
                    # Fall back to document text if context_text not available
                    content = metadata.get("context_text", doc_text)

                    documents.append({
                        "content": content,
                        "score": similarity,
                        "doc_id": metadata.get("doc_id"),
                        "chunk_id": chunk_id,  # 添加 chunk_id 用于后续查询 teaching_script_tts
                        "kb_id": metadata.get("kb_id"),
                        "chunk_index": metadata.get("chunk_index"),
                        "content_type": metadata.get("content_type", "unknown"),
                        "context_text": metadata['context_text'],  # For direct match logic
                        "source": "vector",
                        # Math textbook specific metadata
                        "book_title": metadata.get("book_title", ""),
                        "chapter_title": metadata.get("chapter_title", ""),
                        "section_title": metadata.get("section_title", ""),
                        # Keep original embedding text for reference
                        "embedding_text": doc_text,
                    })

            logger.info(
                "Math textbook vector search completed",
                query=query[:100],
                results_count=len(documents),
                kb_id=MATH_KB_ID
            )

            # Rerank if enabled
            if enable_rerank and documents:
                documents = await self._rerank(query, documents, top_k)

            return documents

        except Exception as e:
            logger.error(f"Math textbook vector search failed: query={query}, error={str(e)}", exc_info=True)
            return []

    async def _keyword_search(
        self,
        query: str,
        kb_ids: Optional[List[str]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Keyword-based search using ElasticSearch.

        For math textbook, uses context_text field instead of content field.

        Args:
            query: Search query
            kb_ids: Knowledge base IDs filter
            top_k: Number of results

        Returns:
            Search results with context_text as content
        """

        # Early return for empty query
        if not query or not query.strip():
            logger.warning(
                "Math textbook keyword search skipped: empty query",
                query_len=len(query) if query else 0
            )
            return []

        try:
            # Build ElasticSearch query
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["summary^3", "content"],
                                    "type": "best_fields"
                                }
                            }
                        ]
                    }
                }
            }

            # Add kb_id filter if provided
            if kb_ids:
                es_query["query"]["bool"]["filter"] = [
                    {"terms": {"kb_id": kb_ids}}
                ]

            # Search
            results = await es_db.search(
                index="doc",
                query=es_query,
                size=top_k
            )

            # Format results - use context_text for math textbook
            documents = []
            if results and results.get("hits"):
                for hit in results["hits"]["hits"]:
                    source = hit["_source"]
                    score = hit["_score"]

                    # Normalize score to 0-1 range (approximate)
                    normalized_score = min(1.0, score / 10.0)

                    # For math textbook, use context_text (answer) as content
                    # Fall back to content field if context_text not available
                    content = source.get("context_text", source.get("content", ""))

                    documents.append({
                        "content": content,
                        "score": normalized_score,
                        "doc_id": source.get("doc_id"),
                        "chunk_id": source.get("chunk_id"),  # 添加 chunk_id
                        "kb_id": source.get("kb_id"),
                        "chunk_index": source.get("chunk_index"),
                        "content_type": source.get("content_type", "unknown"),
                        "context_text": source.get("context_text", ""),  # For direct match logic
                        "source": "keyword",
                        # Math textbook specific metadata
                        "book_title": source.get("book_title", ""),
                        "chapter_title": source.get("chapter_title", ""),
                        "section_title": source.get("section_title", ""),
                    })

            logger.info(
                "Math textbook keyword search completed",
                query=query[:100],
                results_count=len(documents),
                kb_id=MATH_KB_ID
            )

            return documents

        except Exception as e:
            logger.error(f"Math textbook keyword search failed: query={query}, error={str(e)}", exc_info=True)
            return []


# Global instance for math textbook retrieval
math_textbook_retrieval = MathTextbookRetrieval()
