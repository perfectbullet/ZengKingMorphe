"""
RAG retrieval service with hybrid search.
"""
from typing import List, Dict, Any, Optional
from app.core.logging import get_logger
from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db

logger = get_logger(__name__)


class RAGRetrieval:
    """RAG retrieval service with hybrid search (vector + keyword)."""
    
    async def search(
        self,
        query: str,
        kb_ids: Optional[List[str]] = None,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using RAG.
        
        Args:
            query: Search query
            kb_ids: Knowledge base IDs to search (optional)
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search (vector + keyword)
            
        Returns:
            List of relevant documents with scores
        """
        try:
            if use_hybrid:
                return await self._hybrid_search(query, kb_ids, top_k)
            else:
                return await self._vector_search(query, kb_ids, top_k)
                
        except Exception as e:
            logger.error(f"RAG search failed: query={query}, error={str(e)}", exc_info=True)
            raise
    
    async def _vector_search(
        self,
        query: str,
        kb_ids: Optional[List[str]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Vector-based semantic search using Chroma.
        
        Args:
            query: Search query
            kb_ids: Knowledge base IDs filter
            top_k: Number of results
            
        Returns:
            Search results
        """
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
            
            # Format results
            documents = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                for i, doc_text in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 1.0
                    # Convert distance to similarity score (0-1)
                    similarity = max(0.0, 1.0 - distance)
                    
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    
                    documents.append({
                        "content": doc_text,
                        "score": similarity,
                        "doc_id": metadata.get("doc_id"),
                        "kb_id": metadata.get("kb_id"),
                        "chunk_index": metadata.get("chunk_index"),
                        "source": "vector"
                    })
            
            logger.info(f"Vector search completed: query={query[:100]}, results_count={len(documents)}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Vector search failed: query={query}, error={str(e)}", exc_info=True)
            return []
    
    async def _keyword_search(
        self,
        query: str,
        kb_ids: Optional[List[str]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Keyword-based search using ElasticSearch.
        
        Args:
            query: Search query
            kb_ids: Knowledge base IDs filter
            top_k: Number of results
            
        Returns:
            Search results
        """
        try:
            # Build ElasticSearch query
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["content^2", "summary"],
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
            
            # Format results
            documents = []
            if results and results.get("hits"):
                for hit in results["hits"]["hits"]:
                    source = hit["_source"]
                    score = hit["_score"]
                    
                    # Normalize score to 0-1 range (approximate)
                    normalized_score = min(1.0, score / 10.0)
                    
                    documents.append({
                        "content": source.get("content", ""),
                        "score": normalized_score,
                        "doc_id": source.get("doc_id"),
                        "kb_id": source.get("kb_id"),
                        "chunk_index": source.get("chunk_index"),
                        "source": "keyword"
                    })
            
            logger.info(f"Keyword search completed: query={query[:100]}, results_count={len(documents)}")

            return documents
            
        except Exception as e:
            logger.error(f"Keyword search failed: query={query}, error={str(e)}", exc_info=True)
            return []
    
    async def _hybrid_search(
        self,
        query: str,
        kb_ids: Optional[List[str]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and keyword search using RRF.
        
        Args:
            query: Search query
            kb_ids: Knowledge base IDs filter
            top_k: Number of results
            
        Returns:
            Fused search results
        """
        try:
            # Get results from both searches
            vector_results = await self._vector_search(query, kb_ids, top_k * 2)
            keyword_results = await self._keyword_search(query, kb_ids, top_k * 2)
            
            # Apply Reciprocal Rank Fusion (RRF)
            fused_results = self._rrf_fusion(vector_results, keyword_results, k=60)
            
            # Return top-k results
            return fused_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: query={query}, error={str(e)}", exc_info=True)
            # Fallback to vector search only
            return await self._vector_search(query, kb_ids, top_k)
    
    def _rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion algorithm.
        
        RRF formula: score(d) = Σ 1 / (k + rank_i(d))
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            k: RRF constant (default 60)
            
        Returns:
            Fused and ranked results
        """
        # Create document score map
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        # Add vector search scores
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = f"{result['doc_id']}_{result['chunk_index']}"
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = {
                    "content": result["content"],
                    "doc_id": result["doc_id"],
                    "kb_id": result["kb_id"],
                    "chunk_index": result["chunk_index"],
                    "rrf_score": 0.0,
                    "vector_score": result["score"],
                    "keyword_score": 0.0,
                    "vector_rank": rank,
                    "keyword_rank": None
                }
            
            # Add RRF score
            doc_scores[chunk_id]["rrf_score"] += 1.0 / (k + rank)
        
        # Add keyword search scores
        for rank, result in enumerate(keyword_results, start=1):
            chunk_id = f"{result['doc_id']}_{result['chunk_index']}"
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = {
                    "content": result["content"],
                    "doc_id": result["doc_id"],
                    "kb_id": result["kb_id"],
                    "chunk_index": result["chunk_index"],
                    "rrf_score": 0.0,
                    "vector_score": 0.0,
                    "keyword_score": result["score"],
                    "vector_rank": None,
                    "keyword_rank": rank
                }
            else:
                doc_scores[chunk_id]["keyword_score"] = result["score"]
                doc_scores[chunk_id]["keyword_rank"] = rank
            
            # Add RRF score
            doc_scores[chunk_id]["rrf_score"] += 1.0 / (k + rank)
        
        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        logger.info(f"RRF fusion completed: vector_count={len(vector_results)}", keyword_count=len(keyword_results), fused_count=len(sorted_docs))
        
        return sorted_docs


# Global RAG retrieval instance
rag_retrieval = RAGRetrieval()
