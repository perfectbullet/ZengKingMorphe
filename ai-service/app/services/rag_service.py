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
    
    async def faq_hybrid_search(
        self,
        query: str,
        employee_id: str,
        faq_sim_threshold: float = 0.0,
        faq_top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        FAQ混合搜索（向量+关键词双路召回+RRF融合）。
        
        Args:
            query: 用户查询
            employee_id: 员工ID（用于过滤FAQ）
            faq_sim_threshold: FAQ相似度阈值（高于此值直接返回答案）
            faq_top_k: 返回Top-K个FAQ
            
        Returns:
            FAQ搜索结果列表（按RRF分数降序）
        """
        try:
            logger.info(f"FAQ hybrid search: query={query[:100]}, employee_id={employee_id}, threshold={faq_sim_threshold}, top_k={faq_top_k}")
            
            # Step 1: Vector search in ChromaDB
            vector_results = await self._faq_vector_search(query, employee_id, faq_top_k * 2)
            
            # Step 2: Keyword search in ElasticSearch
            keyword_results = await self._faq_keyword_search(query, employee_id, faq_top_k * 2)
            
            # Step 3: RRF fusion
            fused_results = self._faq_rrf_fusion(vector_results, keyword_results, k=60)
            logger.info(f"FAQ RRF fusion done: total_fused={fused_results}")
            # Step 4: Filter by similarity threshold and return top-k
            filtered_results = [
                result for result in fused_results
                if result["rrf_score"] >= faq_sim_threshold
            ]
            
            logger.info(f"FAQ hybrid search completed: total_fused={len(fused_results)}, above_threshold={len(filtered_results)}, returning_top_k={min(faq_top_k, len(filtered_results))}")
            
            return filtered_results[:faq_top_k]
            
        except Exception as e:
            logger.error(f"FAQ hybrid search failed: query={query}, error={str(e)}", exc_info=True)
            return []
    
    async def _faq_vector_search(
        self,
        query: str,
        employee_id: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        FAQ向量搜索（ChromaDB）。
        
        Args:
            query: 用户查询
            employee_id: 员工ID
            top_k: 返回Top-K
            
        Returns:
            向量搜索结果
        """
        try:
            # Query ChromaDB faqs collection
            where_filter = {"employee_id": employee_id}
            
            results = await chroma_db.query_documents(
                collection_name="faq",
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )
            
            # Format results
            faq_results = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                logger.info(f"FAQ vector search raw results: distances={results.get('distances', [[]])[0][:3] if results.get('distances') else 'None'}")
                for i, doc_text in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 2.0
                    # ChromaDB cosine distance: 0 (identical) to 2 (opposite)
                    # Convert to similarity: 1.0 (identical) to 0.0 (opposite)
                    similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                    
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    
                    if i < 3:  # Log first 3 results for debugging
                        logger.info(f"FAQ vector result {i+1}: distance={distance:.4f}, similarity={similarity:.4f}, faq_id={metadata.get('faq_id')}")
                    
                    faq_results.append({
                        "faq_id": metadata.get("faq_id"),
                        "question_name": metadata.get("question_name"),
                        "combined_text": doc_text,
                        "score": similarity,
                        "source": "vector"
                    })
            
            logger.info(f"FAQ vector search: query={query[:50]}, results_count={len(faq_results)}")
            
            return faq_results
            
        except Exception as e:
            logger.error(f"FAQ vector search failed: query={query}, error={str(e)}", exc_info=True)
            return []
    
    async def _faq_keyword_search(
        self,
        query: str,
        employee_id: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        FAQ关键词搜索（ElasticSearch）。
        
        Args:
            query: 用户查询
            employee_id: 员工ID
            top_k: 返回Top-K
            
        Returns:
            关键词搜索结果
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
                                    "fields": ["question_name^3", "similar_questions^2", "combined_text"],
                                    "type": "best_fields"
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"employee_id": employee_id}},
                            {"term": {"is_enable": 1}}
                        ]
                    }
                }
            }
            
            # Search ElasticSearch
            results = await es_db.search(
                index="faq",  # Use short identifier, will be mapped to digital_employee_faqs
                query=es_query,
                size=top_k
            )
            
            logger.info(f"FAQ keyword search ES results: total_hits={results.get('hits', {}).get('total', {}).get('value', 0)}, returned={len(results.get('hits', {}).get('hits', []))}")
            
            # Format results
            faq_results = []
            if results and results.get("hits"):
                for hit in results["hits"]["hits"]:
                    source = hit["_source"]
                    score = hit["_score"]
                    
                    # Normalize score to 0-1 range
                    normalized_score = min(1.0, score / 10.0)
                    
                    faq_results.append({
                        "faq_id": source.get("faq_id"),
                        "question_name": source.get("question_name"),
                        "combined_text": source.get("combined_text"),
                        "score": normalized_score,
                        "source": "keyword"
                    })
            
            logger.debug(f"FAQ keyword search: query={query[:50]}, results_count={len(faq_results)}")
            
            return faq_results
            
        except Exception as e:
            logger.error(f"FAQ keyword search failed: query={query}, error={str(e)}", exc_info=True)
            return []
    
    def _faq_rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        FAQ结果的RRF融合。
        
        Args:
            vector_results: 向量搜索结果
            keyword_results: 关键词搜索结果
            k: RRF常数（默认60）
            
        Returns:
            融合后的排序结果
        """
        faq_scores: Dict[str, Dict[str, Any]] = {}
        
        # Add vector search scores
        for rank, result in enumerate(vector_results, start=1):
            faq_id = result["faq_id"]
            if faq_id not in faq_scores:
                faq_scores[faq_id] = {
                    "faq_id": faq_id,
                    "question_name": result["question_name"],
                    "combined_text": result["combined_text"],
                    "rrf_score": 0.0,
                    "vector_score": result["score"],
                    "keyword_score": 0.0,
                    "vector_rank": rank,
                    "keyword_rank": None
                }
            
            faq_scores[faq_id]["rrf_score"] += 1.0 / (k + rank)
        
        # Add keyword search scores
        for rank, result in enumerate(keyword_results, start=1):
            faq_id = result["faq_id"]
            if faq_id not in faq_scores:
                faq_scores[faq_id] = {
                    "faq_id": faq_id,
                    "question_name": result["question_name"],
                    "combined_text": result["combined_text"],
                    "rrf_score": 0.0,
                    "vector_score": 0.0,
                    "keyword_score": result["score"],
                    "vector_rank": None,
                    "keyword_rank": rank
                }
            else:
                faq_scores[faq_id]["keyword_score"] = result["score"]
                faq_scores[faq_id]["keyword_rank"] = rank
            
            faq_scores[faq_id]["rrf_score"] += 1.0 / (k + rank)
        
        # Sort by RRF score
        sorted_faqs = sorted(
            faq_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        logger.debug(f"FAQ RRF fusion: vector_count={len(vector_results)}, keyword_count={len(keyword_results)}, fused_count={len(sorted_faqs)}")
        
        return sorted_faqs


# Global RAG retrieval instance
rag_retrieval = RAGRetrieval()
