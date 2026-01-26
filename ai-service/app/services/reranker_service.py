"""
BGE Reranker Service

Provides document reranking using BGE-Reranker models for improved RAG retrieval quality.
Supports both local (FlagEmbedding library) and remote (API) reranking backends.
"""
import asyncio
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query relevance.

        Args:
            query: User query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        pass


class BGEReranker(BaseReranker):
    """
    BGE Reranker using FlagEmbedding library.

    Supports BGE-Reranker-Large and other BGE reranker models.
    Runs on CPU by default, can use GPU if available.

    Model: BAAI/bge-reranker-large
    - Input: (query, document) pairs
    - Output: Relevance score (higher = more relevant)
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-ranker-large",
        device: str = "cpu",  # cpu or cuda
        batch_size: int = 32,
        use_fp16: bool = False,
    ):
        """
        Initialize BGE Reranker.

        Args:
            model_name_or_path: Model name (for huggingface) or local path
            device: Device to run on (cpu or cuda)
            batch_size: Batch size for inference
            use_fp16: Whether to use FP16 precision (GPU only)
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self._reranker = None

    def _load_model(self):
        """Lazy load the reranker model."""
        if self._reranker is not None:
            return

        try:
            from FlagEmbedding import FlagReranker

            logger.info(
                "Loading BGE Reranker model",
                model=self.model_name_or_path,
                device=self.device,
            )

            self._reranker = FlagReranker(
                model_name_or_path=self.model_name_or_path,
                device=self.device,
                use_fp16=self.use_fp16,
            )

            logger.info("BGE Reranker model loaded successfully")

        except ImportError as e:
            logger.error(
                "FlagEmbedding library not installed",
                error="pip install -U FlagEmbedding",
                exc_info=True,
            )
            raise RuntimeError(
                "FlagEmbedding library is required for BGE Reranker. "
                "Install with: pip install -U FlagEmbedding"
            ) from e

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents using BGE Reranker.

        Args:
            query: User query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        # Limit top_k to number of documents
        top_k = min(top_k, len(documents))

        # Lazy load model
        self._load_model()

        logger.info(
            "Reranking documents",
            query_length=len(query),
            documents_count=len(documents),
            top_k=top_k,
        )

        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        try:
            # Prepare (query, document) pairs
            pairs = [[query, doc] for doc in documents]

            # Run reranking
            scores = await loop.run_in_executor(
                None,
                self._reranker.compute_score,
                pairs,
            )

            # Ensure scores is a list
            if not isinstance(scores, list):
                scores = list(scores)

            # Create (index, score) pairs and sort by score descending
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top_k results
            result = indexed_scores[:top_k]

            logger.info(
                "Reranking completed",
                top_k=top_k,
                max_score=max(result, key=lambda x: x[1])[1] if result else 0,
                min_score=min(result, key=lambda x: x[1])[1] if result else 0,
            )

            return result

        except Exception as e:
            logger.error(
                "Reranking failed",
                query=query[:100],
                error=str(e),
                exc_info=True,
            )
            # Fallback: return original order with zero scores
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]


class NoOpReranker(BaseReranker):
    """
    No-op reranker that returns documents in original order.

    Used as fallback when reranker is not available or for testing.
    """

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Return documents in original order with equal scores.

        Args:
            query: User query (ignored)
            documents: List of document texts
            top_k: Number of results to return

        Returns:
            List of (original_index, score) tuples
        """
        top_k = min(top_k, len(documents))
        return [(i, 1.0) for i in range(top_k)]


class HybridReranker(BaseReranker):
    """
    Hybrid reranker that combines multiple reranking strategies.

    Can combine:
    - BGE Reranker (semantic relevance)
    - Keyword matching (exact term overlap)
    - Position bias (prefer earlier results from initial retrieval)
    """

    def __init__(
        self,
        primary_reranker: BaseReranker,
        keyword_weight: float = 0.2,
        position_weight: float = 0.1,
    ):
        """
        Initialize hybrid reranker.

        Args:
            primary_reranker: Main reranker (e.g., BGEReranker)
            keyword_weight: Weight for keyword matching score (0-1)
            position_weight: Weight for position bias score (0-1)
        """
        self.primary_reranker = primary_reranker
        self.keyword_weight = keyword_weight
        self.position_weight = position_weight

    def _keyword_score(self, query: str, document: str) -> float:
        """Calculate keyword overlap score."""
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())

        if not query_terms:
            return 0.0

        # Jaccard similarity
        intersection = query_terms & doc_terms
        union = query_terms | doc_terms

        return len(intersection) / len(union) if union else 0.0

    def _position_score(self, index: int, total: int) -> float:
        """Calculate position bias score (prefer earlier results)."""
        if total <= 1:
            return 1.0
        # Exponential decay: 1.0 at index 0, ~0.37 at index = total/3
        import math
        return math.exp(-3.0 * index / total)

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Rerank using hybrid strategy.

        Args:
            query: User query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of (original_index, score) tuples
        """
        # Get primary reranker scores
        primary_results = await self.primary_reranker.rerank(query, documents, top_k=len(documents))

        # Build index -> score mapping
        primary_scores = {idx: score for idx, score in primary_results}

        # Calculate combined scores
        combined_scores = []
        for idx, doc in enumerate(documents):
            semantic_score = primary_scores.get(idx, 0.0)
            keyword_score = self._keyword_score(query, doc)
            position_score = self._position_score(idx, len(documents))

            # Normalize scores to 0-1 range (approximate)
            semantic_norm = min(semantic_score, 1.0)  # BGE scores can be > 1
            keyword_norm = keyword_score  # Already 0-1
            position_norm = position_score  # Already 0-1

            # Combined score with weights
            combined = (
                (1 - self.keyword_weight - self.position_weight) * semantic_norm +
                self.keyword_weight * keyword_norm +
                self.position_weight * position_norm
            )

            combined_scores.append((idx, combined))

        # Sort by combined score descending
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        return combined_scores[:top_k]


# Global reranker instance (lazy initialization)
_reranker: Optional[BaseReranker] = None


def get_reranker(
    reranker_type: str = "bge",
    **kwargs
) -> BaseReranker:
    """
    Get or create the global reranker instance.

    Args:
        reranker_type: Type of reranker ("bge", "noop", "hybrid")
        **kwargs: Additional arguments for reranker initialization

    Returns:
        Reranker instance
    """
    global _reranker

    if _reranker is not None:
        return _reranker

    logger.info("Creating reranker", type=reranker_type)

    if reranker_type == "bge":
        model_path = kwargs.get("model_path", "BAAI/bge-reranker-large")
        device = kwargs.get("device", "cpu")
        _reranker = BGEReranker(model_name_or_path=model_path, device=device)
    elif reranker_type == "noop":
        _reranker = NoOpReranker()
    elif reranker_type == "hybrid":
        primary = get_reranker(kwargs.get("primary_type", "bge"), **kwargs)
        _reranker = HybridReranker(
            primary_reranker=primary,
            keyword_weight=kwargs.get("keyword_weight", 0.2),
            position_weight=kwargs.get("position_weight", 0.1),
        )
    else:
        logger.warning(f"Unknown reranker type: {reranker_type}, using NoOpReranker")
        _reranker = NoOpReranker()

    return _reranker


async def rerank_documents(
    query: str,
    documents: List[str],
    top_k: int = 10,
    reranker_type: str = "bge",
) -> List[Tuple[int, float]]:
    """
    Convenience function to rerank documents.

    Args:
        query: User query
        documents: List of document texts to rerank
        top_k: Number of top results to return
        reranker_type: Type of reranker to use

    Returns:
        List of (original_index, score) tuples, sorted by score descending
    """
    reranker = get_reranker(reranker_type)
    return await reranker.rerank(query, documents, top_k=top_k)
