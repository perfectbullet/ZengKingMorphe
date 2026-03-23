"""
BGE Reranker 客户端

调用 BGE Reranker API 对检索结果进行重排序
API 地址: http://192.168.8.233:8091/v1/rerank
"""

import requests
from typing import List, Tuple, Union, Optional
from loguru import logger


class BGERerankerClientError(Exception):
    """BGE Reranker 客户端异常"""


class BGERerankerClient:
    """BGE Reranker 客户端"""

    def __init__(
        self,
        base_url: str = "http://192.168.8.233:8091",
        api_key: str = "not-needed",
        model: str = "bge-reranker-m3",
        timeout: int = 120,
    ):
        """
        初始化 BGE Reranker 客户端

        Args:
            base_url: BGE Reranker 服务地址
            api_key: API Key（vLLM 通常不需要）
            model: 模型名称
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        # 检查服务健康状态
        self._check_health()

    def _check_health(self) -> None:
        """
        检查服务健康状态

        Raises:
            BGERerankerClientError: 服务不可用时抛出异常
        """
        try:
            # 尝试 /health 端点
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info(f"BGE Reranker 服务连接成功: {self.base_url}")
                return
        except requests.RequestException:
            pass

        try:
            # 尝试 /v1/models 端点
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            resp.raise_for_status()
            model_id = resp.json().get('data', [{}])[0].get('id', 'unknown')
            logger.info(f"BGE Reranker 服务连接成功: {self.base_url}, 模型: {model_id}")
        except requests.RequestException as e:
            raise BGERerankerClientError(
                f"BGE Reranker 服务连接失败: {e}\n"
                f"请确认服务已启动: {self.base_url}"
            )

    def get_model_info(self) -> dict:
        """
        获取模型信息

        Returns:
            模型信息字典

        Raises:
            BGERerankerClientError: 获取失败时抛出异常
        """
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise BGERerankerClientError(f"获取模型信息失败: {e}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, str, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回前 N 个结果，None 返回全部

        Returns:
            List of (index, document, score) 按分数降序排列

        Raises:
            BGERerankerClientError: Rerank 请求失败时抛出异常
        """
        if not documents:
            return []

        if top_n is None:
            top_n = len(documents)

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        try:
            # 使用正确的端点: /v1/rerank
            resp = requests.post(
                f"{self.base_url}/v1/rerank",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            # vLLM rerank API 返回格式: {"results": [{"index": 0, "relevance_score": 8.5}, ...]}
            ranked = []
            for item in result.get("results", []):
                idx = item.get("index")
                score = item.get("relevance_score")
                doc = documents[idx] if idx is not None and idx < len(documents) else ""
                ranked.append((idx, doc, score))

            logger.debug(f"Rerank 完成: query='{query[:50]}...', 返回 {len(ranked)} 个结果")
            return ranked

        except requests.RequestException as e:
            raise BGERerankerClientError(
                f"Rerank 请求失败: {e}\n"
                f"查询: {query[:100]}...\n"
                f"文档数量: {len(documents)}"
            )

    def compute_relevance(
        self,
        query: str,
        document: str,
    ) -> float:
        """
        计算单个查询和文档的相关性分数

        Args:
            query: 查询文本
            document: 文档文本

        Returns:
            相关性分数 (-10 到 +10)

        Raises:
            BGERerankerClientError: 计算失败时抛出异常
        """
        results = self.rerank(query, [document], top_n=1)
        if results:
            return results[0][2]
        return 0.0
