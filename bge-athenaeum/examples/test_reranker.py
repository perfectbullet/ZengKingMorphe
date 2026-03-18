"""
BGE Reranker V2-M3 服务客户端
使用 OpenAI SDK 风格的 API 调用 vLLM Rerank API

服务地址: http://192.168.8.233:8091

运行测试: python test_reranker.py

依赖: pip install openai requests
"""
from openai import OpenAI
from typing import List, Tuple, Union
import requests


class BGERerankerClient:
    """BGE-Reranker-V2-M3 客户端 (OpenAI SDK 风格)"""

    def __init__(
        self,
        base_url: str = "http://192.168.8.233:8091",
        api_key: str = "not-needed",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model_name = "bge-reranker-m3"
        # 初始化 OpenAI 客户端（用于统一风格）
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{self.base_url}/v1",
            timeout=timeout,
        )
        self._check_health()

    def _check_health(self):
        """检查服务健康状态"""
        try:
            # 使用 /health 端点
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                print("服务连接成功")
            else:
                # 尝试 /v1/models 端点
                resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
                resp.raise_for_status()
                print(f"服务连接成功: {resp.json().get('data', [{}])[0].get('id', 'unknown')}")
        except requests.RequestException as e:
            raise RuntimeError(f"服务连接失败，请确认容器已启动: {e}")

    def get_model_info(self) -> dict:
        """获取模型信息"""
        resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Union[int, None] = None,
    ) -> List[Tuple[int, str, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 K 个结果，None 返回全部

        Returns:
            List of (index, document, score) 按分数降序排列
        """
        if top_k is None:
            top_k = len(documents)

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

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
            doc = documents[idx] if idx is not None else ""
            ranked.append((idx, doc, score))

        return ranked

    def rerank_batch(
        self,
        queries: List[str],
        documents: List[str],
        top_k: Union[int, None] = None,
    ) -> List[List[Tuple[int, str, float]]]:
        """
        批量重排序（多个查询，同一文档集）

        Args:
            queries: 查询文本列表
            documents: 文档列表
            top_k: 每个查询返回前 K 个结果

        Returns:
            每个查询的重排序结果列表
        """
        all_results = []

        for i, query in enumerate(queries):
            print(f"  处理查询 {i + 1}/{len(queries)}: {query[:30]}...")
            ranked = self.rerank(query, documents, top_k)
            all_results.append(ranked)

        return all_results

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
        """
        results = self.rerank(query, [document], top_k=1)
        if results:
            return results[0][2]
        return -10.0


# ============ 使用示例 ============

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BGE Reranker V2-M3 服务测试 (OpenAI SDK 风格)")
    print("=" * 60)

    # 初始化客户端
    client = BGERerankerClient()  # 默认连接到 http://192.168.8.233:8091

    # 示例 1: 基本重排序
    print("\n[示例 1] 基本文档重排序")
    query = "什么是人工智能？"
    documents = [
        "人工智能是指由人制造出来的机器所表现出来的智能。",
        "今天天气很好，适合出去散步。",
        "机器学习是人工智能的一个分支。",
        "我昨天吃了一顿美味的晚餐。",
        "深度学习使用神经网络模拟人脑。",
    ]

    results = client.rerank(query, documents, top_k=3)
    print(f"  查询: {query}")
    print(f"  重排序结果 (Top {len(results)}):")
    for i, (idx, doc, score) in enumerate(results, 1):
        print(f"    {i}. [{score:.4f}] {doc}")

    # 示例 2: 跨语言重排序
    print("\n[示例 2] 跨语言重排序 (中文查询 - 英文文档)")
    query = "机器学习算法"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather is sunny today.",
        "Deep learning uses neural networks for complex tasks.",
        "I had pizza for dinner last night.",
        "Supervised learning requires labeled training data.",
    ]

    results = client.rerank(query, documents, top_k=3)
    print(f"  查询: {query}")
    print(f"  重排序结果:")
    for i, (idx, doc, score) in enumerate(results, 1):
        print(f"    {i}. [{score:.4f}] {doc}")

    # 示例 3: RAG 场景
    print("\n[示例 3] RAG 检索重排序场景")
    knowledge_base = [
        "BGE-Reranker-V2-M3 是 BAAI 开发的轻量级重排序模型。",
        "该模型支持 8192 tokens 的上下文长度。",
        "模型采用中英双语训练，支持跨语言检索。",
        "Reranker 用于对检索结果进行重新排序，提高最终结果的相关性。",
        "BGE-M3 是多语言嵌入模型，而 BGE-Reranker-V2-M3 是重排序模型。",
        "使用半精度 (FP16) 推理可以节省显存并加速计算。",
        "vLLM 是一个高性能的大语言模型推理引擎。",
        "Python 是一种广泛使用的编程语言。",
    ]

    query = "BGE-Reranker 支持多长的上下文？"
    results = client.rerank(query, knowledge_base, top_k=3)
    print(f"  查询: {query}")
    print(f"  最相关文档:")
    for i, (idx, doc, score) in enumerate(results, 1):
        print(f"    {i}. [{score:.4f}] {doc}")

    # 示例 4: 批量查询
    print("\n[示例 4] 批量查询处理")
    queries = [
        "Python 如何读取文件？",
        "什么是深度学习？",
        "Reranker 的作用是什么？",
    ]

    docs = [
        "Python 使用 open() 函数读取文件。",
        "深度学习是机器学习的子集，使用神经网络。",
        "Java 是一种面向对象的编程语言。",
        "Reranker 对检索结果重新排序，提高相关性。",
        "TensorFlow 是深度学习框架。",
        "C++ 是一种高性能编程语言。",
    ]

    all_results = client.rerank_batch(queries, docs, top_k=2)
    for query, results in zip(queries, all_results):
        print(f"\n  查询: {query}")
        for i, (idx, doc, score) in enumerate(results, 1):
            print(f"    {i}. [{score:.4f}] {doc}")

    # 示例 5: 分数范围测试
    print("\n[示例 5] 分数范围测试")
    query = "科技"
    documents = [
        "人工智能和机器学习正在改变世界。",
        "今天去公园散步。",
        "Python 和 Java 是流行的编程语言。",
        "昨晚的晚餐很美味。",
        "深度学习神经网络需要大量计算资源。",
    ]

    results = client.rerank(query, documents, top_k=None)
    print(f"  查询: {query}")
    print(f"  所有文档及分数:")
    scores = [r[2] for r in results]
    print(f"    分数范围: {min(scores):.4f} ~ {max(scores):.4f}")
    for i, (idx, doc, score) in enumerate(results, 1):
        status = "✓ 相关" if score > 0 else "✗ 不相关"
        print(f"    {i}. [{score:.4f}] {doc} ({status})")

    # 示例 6: 模型信息
    print("\n[示例 6] 模型信息")
    model_info = client.get_model_info()
    print(f"  模型: {model_info.get('data', [{}])[0].get('id', 'unknown')}")
    print(f"  类型: {model_info.get('object', 'unknown')}")

    # 示例 7: 单个文档相关性计算
    print("\n[示例 7] 单个文档相关性计算")
    query = "机器学习"
    doc = "机器学习是人工智能的一个重要分支，研究如何使计算机系统从数据中学习。"
    score = client.compute_relevance(query, doc)
    print(f"  查询: {query}")
    print(f"  文档: {doc}")
    print(f"  相关性分数: {score:.4f}")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60 + "\n")
