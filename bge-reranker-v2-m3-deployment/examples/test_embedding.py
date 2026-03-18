"""
BGE-M3 服务客户端
使用 OpenAI SDK 调用 vLLM Embedding API

服务地址: http://192.168.8.233:8092

运行测试: python test_embedding.py

依赖: pip install openai numpy
"""
from openai import OpenAI
from typing import List, Tuple, Union
import numpy as np
import time


class BGEEmbeddingClient:
    """BGE-M3 嵌入模型客户端 (使用 OpenAI SDK)"""

    def __init__(
        self,
        base_url: str = "http://192.168.8.233:8092",
        api_key: str = "not-needed",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model_name = "BAAI/bge-m3"
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{self.base_url}/v1",
            timeout=timeout,
        )
        self._check_health()

    def _check_health(self):
        """检查服务健康状态"""
        try:
            # 使用 v1/health 端点
            import requests
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                print("服务连接成功")
            else:
                # 尝试 v1/models 端点
                resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
                resp.raise_for_status()
                print(f"服务连接成功: {resp.json().get('data', [{}])[0].get('id', 'unknown')}")
        except Exception as e:
            raise RuntimeError(f"服务连接失败，请确认容器已启动: {e}")

    def get_model_info(self) -> dict:
        """获取模型信息"""
        import requests
        resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> Union[List[float], List[List[float]]]:
        """
        生成文本嵌入向量

        Args:
            texts: 单个文本或文本列表
            normalize: 是否对向量进行 L2 归一化

        Returns:
            单个文本: 返回 List[float]
            多个文本: 返回 List[List[float]]
        """
        single_input = isinstance(texts, str)
        texts = [texts] if single_input else texts

        # 使用 OpenAI SDK 生成嵌入
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            encoding_format="float",
        )

        # 提取嵌入向量
        embeddings = [item.embedding for item in response.data]

        if normalize:
            embeddings = [self._l2_normalize(emb) for emb in embeddings]

        return embeddings[0] if single_input else embeddings

    def _l2_normalize(self, vector: List[float]) -> List[float]:
        """L2 归一化"""
        vec = np.array(vector)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vector
        return (vec / norm).tolist()

    def similarity(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, str, float]]:
        """
        计算查询与文档的相似度

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 K 个最相关的文档

        Returns:
            List of (index, document, score) 按分数降序排列
        """
        query_emb = self.embed(query, normalize=True)
        doc_embs = self.embed(documents, normalize=True)

        scores = []
        for i, doc_emb in enumerate(doc_embs):
            score = self._cosine_similarity(query_emb, doc_emb)
            scores.append((i, documents[i], score))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        批量生成嵌入向量（大列表分批处理）

        Args:
            texts: 文本列表
            batch_size: 每批处理的文本数量
            normalize: 是否对向量进行 L2 归一化

        Returns:
            嵌入向量列表
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  处理批次 {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            embeddings = self.embed(batch, normalize=normalize)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def test_long_text_embedding(
        self,
        lengths: List[int] = [1000, 2000, 4000, 8000],
    ) -> None:
        """
        测试不同长度的中文文本嵌入

        Args:
            lengths: 要测试的文本长度列表（字符数）
        """
        # 生成测试用的中文字符
        chars = "人工智能是计算机科学的一个分支机器学习深度学习神经网络自然语言处理计算机视觉数据挖掘大数据云计算物联网区块链元宇宙数字孪生边缘计算量子计算算法编程软件硬件网络通信操作系统数据库前端后端全栈开发敏捷开发DevOps持续集成持续部署"

        print(f"\n{'=' * 60}")
        print(f"长文本嵌入测试 (测试 {len(lengths)} 种长度)")
        print(f"{'=' * 60}")

        results = []

        for length in lengths:
            print(f"\n测试 {length} 个中文字符...")

            # 生成指定长度的文本
            text = (chars * ((length // len(chars)) + 1))[:length]
            print(f"  实际文本长度: {len(text)} 个字符")

            try:
                start_time = time.time()
                embedding = self.embed(text, normalize=False)
                elapsed = time.time() - start_time

                print(f"  ✓ 成功! 耗时: {elapsed:.2f}秒")
                print(f"    嵌入维度: {len(embedding)}")
                print(f"    前5个值: {embedding[:5]}")

                results.append({
                    'length': length,
                    'success': True,
                    'time': elapsed,
                    'embedding_dim': len(embedding)
                })

            except Exception as e:
                print(f"  ✗ 失败! 错误: {str(e)[:100]}")
                results.append({
                    'length': length,
                    'success': False,
                    'error': str(e)[:100]
                })

        # 打印汇总
        print(f"\n{'=' * 60}")
        print("测试结果汇总")
        print(f"{'=' * 60}")
        print(f"{'长度':<10} {'状态':<10} {'耗时(秒)':<15} {'嵌入维度':<10}")
        print("-" * 60)
        for r in results:
            status = "✓ 成功" if r['success'] else "✗ 失败"
            time_str = f"{r['time']:.2f}" if r.get('time') else "N/A"
            dim_str = f"{r.get('embedding_dim', 'N/A')}" if r['success'] else "N/A"
            print(f"{r['length']:<10} {status:<10} {time_str:<15} {dim_str:<10}")

        return results


# ============ 使用示例 ============

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BGE-M3 Embedding 服务测试 (OpenAI SDK)")
    print("=" * 60)

    # 初始化客户端
    client = BGEEmbeddingClient()  # 默认连接到 http://192.168.8.233:8092

    # 示例 1: 基本嵌入生成
    print("\n[示例 1] 基本文本嵌入")
    text = "人工智能是计算机科学的一个分支"
    embedding = client.embed(text)
    print(f"  文本: {text}")
    print(f"  嵌入维度: {len(embedding)}")
    print(f"  前10个维度: {embedding[:10]}")

    # 示例 2: 批量嵌入生成
    print("\n[示例 2] 批量文本嵌入")
    texts = [
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "今天天气很好",
        "Python 是一种编程语言",
    ]
    embeddings = client.embed(texts)
    print(f"  文本数量: {len(texts)}")
    print(f"  嵌入数量: {len(embeddings)}")
    print(f"  每个嵌入维度: {len(embeddings[0])}")

    # 示例 3: 相似度计算
    print("\n[示例 3] 文本相似度检索")
    query = "什么是机器学习？"
    documents = [
        "机器学习是人工智能的一个重要领域。",
        "今天我去了公园散步。",
        "深度学习基于神经网络模型。",
        "Python 是最流行的编程语言之一。",
        "监督学习需要标注数据。",
    ]

    results = client.similarity(query, documents, top_k=3)
    print(f"  查询: {query}")
    print(f"  最相关文档 (Top {len(results)}):")
    for i, (idx, doc, score) in enumerate(results, 1):
        print(f"    {i}. [{score:.4f}] {doc}")

    # 示例 4: 跨语言相似度
    print("\n[示例 4] 跨语言相似度 (中文查询 - 英文文档)")
    query = "深度学习"
    documents = [
        "Deep learning is a subset of machine learning.",
        "The weather is sunny today.",
        "Neural networks are foundation of deep learning.",
        "I had pizza for dinner.",
        "Python is used for machine learning and data science.",
    ]

    results = client.similarity(query, documents, top_k=3)
    print(f"  查询: {query}")
    print(f"  相关文档:")
    for i, (idx, doc, score) in enumerate(results, 1):
        print(f"    {i}. [{score:.4f}] {doc}")

    # 示例 5: 向量归一化效果
    print("\n[示例 5] 向量归一化对比")
    text1 = "机器学习"
    text2 = "机器学习算法"
    text3 = "深度学习"

    emb1_norm = client.embed(text1, normalize=True)
    emb2_norm = client.embed(text2, normalize=True)
    emb3_norm = client.embed(text3, normalize=True)

    norm1 = np.linalg.norm(emb1_norm)
    print(f"  归一化后向量范数: {norm1:.10f} (应接近1.0)")

    sim_12 = client._cosine_similarity(emb1_norm, emb2_norm)
    sim_13 = client._cosine_similarity(emb1_norm, emb3_norm)
    sim_23 = client._cosine_similarity(emb2_norm, emb3_norm)
    print(f"  相似度: '{text1}' vs '{text2}' = {sim_12:.4f}")
    print(f"  相似度: '{text1}' vs '{text3}' = {sim_13:.4f}")
    print(f"  相似度: '{text2}' vs '{text3}' = {sim_23:.4f}")

    # 示例 6: 模型信息
    print("\n[示例 6] 模型信息")
    model_info = client.get_model_info()
    print(f"  模型: {model_info.get('data', [{}])[0].get('id', 'unknown')}")
    print(f"  类型: {model_info.get('object', 'unknown')}")

    # 示例 7: 长文本嵌入测试
    print("\n[示例 7] 长文本嵌入测试")
    client.test_long_text_embedding([1000, 2000, 4000, 8000])

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60 + "\n")
