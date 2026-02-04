#!/usr/bin/env python3
"""
BGE Reranker V2 M3 API 客户端测试代码

API 格式:
POST http://<ip>:6006/v1/rerank
Headers:
  Authorization: Bearer sk-aaabbbcccdddeeefffggghhhiiijjjkkk
  Content-Type: application/json
Body:
{
  "model": "bge-reranker-v2-m3",
  "query": "苹果",
  "documents": ["...", "...", "..."]
}
"""

import requests
import json
from typing import List, Dict, Any


class BGERerankerClient:
    """BGE Reranker API 客户端"""

    def __init__(
        self,
        base_url: str = "http://192.168.8.233:8091",
        api_key: str = "sk-aaabbbcccdddeeefffggghhhiiijjjkkk"
    ):
        """
        初始化客户端

        Args:
            base_url: API 服务地址，如 http://192.168.8.233:8091
            api_key: API 密钥
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def rerank(
        self,
        query: str,
        documents: List[str],
        model: str = "bge-reranker-v2-m3"
    ) -> Dict[str, Any]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 候选文档列表
            model: 模型名称，默认 bge-reranker-v2-m3

        Returns:
            重排序结果，格式:
            {
                "model": "bge-reranker-v2-m3",
                "results": [
                    {"index": 2, "relevance_score": 0.95, "document": "..."},
                    {"index": 0, "relevance_score": 0.78, "document": "..."},
                    ...
                ]
            }
        """
        url = f"{self.base_url}/v1/rerank"

        payload = {
            "model": model,
            "query": query,
            "documents": documents
        }

        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

def print_separator(title: str = ""):
    """打印分隔线"""
    print("=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def main():
    """测试示例"""

    # 配置连接参数
    BASE_URL = "http://192.168.8.233:8091"
    API_KEY = "sk-aaabbbcccdddeeefffggghhhiiijjjkkk"

    client = BGERerankerClient(base_url=BASE_URL, api_key=API_KEY)

    # 1. 基础重排序示例 - 中文
    print_separator("示例1: 中文查询重排序")

    query = "什么是珐琅工艺"

    documents = [
        "珐琅是一种将玻璃质釉料附着在金属表面的装饰工艺，具有色彩鲜艳、历久弥新的特点。",
        "珠宝首饰设计需要考虑美学、工艺和佩戴舒适度等多个因素。",
        "中国古代珐琅工艺最早出现在元代，明清时期达到顶峰，代表作品有景泰蓝。",
        "现代珐琅工艺在传承传统技法的基础上，融入了现代设计理念和新技术。",
        "不锈钢是一种常用的金属材料，具有良好的耐腐蚀性和强度。",
        "笔者从大学时代起开始接触珐琅工艺，先后有机会向多位珐琅艺术家求教，自己对珐琅工艺的由衷喜爱和一点点粗浅认识，完全要感谢这几位老师的指导，其中包括日本的原典生先生、法国的Jean Francois先生和美国的Don Viehman先生。在学习和创作的过程中，我深深地为珐琅工艺的魅力所折服，也真切地体会到这种工艺的难以把握。我所接触过的优秀的珐琅艺术家，都无一例外地把自己毕生的精力投入珐琅工艺的研究和创作当中。这是因为珐琅工艺没有捷径，想要真正地掌握这门技艺，制作者需要在无数次实验、",
        "笔者从大学时代起开始接触珐琅工艺，先后有机会向多位珐琅艺术家求教，自己对珐琅工艺的由衷喜爱和一点点粗浅认识，完全要感谢这几位老师的指导，其中包括日本的原典生先生、法国的Jean Francois先生和美国的Don Viehman先生。在学习和创作的过程中，我深深地为珐琅工艺的魅力所折服，也真切地体会到这种工艺的难以把握。我所接触过的优秀的珐琅艺术家，都无一例外地把自己毕生的精力投入珐琅工艺的研究和创作当中。这是因为珐琅工艺没有捷径，想要真正地掌握这门技艺，制作者需要在无数次实验、无数次失败中找到自己的方法，只有经过这些失败并从失败中总结经验，才有可能逐渐地理解珐琅工艺，才能用它创作出满意的作品。每一件优秀的珐琅作品，都是技术和艺术的完美结合。对珐琅工艺来说，繁复、精细的工艺和优美、严谨的设计，这二者缺一不可。"
    ]

 
    print(f"查询: {query}")
    print(f"文档数量: {len(documents)}")
    print()

    try:
        result = client.rerank(query=query, documents=documents)

        print("重排序结果:")
        print("-" * 60)
        for item in result.get("results", []):
            idx = item.get("index", 0)
            score = item.get("relevance_score", 0)
            doc = item.get("document", documents[idx])
            print(f"  [{idx}] Score: {score:.4f} | {doc}")
        print()

    except requests.exceptions.HTTPError as e:
        print(f"HTTP 错误: {e}")
        print(f"响应内容: {e.response.text if e.response else 'N/A'}")
        return
    except Exception as e:
        print(f"错误: {e}")
        return

    # 2. 中文复杂查询
    print_separator("示例2: 技术文档搜索")

    query2 = "什么是深度学习?"
    documents2 = [
        "深度学习是机器学习的一个子领域，使用多层神经网络。",
        "今天股市大涨，投资者情绪高涨。",
        "神经网络是受生物大脑启发的计算模型。",
        "深度学习在图像识别、自然语言处理等领域有广泛应用。",
        "我每天早上都喝一杯咖啡提神。"
    ]

    print(f"查询: {query2}")

    result2 = client.rerank(query=query2, documents=documents2)

    print("Top 3 相关文档:")
    for i, item in enumerate(result2.get("results", [])[:3], 1):
        score = item.get("relevance_score", 0)
        doc = item.get("document", "")
        print(f"  {i}. [{score:.4f}] {doc}")
    print()

    # 3. 英文查询
    print_separator("示例3: 英文查询")

    query3 = "machine learning"
    documents3 = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather is beautiful today.",
        "Deep learning uses neural networks with many layers.",
        "I love pizza and pasta.",
        "Supervised learning requires labeled training data."
    ]

    print(f"Query: {query3}")

    result3 = client.rerank(query=query3, documents=documents3)

    print("Ranked results:")
    for item in result3.get("results", []):
        score = item.get("relevance_score", 0)
        doc = item.get("document", "")
        print(f"  [{score:.4f}] {doc}")
    print()

    # 4. RAG 场景示例
    print_separator("示例4: RAG 文档检索场景")

    rag_query = "Python 中如何处理异常?"
    rag_documents = [
        "Python 使用 try-except 语句来处理异常。",
        "Python 是一种高级编程语言。",
        "在 Python 中，可以使用 finally 块确保清理代码被执行。",
        "异常处理是编程中的重要概念。",
        "Python 支持多种数据类型，包括列表、字典和集合。"
    ]

    print(f"查询: {rag_query}")
    print()

    result4 = client.rerank(query=rag_query, documents=rag_documents)

    print("检索到的最相关文档:")
    best = result4.get("results", [{}])[0]
    print(f"  Score: {best.get('relevance_score', 0):.4f}")
    print(f"  Document: {best.get('document', 'N/A')}")
    print()

    # 5. 原始 JSON 输出
    print_separator("示例5: 原始 JSON 响应格式")

    print("完整的 API 响应:")
    print(json.dumps(result4, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
