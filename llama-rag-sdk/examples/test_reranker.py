"""
BGE Reranker 客户端测试脚本

测试 reranker 模块的各种功能：
1. 基本重排序
2. 跨语言重排序
3. RAG 场景
4. 批量查询
5. 分数范围测试
6. 单个文档相关性计算
7. 模型信息获取

运行: python examples/test_reranker.py
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_rag_sdk.retrieval.reranker import BGERerankerClient, BGERerankerClientError


def test_basic_rerank():
    """测试 1: 基本文档重排序"""
    print("\n" + "=" * 60)
    print("[测试 1] 基本文档重排序")
    print("=" * 60)

    client = BGERerankerClient()

    query = "什么是人工智能？"
    documents = [
        "人工智能是指由人制造出来的机器所表现出来的智能。",
        "今天天气很好，适合出去散步。",
        "机器学习是人工智能的一个分支。",
        "我昨天吃了一顿美味的晚餐。",
        "深度学习使用神经网络模拟人脑。",
    ]

    results = client.rerank(query, documents, top_n=3)
    print(f"查询: {query}")
    print(f"重排序结果 (Top {len(results)}):")
    for i, (idx, doc, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.4f}] {doc}")

    # 验证最相关文档排在第一位
    assert results[0][2] > 0, "最相关文档分数应该大于0"
    print("✓ 测试通过")


def test_cross_lingual():
    """测试 2: 跨语言重排序 (中文查询 - 英文文档)"""
    print("\n" + "=" * 60)
    print("[测试 2] 跨语言重排序")
    print("=" * 60)

    client = BGERerankerClient()

    query = "机器学习算法"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather is sunny today.",
        "Deep learning uses neural networks for complex tasks.",
        "I had pizza for dinner last night.",
        "Supervised learning requires labeled training data.",
    ]

    results = client.rerank(query, documents, top_n=3)
    print(f"查询: {query}")
    print(f"重排序结果:")
    for i, (idx, doc, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.4f}] {doc}")

    # 验证能识别跨语言相关性
    assert len(results) > 0, "应该返回结果"
    print("✓ 测试通过")


def test_rag_scenario():
    """测试 3: RAG 检索重排序场景"""
    print("\n" + "=" * 60)
    print("[测试 3] RAG 检索重排序场景")
    print("=" * 60)

    client = BGERerankerClient()

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
    results = client.rerank(query, knowledge_base, top_n=3)
    print(f"查询: {query}")
    print(f"最相关文档:")
    for i, (idx, doc, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.4f}] {doc}")

    # 验证返回正确数量
    assert len(results) == 3, "应该返回 3 个结果"
    print("✓ 测试通过")


def test_batch_queries():
    """测试 4: 批量查询处理"""
    print("\n" + "=" * 60)
    print("[测试 4] 批量查询处理")
    print("=" * 60)

    client = BGERerankerClient()

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

    all_results = client.rerank_batch(queries, docs, top_n=2)
    for query, results in zip(queries, all_results):
        print(f"\n查询: {query}")
        for i, (idx, doc, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.4f}] {doc}")

    # 验证批量结果
    assert len(all_results) == len(queries), "应该为每个查询返回结果"
    print("\n✓ 测试通过")


def test_score_range():
    """测试 5: 分数范围测试"""
    print("\n" + "=" * 60)
    print("[测试 5] 分数范围测试")
    print("=" * 60)

    client = BGERerankerClient()

    query = "科技"
    documents = [
        "人工智能和机器学习正在改变世界。",
        "今天去公园散步。",
        "Python 和 Java 是流行的编程语言。",
        "昨晚的晚餐很美味。",
        "深度学习神经网络需要大量计算资源。",
    ]

    results = client.rerank(query, documents, top_n=None)
    print(f"查询: {query}")
    print(f"所有文档及分数:")

    scores = [r[2] for r in results]
    print(f"  分数范围: {min(scores):.4f} ~ {max(scores):.4f}")

    for i, (idx, doc, score) in enumerate(results, 1):
        status = "✓ 相关" if score > 0 else "✗ 不相关"
        print(f"  {i}. [{score:.4f}] {doc} ({status})")

    # 验证分数范围
    assert max(scores) > min(scores), "最高分应该大于最低分"
    print("✓ 测试通过")


def test_model_info():
    """测试 6: 模型信息获取"""
    print("\n" + "=" * 60)
    print("[测试 6] 模型信息")
    print("=" * 60)

    client = BGERerankerClient()

    model_info = client.get_model_info()
    model_id = model_info.get('data', [{}])[0].get('id', 'unknown')
    obj_type = model_info.get('object', 'unknown')

    print(f"模型: {model_id}")
    print(f"类型: {obj_type}")

    # 验证返回模型信息
    assert model_id != 'unknown', "应该能获取模型 ID"
    print("✓ 测试通过")


def test_single_relevance():
    """测试 7: 单个文档相关性计算"""
    print("\n" + "=" * 60)
    print("[测试 7] 单个文档相关性计算")
    print("=" * 60)

    client = BGERerankerClient()

    query = "机器学习"
    doc = "机器学习是人工智能的一个重要分支，研究如何使计算机系统从数据中学习。"
    score = client.compute_relevance(query, doc)

    print(f"查询: {query}")
    print(f"文档: {doc}")
    print(f"相关性分数: {score:.4f}")

    # 验证相关文档得分高
    assert score > 0, "相关文档分数应该大于0"
    print("✓ 测试通过")


def test_empty_documents():
    """测试 8: 空文档列表处理"""
    print("\n" + "=" * 60)
    print("[测试 8] 空文档列表处理")
    print("=" * 60)

    client = BGERerankerClient()

    results = client.rerank("测试查询", [], top_n=3)
    print(f"空文档查询结果: {results}")

    assert len(results) == 0, "空文档应该返回空列表"
    print("✓ 测试通过")


def test_top_n_none():
    """测试 9: top_n=None 返回全部结果"""
    print("\n" + "=" * 60)
    print("[测试 9] top_n=None 返回全部结果")
    print("=" * 60)

    client = BGERerankerClient()

    documents = ["测试文档1", "测试文档2", "测试文档3", "测试文档4", "测试文档5"]
    results = client.rerank("测试查询", documents, top_n=None)

    print(f"输入文档数量: {len(documents)}")
    print(f"返回结果数量: {len(results)}")

    assert len(results) == len(documents), "应该返回所有文档"
    print("✓ 测试通过")


def test_long_documents():
    """测试 10: 长文档处理"""
    print("\n" + "=" * 60)
    print("[测试 10] 长文档处理")
    print("=" * 60)

    client = BGERerankerClient()

    # 模拟长文档（类似 RAG 检索结果）
    long_doc = """
    人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。
    """ * 10  # 重复10次模拟长文本

    query = "什么是人工智能？"
    results = client.rerank(query, [long_doc], top_n=1)

    print(f"查询: {query}")
    print(f"文档长度: {len(long_doc)} 字符")
    print(f"相关性分数: {results[0][2]:.4f}")

    assert len(results) == 1, "应该返回一个结果"
    print("✓ 测试通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("BGE Reranker 客户端测试套件")
    print("=" * 60)

    tests = [
        test_basic_rerank,
        test_cross_lingual,
        test_rag_scenario,
        test_batch_queries,
        test_score_range,
        test_model_info,
        test_single_relevance,
        test_empty_documents,
        test_top_n_none,
        test_long_documents,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ 测试失败: {e}")
            failed += 1
        except BGERerankerClientError as e:
            print(f"✗ 客户端错误: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ 未知错误: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed == 0:
        print("🎉 所有测试通过！")
        return 0
    else:
        print(f"⚠️  {failed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
