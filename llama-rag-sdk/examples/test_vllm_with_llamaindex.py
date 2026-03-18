"""
使用 LlamaIndex + vLLM Embedding API 测试真实文本

使用 LlamaIndex 官方 OpenAIEmbedding 类连接 BGE-M3 vLLM API
参考: bge-athenaeum/client/test_embedding.py
vLLM 服务地址: http://192.168.8.233:8092
"""
# 标准库导入
import os
import re
import time
import warnings
from pathlib import Path

# 禁用 LlamaIndex 遥测
os.environ["LlamaIndex_TELEMETRY"] = "false"

# 抑制 Pydantic 的 validate_default 警告 (Pydantic V2 兼容性问题)
warnings.filterwarnings("ignore", message=".*validate_default.*", category=UserWarning)

# 第三方库导入
import numpy as np
import chromadb

# LlamaIndex 导入
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore


def clean_text(text: str) -> str:
    """清理 markdown 文本"""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\(\)', r'\1', text)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    return text.strip()


def split_by_headings(text: str, filename: str) -> list[tuple[str, str]]:
    """按标题分割 markdown 文本"""
    lines = text.split("\n")
    sections = []
    current_heading = filename
    current_content = []

    for line in lines:
        if line.startswith("#"):
            if current_content:
                sections.append((current_heading, "\n".join(current_content)))
            current_heading = line.strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append((current_heading, "\n".join(current_content)))

    return sections


def test_basic_embedding():
    """测试基本 embedding 功能"""
    print("=" * 60)
    print("测试 1: 基本嵌入生成")
    print("=" * 60)

    embed_model = OpenAIEmbedding(
        model_name="BAAI/bge-m3",
        api_key="not-needed",
        api_base="http://192.168.8.233:8092/v1",
        embed_batch_size=32,
        timeout=300,
    )

    test_texts = [
        "珐琅是什么？",
        "珐琅工艺的历史",
        "釉料的成分",
        "烧制温度",
        "透明釉料的特点",
    ]

    print(f"\n处理 {len(test_texts)} 个文本...")
    embeddings = embed_model.get_text_embedding_batch(test_texts)

    print(f"✓ 成功生成 {len(embeddings)} 个 embedding")
    for i, (text, emb) in enumerate(zip(test_texts, embeddings)):
        print(f"  [{i}] {text:15s} -> 维度: {len(emb)}")

    return embeddings


def test_long_text():
    """测试长文本 embedding"""
    print("\n" + "=" * 60)
    print("测试 2: 长文本嵌入")
    print("=" * 60)

    embed_model = OpenAIEmbedding(
        model_name="BAAI/bge-m3",
        api_key="not-needed",
        api_base="http://192.168.8.233:8092/v1",
        embed_batch_size=32,
        timeout=300,
    )

    base_text = "珐琅工艺是一门历史悠久的手工艺，从古希腊的青铜时代发展到今天。"
    test_lengths = [100, 200, 500, 1000, 2000, 4000, 8000]

    print(f"\n测试 {len(test_lengths)} 种文本长度...")
    for length in test_lengths:
        text = base_text * ((length // len(base_text)) + 1)
        text = text[:length]

        start = time.time()
        emb = embed_model.get_text_embedding(text)
        elapsed = time.time() - start

        has_nan = any(isinstance(x, float) and str(x) == 'nan' for x in emb) if emb else False
        nan_info = " [包含 NaN!]" if has_nan else ""

        print(f"  长度 {length:5d} -> 维度: {len(emb) if emb else 'None':4d} 耗时: {elapsed:5.2f}s{nan_info}")

        if has_nan:
            print(f"    前 10 个值: {emb[:10]}")
            break


def test_real_markdown():
    """测试真实 markdown 文件并创建 LlamaIndex 索引"""
    print("\n" + "=" * 60)
    print("测试 3: 真实 Markdown 文件 + LlamaIndex 索引")
    print("=" * 60)

    # 读取珐琅工艺 markdown 文件
    md_file = Path("data/sample_md/01珐琅工艺-16pages-part1-page1-16.md")
    if not md_file.exists():
        print(f"文件不存在: {md_file}")
        return

    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 按标题分割
    sections = split_by_headings(content, md_file.name)
    print(f"\n文件包含 {len(sections)} 个章节")

    # 创建文档列表
    documents = []
    for heading, text in sections:
        cleaned = clean_text(text)
        if len(cleaned) >= 10:
            documents.append(Document(
                text=cleaned,
                metadata={
                    "source": md_file.name,
                    "heading": heading,
                    "length": len(cleaned)
                }
            ))

    print(f"有效文档数量: {len(documents)}\n")

    # 初始化 embedding 模型
    embed_model = OpenAIEmbedding(
        model_name="BAAI/bge-m3",
        api_key="not-needed",
        api_base="http://192.168.8.233:8092/v1",
        embed_batch_size=32,
        timeout=300,
    )

    # 创建 ChromaDB 向量存储
    chroma_client = chromadb.HttpClient(host="192.168.8.233", port=8200)
    collection_name = "test_vllm_llamaindex_openai"

    # 删除已存在的集合
    try:
        chroma_client.delete_collection(collection_name)
        print(f"已删除旧集合: {collection_name}")
    except Exception:
        pass

    collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 创建索引
    print("开始创建索引...")
    start = time.time()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    elapsed = time.time() - start
    print(f"✓ 索引创建完成! 耗时: {elapsed:.2f}秒")
    print(f"  文档数量: {len(documents)}")

    # 测试检索
    print("\n" + "-" * 60)
    print("测试检索功能")
    print("-" * 60)

    test_queries = [
        "什么是珐琅？",
        "珐琅釉料由什么组成？",
        "透明釉料有什么特点？",
        "珐琅工艺的起源",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        try:
            retriever = index.as_retriever(similarity_top_k=3)
            results = retriever.retrieve(query)

            for i, node in enumerate(results, 1):
                score = node.score if hasattr(node, 'score') else "N/A"
                text_preview = node.text[:100] + "..." if len(node.text) > 100 else node.text
                print(f"  [{i}] [score={score:.4f}] {text_preview}")
        except Exception as e:
            print(f"  ✗ 检索失败: {e}")

    print(f"\n集合名称: {collection_name}")
    print(f"向量数量: {collection.count()}")


def test_similarity():
    """测试相似度计算"""
    print("\n" + "=" * 60)
    print("测试 4: 文本相似度检索")
    print("=" * 60)

    embed_model = OpenAIEmbedding(
        model_name="BAAI/bge-m3",
        api_key="not-needed",
        api_base="http://192.168.8.233:8092/v1",
        embed_batch_size=32,
        timeout=300,
    )

    query = "什么是珐琅？"
    documents = [
        "珐琅是一种装饰工艺，通过在金属表面熔融玻璃粉末来制作彩色装饰。",
        "珐琅工艺历史悠久，起源于古埃及，后来传入中国。",
        "釉料是珐琅的主要成分，包括玻璃粉末和金属氧化物。",
        "烧制温度对珐琅的质量有很大影响。",
        "透明釉料可以让金属底色透出来。",
        "今天天气很好，适合户外活动。",
        "Python 是一种编程语言。",
    ]

    query_emb = np.array(embed_model.get_text_embedding(query))
    doc_embs = np.array(embed_model.get_text_embedding_batch(documents))

    def cosine_similarity(v1, v2):
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

    scores = [(i, doc, cosine_similarity(query_emb, doc_emb)) for i, doc_emb, doc in zip(range(len(documents)), doc_embs, documents)]
    scores.sort(key=lambda x: x[2], reverse=True)

    print(f"\n查询: {query}")
    print(f"最相关文档 (Top 5):")
    for i, (idx, doc, score) in enumerate(scores[:5], 1):
        print(f"  {i}. [{score:.4f}] {doc}")


def main():
    print("\n" + "=" * 60)
    print("LlamaIndex + vLLM Embedding 测试 (OpenAIEmbedding)")
    print("服务地址: http://192.168.8.233:8092")
    print("=" * 60)

    # test_basic_embedding()
    # test_long_text()
    test_real_markdown()
    # test_similarity()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
