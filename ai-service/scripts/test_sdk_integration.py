#!/home/zj/miniconda3/envs/morphe/bin/python
"""
llama-rag-sdk 集成测试脚本

测试 SDK 集成后的基本功能：
1. 文档上传 + 索引（使用 RAGSystem）
2. 召回测试
3. RAG 回复测试

使用方法:
    conda activate morphe
    python scripts/test_sdk_integration.py
"""
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "llama-rag-sdk"))

from app.core.logging import get_logger
from app.services.rag_service import rag_retrieval
from app.core.config import settings
from app.core.database import mongodb

# 直接使用 RAGSystem
from llama_rag_sdk.rag_system import RAGSystem

logger = get_logger(__name__)


async def setup_database(max_retries: int = 3):
    """初始化数据库连接，失败时重试"""
    for attempt in range(1, max_retries + 1):
        try:
            await mongodb.connect()
            print("✅ 数据库连接成功")
            return True
        except Exception as e:
            if attempt < max_retries:
                print(f"⚠️ 数据库连接失败 (尝试 {attempt}/{max_retries}): {e}")
                print("   Retrying in 2 seconds...")
                await asyncio.sleep(2)
            else:
                print(f"❌ 数据库连接失败 (已重试 {max_retries} 次): {e}")
                return False


async def teardown_database():
    """关闭数据库连接"""
    try:
        await mongodb.disconnect()
        print("✅ 数据库连接已关闭")
    except Exception as e:
        print(f"⚠️ 数据库关闭时出错: {e}")


async def test_basic_import():
    """测试 1: 基本导入测试"""
    print("\n=== 测试 1: 基本导入 ===")
    try:
        from app.services.rag_service import rag_retrieval
        from llama_rag_sdk.rag_system import RAGSystem
        print("✅ 导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


async def test_document_index(file_path: str, kb_id: str, doc_id: str):
    """测试 2: 文档索引（使用 RAGSystem）"""
    print(f"\n=== 测试 2: 文档索引 ===")
    print(f"文件: {file_path}")
    print(f"KB ID: {kb_id}")

    try:
        # 使用 RAGSystem 直接索引文档
        async with RAGSystem(
            collection_name="rag_documents",
            enable_summarization=False,  # 测试时关闭摘要生成
        ) as rag:
            # RAGSystem 使用 MinerU 解析 PDF + 结构感知分块
            result_doc_ids = await rag.index_document(
                file_path,
                metadata={
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                }
            )

            print(f"✅ 文档索引成功: {len(result_doc_ids)} 个 chunk")
            return True
    except Exception as e:
        print(f"❌ 文档索引失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_retrieve(query: str, kb_id: str):
    """测试 3: 召回测试"""
    print(f"\n=== 测试 3: 召回测试 ===")
    print(f"查询: {query}")
    print(f"KB ID: {kb_id}")

    try:
        results = await rag_retrieval.search(
            query=query,
            kb_ids=[kb_id],
            top_k=3,
        )

        print(f"✅ 召回成功，结果数: {len(results)}")
        for i, doc in enumerate(results):
            print(f"  [{i+1}] score={doc['score']:.4f}, content={doc['content'][:100]}...")

        return True
    except Exception as e:
        print(f"❌ 召回失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_faq_search(query: str, employee_id: str):
    """测试 4: FAQ 搜索"""
    print(f"\n=== 测试 4: FAQ 搜索 ===")
    print(f"查询: {query}")
    print(f"Employee ID: {employee_id}")

    try:
        results = await rag_retrieval.faq_search(
            query=query,
            employee_id=employee_id,
            faq_top_k=3,
        )

        print(f"✅ FAQ 搜索成功，结果数: {len(results)}")
        for i, faq in enumerate(results):
            print(f"  [{i+1}] score={faq['score']:.4f}, faq_id={faq.get('faq_id')}, question={faq.get('question_name', 'N/A')[:50]}...")

        return True
    except Exception as e:
        print(f"❌ FAQ 搜索失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试流程"""
    import argparse

    parser = argparse.ArgumentParser(description="SDK 集成测试")
    parser.add_argument("--test", choices=["import", "index", "retrieve", "faq", "all"], default="all",
                        help="测试类型")
    parser.add_argument("--file", type=str, help="测试文件路径")
    parser.add_argument("--kb-id", type=str, default="kb_test", help="知识库 ID")
    parser.add_argument("--doc-id", type=str, default="doc_test_001", help="文档 ID")
    parser.add_argument("--query", type=str, default="测试查询", help="查询内容")
    parser.add_argument("--employee-id", type=str, default="hutao", help="员工 ID")

    args = parser.parse_args()

    print("=" * 60)
    print("llama-rag-sdk 集成测试")
    print("=" * 60)

    # 初始化数据库连接
    if not await setup_database():
        print("\n❌ 数据库连接失败，退出测试")
        return False

    try:
        results = {}

        if args.test in ["import", "all"]:
            results["import"] = await test_basic_import()

        if args.test in ["index", "all"] and args.file:
            results["index"] = await test_document_index(args.file, args.kb_id, args.doc_id)

        if args.test in ["retrieve", "all"]:
            results["retrieve"] = await test_retrieve(args.query, args.kb_id)

        if args.test in ["faq", "all"]:
            results["faq"] = await test_faq_search(args.query, args.employee_id)

        # 总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        for name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{name}: {status}")

        all_passed = all(results.values())
        print(f"\n总体结果: {'✅ 全部通过' if all_passed else '❌ 部分失败'}")

        return all_passed
    finally:
        # 清理数据库连接
        await teardown_database()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
