"""
测试人工概念检索服务。

测试场景：
1. 精确匹配 concept_name
2. alias 命中
3. LightRAG 召回
4. 未命中情况
5. 配置缺失情况
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置测试环境变量
os.environ["CONCEPT_RETRIEVAL_ENABLED"] = "true"
os.environ["CONCEPT_RETRIEVAL_DOMAIN"] = "math"
os.environ["CONCEPT_RETRIEVAL_ENTITY_TYPE"] = "MANUAL_CONCEPT"

from app.services.concept_retrieval_service import (
    ConceptRetrievalService,
    ConceptContext,
    get_concept_retrieval_service,
)


def test_exact_match_concept_name():
    """测试精确匹配 concept_name。"""
    print("\n=== Test 1: 精确匹配 concept_name ===")

    # 创建临时配置文件
    config_content = [
        {
            "concept_name": "二项式定理",
            "content": "二项式定理是代数学中的一个重要定理...",
            "md_content": "# 二项式定理\n\n二项式定理是代数学中的一个重要定理...",
            "review_status": "correct",
            "domain": "math",
            "doc_id": "test_001",
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        import json
        for item in config_content:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        config_path = f.name

    try:
        os.environ["CONCEPT_RETRIEVAL_CONFIG"] = config_path

        # 重置服务单例
        import app.services.concept_retrieval_service as crs
        crs._concept_retrieval_service = None

        service = get_concept_retrieval_service()

        async def test():
            result = await service.aretrieve("请帮我讲解二项式定理")

            assert result.hit == True, "应该命中概念"
            assert result.concept_name == "二项式定理", "概念名称应该是二项式定理"
            assert result.hit_reason == "query_contains_concept_name", "命中原因应该是精确匹配"
            assert result.confidence == 1.0, "精确匹配置信度应该是1.0"
            assert result.content is not None, "应该有内容"

            print(f"✓ 命中成功: concept_name={result.concept_name}, reason={result.hit_reason}")

        asyncio.run(test())

    finally:
        # 清理临时文件
        os.unlink(config_path)


def test_alias_match():
    """测试 alias 命中。"""
    print("\n=== Test 2: alias 命中 ===")

    config_content = [
        {
            "concept_name": "二项式定理",
            "aliases": ["牛顿二项式公式", "二项式"],
            "md_content": "# 二项式定理\n\n二项式定理是代数学中的一个重要定理...",
            "review_status": "correct",
            "doc_id": "test_002",
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        import json
        for item in config_content:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        config_path = f.name

    try:
        os.environ["CONCEPT_RETRIEVAL_CONFIG"] = config_path

        # 重置服务单例
        import app.services.concept_retrieval_service as crs
        crs._concept_retrieval_service = None

        service = get_concept_retrieval_service()

        async def test():
            result = await service.aretrieve("解释一下牛顿二项式公式")

            assert result.hit == True, "应该命中概念"
            assert result.concept_name == "二项式定理", "概念名称应该是二项式定理"
            assert result.hit_reason == "query_contains_alias", "命中原因应该是alias匹配"
            assert result.confidence == 1.0, "alias匹配置信度应该是1.0"

            print(f"✓ alias命中成功: concept_name={result.concept_name}, matched_alias={result.metadata.get('matched_alias')}")

        asyncio.run(test())

    finally:
        os.unlink(config_path)


def test_miss():
    """测试未命中情况。"""
    print("\n=== Test 3: 未命中 ===")

    config_content = [
        {
            "concept_name": "二项式定理",
            "md_content": "# 二项式定理\n\n二项式定理是代数学中的一个重要定理...",
            "review_status": "correct",
            "doc_id": "test_003",
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        import json
        for item in config_content:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        config_path = f.name

    try:
        os.environ["CONCEPT_RETRIEVAL_CONFIG"] = config_path
        os.environ["CONCEPT_RETRIEVAL_ENABLE_LIGHTRAG"] = "false"  # 禁用 LightRAG

        # 重置服务单例
        import app.services.concept_retrieval_service as crs
        crs._concept_retrieval_service = None

        service = get_concept_retrieval_service()

        async def test():
            result = await service.aretrieve("请帮我讲解分类加法计数原理")

            assert result.hit == False, "不应该命中概念"
            assert result.hit_reason == "not_found", "未命中原因应该是not_found"

            print(f"✓ 未命中测试通过: reason={result.hit_reason}")

        asyncio.run(test())

    finally:
        os.unlink(config_path)


def test_disabled():
    """测试功能禁用情况。"""
    print("\n=== Test 4: 功能禁用 ===")

    os.environ["CONCEPT_RETRIEVAL_ENABLED"] = "false"

    # 重置服务单例
    import app.services.concept_retrieval_service as crs
    crs._concept_retrieval_service = None

    service = get_concept_retrieval_service()

    async def test():
        result = await service.aretrieve("请帮我讲解二项式定理")

        assert result.hit == False, "功能禁用时不应该命中"
        assert result.hit_reason == "disabled", "未命中原因应该是disabled"

        print(f"✓ 禁用测试通过: reason={result.hit_reason}")

    asyncio.run(test()


def test_empty_query():
    """测试空查询情况。"""
    print("\n=== Test 5: 空查询 ===")

    os.environ["CONCEPT_RETRIEVAL_ENABLED"] = "true"

    # 重置服务单例
    import app.services.concept_retrieval_service as crs
    crs._concept_retrieval_service = None

    service = get_concept_retrieval_service()

    async def test():
        result = await service.aretrieve("")

        assert result.hit == False, "空查询不应该命中"
        assert result.hit_reason == "empty_query", "未命中原因应该是empty_query"

        print(f"✓ 空查询测试通过: reason={result.hit_reason}")

    asyncio.run(test()


def run_all_tests():
    """运行所有测试。"""
    print("开始测试人工概念检索服务...")

    try:
        test_exact_match_concept_name()
        test_alias_match()
        test_miss()
        test_disabled()
        test_empty_query()

        print("\n✅ 所有测试通过！")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 测试错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())