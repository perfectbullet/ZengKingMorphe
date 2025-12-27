"""
FAQ集成测试脚本

测试外部API集成、FAQ向量化、多路召回和对话流程。
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.database import mongodb, get_database
from app.core.elasticsearch import es_db
from app.core.chroma import chroma_db
from app.core.logging import get_logger

logger = get_logger(__name__)


async def test_faq_data_in_mongodb():
    """测试MongoDB中的FAQ数据"""
    print("\n=== 测试1: 检查MongoDB中的FAQ数据 ===")
    
    db = await get_database()
    
    # 查询  digital_employee_configs
    digital_configs = await db.digital_employee_configs.find().to_list(length=10)
    print(f"✓ 找到 {len(digital_configs)} 个数字员工配置")
    
    if digital_configs:
        config = digital_configs[0]
        print(f"  - employee_id: {config['employee_id']}")
        print(f"  - name: {config['name']}")
        print(f"  - faq_count: {config.get('faq_count', 0)}")
        print(f"  - faq_sim_threshold: {config.get('faq_sim_threshold', 0)}")
        print(f"  - faq_top_k: {config.get('faq_top_k', 1)}")
    
    # 查询FAQs
    faqs = await db.faqs.find().to_list(length=10)
    print(f"✓ 找到 {len(faqs)} 个FAQ记录")
    
    if faqs:
        faq = faqs[0]
        print(f"  - faq_id: {faq['faq_id']}")
        print(f"  - question_name: {faq['question_name'][:50]}...")
        print(f"  - similar_questions count: {len(faq.get('similar_questions', []))}")
        print(f"  - answers count: {len(faq.get('answers', []))}")
        print(f"  - is_enable: {faq.get('is_enable')}")
        print(f"  - vector_id: {faq.get('vector_id')}")
        print(f"  - es_indexed: {faq.get('es_indexed')}")


async def test_faq_hybrid_search():
    """测试FAQ混合搜索"""
    print("\n=== 测试2: FAQ混合搜索 ===")
    
    from app.services.rag_service import rag_retrieval
    
    # 获取一个employee_id
    db = await get_database()
    config = await db.digital_employee_configs.find_one()
    
    if not config:
        print("❌ 没有找到数字员工配置，跳过测试")
        return
    
    employee_id = config['employee_id']
    print(f"使用employee_id: {employee_id}")
    
    # 测试查询
    test_queries = [
        "如何申请退款？",
        "产品保修",
        "联系客服"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = await rag_retrieval.faq_hybrid_search(
            query=query,
            employee_id=employee_id,
            faq_sim_threshold=0.0,
            faq_top_k=3
        )
        
        print(f"✓ 返回 {len(results)} 个结果")
        for i, result in enumerate(results, 1):
            print(f"  {i}. faq_id: {result['faq_id']}")
            print(f"     question: {result['question_name'][:50]}...")
            print(f"     rrf_score: {result['rrf_score']:.4f}")
            print(f"     vector_score: {result.get('vector_score', 0):.4f}")
            print(f"     keyword_score: {result.get('keyword_score', 0):.4f}")


async def test_conversation_workflow():
    """测试对话工作流中的FAQ匹配"""
    print("\n=== 测试3: 对话工作流FAQ匹配 ===")
    
    from app.services.conversation_service import conversation_workflow
    
    # 获取一个employee_id
    db = await get_database()
    config = await db.digital_employee_configs.find_one()
    
    if not config:
        print("❌ 没有找到数字员工配置，跳过测试")
        return
    
    employee_id = config['employee_id']
    
    # 模拟对话状态
    initial_state = {
        "messages": [],
        "user_query": "如何申请退款？",
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "employee_id": employee_id,
        "employee_config": {},
        "is_realtime_query": False,
        "realtime_category": "",
        "realtime_detect_reason": "",
        "intent": "",
        "entities": {},
        "retrieved_docs": [],
        "relevance_score": 0.0,
        "web_search_results": [],
        "final_answer": "",
        "confidence": 0.0,
        "context": {},
        "has_sensitive": False,
        "error": None,
        "faq_matched": None,
        "kb_used": [],
        "web_search_used": False,
        "conversation_id": "test_conv_789",
        "response_time_ms": 0
    }
    
    try:
        result = await conversation_workflow.ainvoke(initial_state)
        
        print(f"✓ 工作流执行完成")
        print(f"  - FAQ匹配: {bool(result.get('faq_matched'))}")
        
        if result.get('faq_matched'):
            faq_info = result['faq_matched']
            print(f"  - faq_id: {faq_info.get('faq_id')}")
            print(f"  - question_name: {faq_info.get('question_name', '')[:50]}...")
            print(f"  - rrf_score: {faq_info.get('rrf_score', 0):.4f}")
            print(f"  - selected_answer长度: {len(faq_info.get('selected_answer', ''))}")
        
        print(f"  - final_answer长度: {len(result.get('final_answer', ''))}")
        print(f"  - confidence: {result.get('confidence', 0):.4f}")
        print(f"  - intent: {result.get('intent', 'unknown')}")
        
    except Exception as e:
        print(f"❌ 工作流执行失败: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主测试函数"""
    print("=" * 60)
    print("FAQ集成功能测试")
    print("=" * 60)
    
    try:
        # 连接数据库
        print("\n正在连接数据库...")
        await mongodb.connect()
        await es_db.connect()
        chroma_db.connect()
        print("✓ 数据库连接成功\n")
        
        # 测试1: 检查MongoDB数据
        await test_faq_data_in_mongodb()
        
        # 测试2: FAQ混合搜索
        await test_faq_hybrid_search()
        
        # 测试3: 对话工作流
        await test_conversation_workflow()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭数据库连接
        print("\n正在关闭数据库连接...")
        await mongodb.disconnect()
        await es_db.disconnect()
        chroma_db.disconnect()
        print("✓ 数据库连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
