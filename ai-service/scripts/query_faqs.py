"""
查询指定员工的 FAQ 数据（MongoDB + ChromaDB + ElasticSearch）
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import mongodb
# from app.core.chroma import chroma_db
from app.core.elasticsearch import es_db
from app.core.logging import get_logger

logger = get_logger(__name__)


async def query_faqs_by_employee(employee_id: str):
    """查询指定员工的所有 FAQ 数据"""
    print(f"\n{'='*80}")
    print(f"查询员工 FAQ 数据: employee_id={employee_id}")
    print(f"{'='*80}\n")
    
    try:
        # 连接数据库
        await mongodb.connect()
        chroma_db.connect()
        await es_db.connect()
        
        db = mongodb.get_db()
        
        # 1. 查询 MongoDB
        print("📦 MongoDB FAQ 数据:")
        print("-" * 80)
        
        faqs_cursor = db.faqs.find({"employee_id": employee_id})
        faqs = await faqs_cursor.to_list(length=None)
        
        if not faqs:
            print(f"❌ 未找到员工 {employee_id} 的 FAQ 数据")
            print(f"   提示: 请先创建会话以触发 FAQ 向量化")
            return
        
        enabled_count = sum(1 for faq in faqs if faq.get("is_enable") == 1)
        disabled_count = len(faqs) - enabled_count
        vectorized = sum(1 for faq in faqs if faq.get("vector_id"))
        es_indexed = sum(1 for faq in faqs if faq.get("es_indexed"))
        
        print(f"✅ 找到 {len(faqs)} 条 FAQ 记录\n")
        print(f"📊 统计信息:")
        print(f"  - 总数: {len(faqs)}")
        print(f"  - 启用: {enabled_count}")
        print(f"  - 禁用: {disabled_count}")
        print(f"  - 向量化: {vectorized}")
        print(f"  - ES索引: {es_indexed}")
        print()
        
        # 显示前 5 条详情
        print("📋 FAQ 详情（前5条）:")
        print("-" * 80)
        for i, faq in enumerate(faqs[:5], 1):
            print(f"\n{i}. FAQ ID: {faq['faq_id']}")
            print(f"   问题: {faq.get('question_name', 'N/A')}")
            print(f"   相似问题: {len(faq.get('similar_questions', []))} 个")
            print(f"   答案: {len(faq.get('answers', []))} 个")
            print(f"   状态: {'✅ 启用' if faq.get('is_enable') == 1 else '❌ 禁用'}")
            print(f"   向量ID: {faq.get('vector_id', 'N/A')}")
            print(f"   ES索引: {'✅ 是' if faq.get('es_indexed') else '❌ 否'}")
        
        if len(faqs) > 5:
            print(f"\n   ... 还有 {len(faqs) - 5} 条 FAQ")
        
        # 2. 查询 ChromaDB
        print(f"\n{'='*80}")
        print("🔍 ChromaDB 向量数据:")
        print("-" * 80)
        
        try:
            chroma_collection = chroma_db.get_collection("faqs")
            collection_count = chroma_collection.count()
            print(f"✅ ChromaDB 'faqs' 集合总数: {collection_count} 条向量")
            
            if enabled_count > 0:
                results = chroma_collection.get(
                    where={"employee_id": employee_id},
                    limit=5,
                    include=["metadatas", "documents"]
                )
                
                print(f"✅ 员工 {employee_id} 的向量: {len(results['ids'])} 条")
                
                if results['ids']:
                    print(f"\n向量样本（前5条）:")
                    for i, (vec_id, metadata, doc) in enumerate(zip(
                        results['ids'], 
                        results['metadatas'], 
                        results['documents']
                    ), 1):
                        print(f"\n{i}. Vector ID: {vec_id}")
                        print(f"   问题: {metadata.get('question_name', 'N/A')}")
                        print(f"   文本长度: {len(doc)} 字符")
                        print(f"   文本预览: {doc[:80]}...")
        except Exception as e:
            print(f"❌ ChromaDB 查询失败: {e}")
        
        # 3. 查询 ElasticSearch
        print(f"\n{'='*80}")
        print("🔎 ElasticSearch 索引数据:")
        print("-" * 80)
        
        try:
            es_response = await es_db.client.search(
                index="digital_employee_faqs",
                body={
                    "query": {"term": {"employee_id": employee_id}},
                    "size": 5
                }
            )
            
            total_hits = es_response['hits']['total']['value']
            print(f"✅ ElasticSearch 索引数: {total_hits} 条")
            
            if es_response['hits']['hits']:
                print(f"\n索引样本（前5条）:")
                for i, hit in enumerate(es_response['hits']['hits'], 1):
                    source = hit['_source']
                    print(f"\n{i}. Document ID: {hit['_id']}")
                    print(f"   问题: {source.get('question_name', 'N/A')}")
                    print(f"   相似问题: {len(source.get('similar_questions', []))} 个")
        except Exception as e:
            print(f"❌ ElasticSearch 查询失败: {e}")
        
        print(f"\n{'='*80}")
        print("✅ 查询完成！")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"查询 FAQ 失败: {e}", exc_info=True)
        print(f"\n❌ 查询失败: {e}")
    
    finally:
        await mongodb.disconnect()
        await es_db.disconnect()


async def list_all_employees():
    """列出所有有 FAQ 数据的员工"""
    print(f"\n{'='*80}")
    print("查询所有员工列表")
    print(f"{'='*80}\n")
    
    try:
        await mongodb.connect()
        db = mongodb.get_db()
        
        # 获取所有唯一的 employee_id
        employee_ids = await db.faqs.distinct("employee_id")
        
        if not employee_ids:
            print("❌ 数据库中没有任何 FAQ 数据")
            print("   提示: 请先创建会话以触发 FAQ 向量化\n")
            return
        
        print(f"✅ 找到 {len(employee_ids)} 个员工有 FAQ 数据:\n")
        
        for employee_id in employee_ids:
            count = await db.faqs.count_documents({"employee_id": employee_id})
            enabled_count = await db.faqs.count_documents({
                "employee_id": employee_id, 
                "is_enable": 1
            })
            
            # 获取员工名称
            employee_config = await db.digital_employee_configs.find_one(
                {"employee_id": employee_id}
            )
            employee_name = employee_config.get("name", "未知") if employee_config else "未知"
            
            print(f"  - {employee_id} ({employee_name}): {count} 条 FAQ (启用: {enabled_count})")
        
        print()
        
    except Exception as e:
        logger.error(f"查询员工列表失败: {e}", exc_info=True)
        print(f"\n❌ 查询失败: {e}")
    
    finally:
        await mongodb.disconnect()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python scripts/query_faqs.py <employee_id>  # 查询指定员工的 FAQ")
        print("  python scripts/query_faqs.py --list         # 列出所有员工\n")
        print("示例:")
        print("  python scripts/query_faqs.py hutao")
        print("  python scripts/query_faqs.py --list\n")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "--list":
        asyncio.run(list_all_employees())
    else:
        employee_id = command
        asyncio.run(query_faqs_by_employee(employee_id))
