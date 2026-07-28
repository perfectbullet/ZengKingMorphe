#!/bin/bash
# 人工概念检索集成验证脚本
# ====================================

echo "开始验证人工概念检索集成..."

# 1. 语法检查
echo "1. 检查 Python 语法..."
python -m py_compile \
  app/services/concept_retrieval_service.py \
  app/services/conversation/conversation_state.py \
  app/services/conversation/conversation_nodes.py \
  app/services/conversation/conversation_helpers.py \
  app/services/conversation/intent_routing.py \
  app/services/conversation_service.py

if [ $? -eq 0 ]; then
    echo "✅ 语法检查通过"
else
    echo "❌ 语法检查失败"
    exit 1
fi

# 2. 检查关键字段是否正确添加
echo "2. 检查关键字段和路由..."

# 检查 conversation_state.py 中的新字段
if grep -q "concept_retrieval_enabled" app/services/conversation/conversation_state.py; then
    echo "✅ concept_retrieval_enabled 字段已添加"
else
    echo "❌ concept_retrieval_enabled 字段缺失"
    exit 1
fi

# 检查 intent_routing.py 中的新路由常量
if grep -q "ROUTE_BRANCH_CONCEPT_HIT" app/services/conversation/intent_routing.py; then
    echo "✅ ROUTE_BRANCH_CONCEPT_HIT 常量已添加"
else
    echo "❌ ROUTE_BRANCH_CONCEPT_HIT 常量缺失"
    exit 1
fi

# 检查 conversation_nodes.py 中的 concept_retrieval 节点
if grep -q "async def concept_retrieval" app/services/conversation/conversation_nodes.py; then
    echo "✅ concept_retrieval 节点已添加"
else
    echo "❌ concept_retrieval 节点缺失"
    exit 1
fi

# 检查 conversation_service.py 中的节点注册
if grep -q '"concept_retrieval"' app/services/conversation_service.py; then
    echo "✅ concept_retrieval 节点已注册"
else
    echo "❌ concept_retrieval 节点未注册"
    exit 1
fi

# 3. 检查没有使用错误的工作目录
echo "3. 检查工作目录配置..."
if grep -r "lightrag_manual_concepts_custom_kg" app/services/; then
    echo "❌ 发现错误的工作目录引用: lightrag_manual_concepts_custom_kg"
    exit 1
else
    echo "✅ 工作目录配置正确"
fi

# 4. 检查环境变量引用
echo "4. 检查环境变量..."
if grep -q "CONCEPT_RETRIEVAL_ENABLED" app/services/conversation/conversation_nodes.py; then
    echo "✅ 环境变量检查正确"
else
    echo "❌ 环境变量检查缺失"
    exit 1
fi

# 5. 验证图结构路由
echo "5. 验证图结构路由..."
if grep -q "ROUTE_BRANCH_RAG.*concept_retrieval" app/services/conversation_service.py; then
    echo "✅ RAG 路由正确指向 concept_retrieval"
else
    echo "❌ RAG 路由配置错误"
    exit 1
fi

if grep -q "route_after_concept_retrieval" app/services/conversation_service.py; then
    echo "✅ concept_retrieval 路由已配置"
else
    echo "❌ concept_retrieval 路由配置缺失"
    exit 1
fi

# 6. 检查 generate_answer 中的 concept_context 处理
echo "6. 检查 generate_answer 节点..."
if grep -q "concept_retrieval_hit.*concept_context" app/services/conversation/conversation_nodes.py; then
    echo "✅ generate_answer 中 concept_context 处理已添加"
else
    echo "❌ generate_answer 中 concept_context 处理缺失"
    exit 1
fi

# 7. 检查 build_generation_messages 中的 concept_context 处理
echo "7. 检查 build_generation_messages 函数..."
if grep -q "concept_retrieval_hit.*concept_context" app/services/conversation/conversation_helpers.py; then
    echo "✅ build_generation_messages 中 concept_context 处理已添加"
else
    echo "❌ build_generation_messages 中 concept_context 处理缺失"
    exit 1
fi

# 8. 检查 chat_stream_v1.py 中的 _STATE_DEFAULTS 同步
echo "8. 检查 _STATE_DEFAULTS 同步..."
concept_fields=("concept_retrieval_enabled" "concept_retrieval_hit" "concept_retrieval_reason" "concept_context" "concept_context_source")
missing_fields=()

for field in "${concept_fields[@]}"; do
    if ! grep -q "\"$field\"" app/api/endpoints/chat_stream_v1.py; then
        missing_fields+=("$field")
    fi
done

if [ ${#missing_fields[@]} -eq 0 ]; then
    echo "✅ _STATE_DEFAULTS 已同步所有新字段"
else
    echo "❌ _STATE_DEFAULTS 缺少字段: ${missing_fields[*]}"
    exit 1
fi

# 9. 最终总结
echo ""
echo "======================================="
echo "✅ 所有验证检查通过！"
echo "======================================="
echo ""
echo "配置建议："
echo "1. 设置 CONCEPT_RETRIEVAL_ENABLED=true"
echo "2. 配置 CONCEPT_RETRIEVAL_CONFIG 指向概念配置文件"
echo "3. 配置 CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR 指向 LightRAG 工作目录"
echo "4. 可选：配置 CONCEPT_RETRIEVAL_WHITELIST 指向白名单文件"
echo ""
echo "测试建议："
echo "- 启动服务后，询问问题：请帮我讲解二项式定理"
echo "- 检查日志中的 concept_retrieval_hit 状态"
echo "- 验证是否使用人工概念内容"
echo ""
echo "手动验证命令："
echo "export CONCEPT_RETRIEVAL_ENABLED=true"
echo "export CONCEPT_RETRIEVAL_DOMAIN=math"
echo "export CONCEPT_RETRIEVAL_CONFIG=/path/to/concepts.jsonl"
echo "export CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR=/path/to/lightrag_manual_concepts"
echo "cd ai-service && uvicorn main:app --reload --port 8000"
echo ""

exit 0
