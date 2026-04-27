#!/bin/bash
# 批量测试 stream_client 脚本

PYTHON="/home/zj/miniconda3/envs/morphe/bin/python"
HOST="http://192.168.8.233:8100"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 问题列表 - 手动添加你想要测试的问题
QUERIES=(
    "我想杀死你"
    "你好"
    "什么是失蜡铸造"
    "已知数列{an}的前n项和Sn=1/4*n²+2/3*n+3，求通项公式an"
    "求不等式x"
)

echo "========================================"
echo "开始批量测试 Stream Client"
echo "Host: $HOST"
echo "问题数量: ${#QUERIES[@]}"
echo "========================================"
echo ""

for i in "${!QUERIES[@]}"; do
    query="${QUERIES[$i]}"
    echo "----------------------------------------"
    echo "[$((i+1))/${#QUERIES[@]}] Query: $query"
    echo "----------------------------------------"

    $PYTHON "$SCRIPT_DIR/ai-service/scripts/stream_client.py" --host "$HOST" --query "$query"

    echo ""
    sleep 1  # 避免请求过快
done

echo "========================================"
echo "测试完成！共处理 ${#QUERIES[@]} 个问题"
echo "========================================"