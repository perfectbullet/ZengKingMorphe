#!/bin/bash
# 批量测试 stream_client 脚本

PYTHON="python"
HOST="http://192.168.8.233:8100"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 问题列表 - 手动添加你想要测试的问题
QUERIES=(
    "我想杀死你"
    "负三的值"
    "什么是失蜡铸造"？
    "What's the date today?"
    "上下这条裤子都不可能去。"
    "12345678。你好，你好，请帮我检测一下。"
    # "他人因某些原因无法主动通过邮件或消息与您联系"
    # "已知数列{an}的前n项和Sn=1/4*n²+2/3*n+3，求通项公式an"
    "求解不等式x的平方减去5x加上6小于0的解"

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