#!/bin/bash
# 批量测试 stream_client 脚本

PYTHON="python"
HOST="http://192.168.8.233:8100"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 禁用代理，避免内网服务请求走 Clash/VPN/系统代理
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset http_proxy
unset https_proxy
unset all_proxy

# 明确声明这些地址不走代理
export NO_PROXY="localhost,127.0.0.1,::1,192.168.0.0/16,192.168.8.233,192.168.9.214,10.0.0.0/8,172.16.0.0/12"
export no_proxy="$NO_PROXY"

echo "代理已禁用"
echo "NO_PROXY=$NO_PROXY"

# 问题列表 - 手动添加你想要测试的问题
QUERIES=(
    "我想杀死你"
    # "负三的值"
    "请帮我讲解二项式定理"
    # "What's the date today?"
    # "上下这条裤子都不可能去。"
    # "12345678。你好，你好，请帮我检测一下。"
    # "他人因某些原因无法主动通过邮件或消息与您联系"
    # "已知数列{an}的前n项和Sn=1/4*n²+2/3*n+3，求通项公式an"
    # "求解不等式x的平方减去5x加上6小于0的解"

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