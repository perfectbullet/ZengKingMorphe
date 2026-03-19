# 下载模型（首次自动）
# docker compose up -d

# 查看日志
# docker compose logs -f qwen25-math-vllm

# 健康检查
curl http://localhost:8095/health
# 返回: {"status":"ok"}



curl http://localhost:8095/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL_ID}"'",
    "messages": [
      {"role": "system", "content": "Please reason step by step, and put your final answer within boxed{}."},
      {"role": "user", "content": "解方程: 2x + 5 = 15"}
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 2048
  }'




  # 启用Python解释器的system prompt
curl http://localhost:8095/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL_ID}"'",
    "messages": [
      {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within boxed{}."},
      {"role": "user", "content": "计算矩阵 [[1,2],[3,4]] 的特征值，用Python验证"}
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 2048,
    "tools": [{"type": "python"}]
  }'