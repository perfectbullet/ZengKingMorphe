"""
批量分类 queries.txt 中的 query，判断是「真实问题」还是「噪声」。

用法:
    cd ai_agents/agents/examples/voice-assistant/tenapp/ten_packages/extension/main_python/agent/
    python classify_queries.py

环境变量:
    OPENAI_API_KEY  - OpenAI API Key（必需）
    OPENAI_MODEL    - 模型名称（默认 gpt-4o-mini）
    OPENAI_BASE_URL - API Base URL（可选，默认 https://api.openai.com/v1）

输出:
    classification_results.json - 同目录下，包含逐条分类结果和统计汇总
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

BATCH_SIZE = 20

SYSTEM_PROMPT = """你是一个语音助手 query 分类器。你的任务是对用户输入进行分类。

分类标准：

**real_question（真实问题）**：用户有明确意图获取信息、求解问题或请求讲解的内容。包括但不限于：
- 知识问答（如"什么是等差数列？"、"什么是失蜡铸造"）
- 数学求解（如"求不等式 x²-5x+6<0 的解"、"二项式定理的公式是什么？"）
- 请求讲解（如"帮我讲解一下集合的概念"）
- 天气查询（如"北京今天天气怎么样？"）
- 常识问题（如"一加一等于几？"、"中国的十大开国元帅有哪些人？"）
- 英语问题（如英文提问的语法、知识类问题）
- 有明确学科或知识领域的完整提问

**noise（噪声）**：不构成有效提问的内容。包括但不限于：
- 旁人对话（如"对，就是那个，或者边的那俩"、"你看那个歪点那个活跃时间"）
- ASR 碎片/不完整句子（如"函数，我们20"、"等一下公式"）
- 重复填充词（如"我知道了"重复多遍、"你你你你你"重复多遍）
- 跟别人说话（如"打断一下，我拿一下"、"等开会的时候再说吧"）
- 设备/技术调试对话（如"唤醒后几秒后提问才有效"、"这个还是得去求一下"）
- 纯感叹或情绪表达（如"真的假的?"、"你是不是有病啊?"）
- 无明确问题意图的短语（如"查看一下"、"麻烦一下"、"挑战一下"）

请对输入的每条 query 进行分类，返回 JSON 对象。每项格式：
{"label": "real_question" 或 "noise", "reason": "简短理由(不超过15字)"}

只返回 JSON 对象，不要返回其他内容。"""


def parse_queries(filepath: Path) -> list[dict]:
    """解析 queries.txt，返回 [{index, query}, ...]"""
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: 编号\tquery
            parts = line.split("\t", 1)
            if len(parts) == 2:
                queries.append(
                    {
                        "index": int(parts[0]),
                        "query": parts[1],
                    }
                )
            else:
                # 没有 tab 分隔，整行作为 query
                queries.append(
                    {
                        "index": len(queries) + 1,
                        "query": line,
                    }
                )
    return queries


def classify_batch(client: OpenAI, model: str, batch: list[dict]) -> list[dict]:
    """对一批 query 调用 OpenAI 进行分类"""
    # 构造用户消息：列出本批次的所有 query
    query_list = ""
    for q in batch:
        # 截断过长的 query，避免 token 过多
        text = q["query"][:200]
        query_list += f"{q['index']}. {text}\n"

    user_message = f"请对以下 query 逐条分类：\n\n{query_list}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    content = response.choices[0].message.content.strip()

    # 提取 JSON（可能被 markdown 代码块包裹）
    if content.startswith("```"):
        # 去掉 ```json 和 ```
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])

    results = json.loads(content)
    return results


def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 解析 query 文件
    script_dir = Path(__file__).parent
    queries_file = script_dir / "queries.txt"
    if not queries_file.exists():
        print(f"错误: 找不到 {queries_file}")
        sys.exit(1)

    queries = parse_queries(queries_file)
    print(f"共读取 {len(queries)} 条 query，开始分类（模型: {model}）...")

    # 分批处理
    all_results = {}
    total_batches = (len(queries) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(queries), BATCH_SIZE):
        batch = queries[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(
            f"  批次 {batch_num}/{total_batches} (query {batch[0]['index']}-{batch[-1]['index']})...",
            end=" ",
            flush=True,
        )

        try:
            results = classify_batch(client, model, batch)
            for r in results:
                idx = r["index"]
                all_results[idx] = {
                    "index": idx,
                    "query": next(
                        (q["query"] for q in queries if q["index"] == idx),
                        "",
                    ),
                    "label": r["label"],
                    "reason": r.get("reason", ""),
                }
            print(f"完成 ({len(results)} 条)")
        except Exception as e:
            print(f"失败: {e}")
            # 标记为 unknown
            for q in batch:
                all_results[q["index"]] = {
                    "index": q["index"],
                    "query": q["query"],
                    "label": "unknown",
                    "reason": f"API 调用失败: {e}",
                }

        # 批次间延迟，避免限流
        if i + BATCH_SIZE < len(queries):
            time.sleep(1)

    # 按编号排序
    sorted_results = sorted(all_results.values(), key=lambda x: x["index"])

    # 统计
    real_count = sum(1 for r in sorted_results if r["label"] == "real_question")
    noise_count = sum(1 for r in sorted_results if r["label"] == "noise")
    unknown_count = sum(1 for r in sorted_results if r["label"] == "unknown")
    total = len(sorted_results)

    print("\n分类完成！")
    print(f"  总计: {total}")
    print(f"  真实问题: {real_count} ({real_count / total * 100:.1f}%)")
    print(f"  噪声: {noise_count} ({noise_count / total * 100:.1f}%)")
    if unknown_count:
        print(f"  未知(失败): {unknown_count}")

    # 写入结果
    output = {
        "summary": {
            "total": total,
            "real_question": real_count,
            "noise": noise_count,
            "unknown": unknown_count,
            "model": model,
        },
        "results": sorted_results,
    }

    output_file = script_dir / "classification_resultsv2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
