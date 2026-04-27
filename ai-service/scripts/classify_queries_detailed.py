#!/usr/bin/env python3
"""
批量分类 user_queries.txt 中的 query，进行细粒度分类。

用法:
    cd ai-service
    conda activate morphe
    python scripts/classify_queries_detailed.py              # 处理全部
    python scripts/classify_queries_detailed.py -n 100       # 随机处理100条

环境变量:
    RAG_Anything_OPENAI_API_KEY  - OpenAI API Key（从 .env 读取）
    RAG_Anything_OPENAI_API_BASE - API Base URL（从 .env 读取）
    RAG_Anything_OPENAI_MODEL    - 模型名称（从 .env 读取）

输出:
    - classification_results.json  - 逐条分类结果
    - classification_results-n-100.json - 随机抽样时的结果
    - classification_summary.txt   - 统计汇总和各分类示例
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

OUTPUT_DIR = Path(__file__).parent

# ============================================================================
# 分类提示词（单条分类版本）
# ============================================================================
SYSTEM_PROMPT = """你是一个用户查询分类专家。你的任务是对用户输入进行细粒度分类。

## 分类类别及定义

### 1. math_problem（数学题目解答）
用户需要求解具体的数学题目，需要通过计算、推导、证明等步骤得出答案。
**特征**：
- 包含数学操作词：求、计算、解、证明、推导、化简、判断、比较
- 包含具体数字、参数或变量
- 需要计算过程或推理步骤
- 可能包含引导语如"你帮我..."、"请..."

**示例**：
- "求不等式 x²-5x+6<0 的解集"
- "计算 lim(x→0) sin(x)/x 的值"
- "证明：若 a>b>0，则 1/a < 1/b"
- "你帮我求一下这个方程的解"

### 2. concept_explain（概念解释）
用户询问概念、定义、定理、公式的含义，不需要计算求解。
**特征**：
- 包含概念性词汇：什么是、是什么、介绍、解释、定义、概念、含义
- 不涉及具体计算或求解
- 旨在理解知识点
- **包括**："你帮我讲解..."、"请介绍一下..."等引导语

**示例**：
- "什么是函数"
- "介绍一下等差数列"
- "导数的几何意义是什么"
- "你帮我讲解一下二项式定理"
- "请介绍一下集合的概念"

### 3. greeting（问候语）
用户进行打招呼、问候、礼貌用语。
**特征**：
- 简短的问候词汇
- 无实质性提问内容

**示例**：
- "你好"
- "早上好"
- "在吗"
- "嗨"
- "hello"

### 4. english_query（英语问题）
问题主体是英文的知识问答或英语学习问题。
**特征**：
- 问题主体是英文
- 可能是英语学习问题或英文知识问答
- **排除**：明显是ASR错误产生的无意义英文片段

**示例**：
- "What is the capital of Canada?"
- "Explain the difference between weather and climate"
- "Who wrote the play Hamlet?"
- "What is the Pythagorean theorem?"

### 5. realtime_query（需要联网检索）
用户询问需要最新信息的问题，如天气、新闻、实时行情等。
**特征**：
- 时间敏感：今天、明天、最近、当前、现在、最新
- 领域：天气、新闻、股价、汇率、行情、价格

**示例**：
- "北京今天天气怎么样"
- "最近有什么新闻"
- "现在黄金价格是多少"
- "明天会下雨吗"

### 6. general_knowledge（常识性问题）
用户询问一般知识、百科、常识类问题，不需要联网获取最新信息。
**特征**：
- 百科类知识
- 历史事实、科学常识
- 不涉及数学专业计算

**示例**：
- "中国的首都在哪里"
- "太阳系有几大行星"
- "一加一等于几"
- "水的化学式是什么"

### 7. chit_chat（闲聊/对话）
用户进行日常对话、表达情绪、与AI闲聊，不属于有效提问。
**特征**：
- 社交性对话
- 情绪表达
- 无明确知识需求
- 可能是对话中的一部分（非完整问题）

**示例**：
- "你在干嘛"
- "我累了"
- "哈哈"
- "谢谢你"
- "你是谁"

### 8. noise（噪声/无效输入）
完全无效的内容，包括多种子类型：
- ASR识别错误产生的无意义文本
- **旁人对话**（非对AI说话）："对"、"是的"、"好的"、"嗯"、"不是"、"不对"
- **设备/技术调试对话**："你看那个接口"、"这个按钮点击没反应"
- 不完整的句子碎片
- 重复填充（如"OK,OK,OK..."重复多次）

**示例**：
- "对，就是那个，或者边的那俩"
- "函数，我们20"
- "你你你你你"
- "你看那个接口返回的是什么"
- "好的好的好的"（重复多次）

### 9. other（其他）
无法归入以上任何类别的问题。

---

## 输出格式

直接返回 JSON 对象，格式如下：
```json
{
  "label": "类别名称",
  "confidence": "high/medium/low",
  "reason": "简短理由（不超过20字）"
}
```

**label 取值**：`math_problem`, `concept_explain`, `greeting`, `realtime_query`, `general_knowledge`, `chit_chat`, `noise`, `other`

**confidence 说明**：
- `high`：类别判断非常明确，无明显歧义
- `medium`：基本可以判断，但有轻微歧义
- `low`：类别判断不太确定，可能属于其他类别

只返回 JSON 对象，不要返回其他内容。"""


def parse_queries(filepath: Path) -> list[dict]:
    """解析 user_queries.txt，返回 [{index, query}, ...]"""
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            queries.append({
                "index": line_no,
                "query": line,
            })
    return queries


def classify_single(client: OpenAI, model: str, query: str) -> tuple:
    """对单个 query 调用 LLM 进行分类

    Returns:
        (result_dict, duration_seconds)
    """
    start_time = time.time()

    # 尝试使用 JSON mode（如果 API 支持）
    # SiliconFlow/Qwen 等部分支持
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请对以下 query 进行分类：\n\n{query[:500]}"},
            ],
            temperature=0.0,
            max_tokens=128,  # 分类返回的JSON很小，降低即可
            response_format={"type": "json_object"},  # 某些API支持，但可能导致错误
        )
    except Exception as e:
        # 如果JSON mode不支持，回退到普通模式
        print(e)
        print("如果JSON mode不支持，回退到普通模式")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请对以下 query 进行分类：\n\n{query[:500]}"},
            ],
            temperature=0.0,
            max_tokens=128,
        )

    duration = time.time() - start_time

    content = response.choices[0].message.content.strip()

    # 提取 JSON（可能被 markdown 代码块包裹）
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])
        if content.startswith("json"):
            content = content[4:]

    try:
        result = json.loads(content)
        return result, duration
    except json.JSONDecodeError:
        return {
            "label": "unknown",
            "confidence": "low",
            "reason": "JSON解析失败",
        }, duration


def get_progress_file(output_file: Path) -> Path:
    """获取进度文件路径"""
    return output_file.with_suffix(".progress.json")


def load_progress(progress_file: Path) -> set:
    """加载已处理的索引"""
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("processed_indices", []))
    return set()


def save_progress(progress_file: Path, processed_indices: set):
    """保存进度"""
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump({"processed_indices": list(processed_indices)}, f)


def main():
    parser = argparse.ArgumentParser(description="分类用户查询")
    parser.add_argument("-n", type=int, metavar="M", help="随机挑选 M 个问题进行分类")
    args = parser.parse_args()

    # 从环境变量读取配置
    api_key = os.getenv("RAG_Anything_OPENAI_API_KEY", "")
    api_base = os.getenv("RAG_Anything_OPENAI_API_BASE", "")
    model = os.getenv("RAG_Anything_OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        print("错误: 请设置 RAG_Anything_OPENAI_API_KEY 环境变量")
        sys.exit(1)

    if not api_base:
        print("错误: 请设置 RAG_Anything_OPENAI_API_BASE 环境变量")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base)

    # 确定输出文件名
    if args.n:
        output_file = OUTPUT_DIR / f"classification_results-n-{args.n}.json"
        summary_file = OUTPUT_DIR / f"classification_summary-n-{args.n}.txt"
    else:
        output_file = OUTPUT_DIR / "classification_results.json"
        summary_file = OUTPUT_DIR / "classification_summary.txt"

    # 解析 query 文件
    queries_file = OUTPUT_DIR / "user_queries.txt"
    if not queries_file.exists():
        print(f"错误: 找不到 {queries_file}")
        print(f"提示: 请先运行 query_user_queries.py 生成 user_queries.txt")
        sys.exit(1)

    queries = parse_queries(queries_file)

    # 随机抽样或全部处理
    if args.n:
        sample_size = min(args.n, len(queries))
        queries = random.sample(queries, sample_size)
        # 重新编号
        for i, q in enumerate(queries, 1):
            q["original_index"] = q["index"]
            q["index"] = i
        print(f"随机抽取 {len(queries)} 条 query（总共 {sample_size} 条）")
    else:
        print(f"共读取 {len(queries)} 条 query")

    print(f"使用模型: {model}")

    # 分类标签定义
    LABEL_NAMES = {
        "math_problem": "数学题目解答",
        "concept_explain": "概念解释",
        "greeting": "问候语",
        "english_query": "英语问题",
        "realtime_query": "联网检索",
        "general_knowledge": "常识性问题",
        "chit_chat": "闲聊/对话",
        "noise": "噪声/无效",
        "other": "其他",
    }

    # 加载进度
    progress_file = get_progress_file(output_file)
    processed_indices = load_progress(progress_file)
    all_results = {}

    # 如果有已保存的结果，加载它们
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_results = {r["index"]: r for r in data.get("results", [])}
                print(f"发现已有结果文件，已处理 {len(all_results)} 条")
        except Exception as e:
            print(f"加载已有结果失败: {e}")

    print(f"开始分类...\n")

    # 记录总体开始时间
    overall_start_time = time.time()
    processing_times = []  # 记录每条的处理时间

    # 逐条处理
    for i, q in enumerate(queries, 1):
        idx = q["index"]

        # 跳过已处理的
        if idx in processed_indices:
            print(f"[{i}/{len(queries)}] 跳过已处理: {idx}")
            continue

        query_text = q["query"][:60]
        print(f"[{i}/{len(queries)}] 处理 query {idx}: {query_text}...", end=" ", flush=True)

        try:
            result, duration = classify_single(client, model, q["query"])
            processing_times.append(duration)

            all_results[idx] = {
                "index": idx,
                "original_index": q.get("original_index", idx),
                "query": q["query"],
                "label": result.get("label", "unknown"),
                "confidence": result.get("confidence", "low"),
                "reason": result.get("reason", ""),
                "duration_seconds": round(duration, 3),
            }

            processed_indices.add(idx)
            print(f"[{result.get('label', 'unknown')}] ({duration:.2f}s)")

        except Exception as e:
            print(f"失败: {e}")
            all_results[idx] = {
                "index": idx,
                "original_index": q.get("original_index", idx),
                "query": q["query"],
                "label": "unknown",
                "confidence": "low",
                "reason": f"API 调用失败: {e}",
                "duration_seconds": 0,
            }
            processed_indices.add(idx)

        # 每处理一条就保存一次（结果 + 进度）
        sorted_results = sorted(all_results.values(), key=lambda x: x["index"])

        # 保存结果
        stats = defaultdict(int)
        for r in sorted_results:
            stats[r["label"]] += 1

        # 计算耗时统计
        if processing_times:
            avg_duration = sum(processing_times) / len(processing_times)
            min_duration = min(processing_times)
            max_duration = max(processing_times)
            total_duration = sum(processing_times)
        else:
            avg_duration = min_duration = max_duration = total_duration = 0

        output_data = {
            "summary": {
                "total": len(sorted_results),
                "stats": dict(stats),
                "model": model,
                "sample_size": args.n if args.n else None,
                "timing": {
                    "avg_duration_seconds": round(avg_duration, 3),
                    "min_duration_seconds": round(min_duration, 3),
                    "max_duration_seconds": round(max_duration, 3),
                    "total_duration_seconds": round(total_duration, 3),
                },
            },
            "label_names": LABEL_NAMES,
            "results": sorted_results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # 保存进度
        save_progress(progress_file, processed_indices)

        # 短暂延迟避免限流
        time.sleep(0.2)

    # 按编号排序
    sorted_results = sorted(all_results.values(), key=lambda x: x["index"])

    # 统计各分类数量
    stats = defaultdict(int)
    for r in sorted_results:
        stats[r["label"]] += 1

    total = len(sorted_results)
    overall_duration = time.time() - overall_start_time

    # 计算耗时统计
    if processing_times:
        avg_duration = sum(processing_times) / len(processing_times)
        min_duration = min(processing_times)
        max_duration = max(processing_times)
    else:
        avg_duration = min_duration = max_duration = 0

    print("\n" + "=" * 60)
    print("分类完成！统计结果：")
    print("=" * 60)
    for label in ["math_problem", "concept_explain", "greeting", "english_query",
                  "realtime_query", "general_knowledge", "chit_chat", "noise", "other", "unknown"]:
        if label in stats:
            name = LABEL_NAMES.get(label, label)
            count = stats[label]
            pct = count / total * 100
            print(f"  {name:15s}: {count:4d} ({pct:5.1f}%)")
    print(f"  {'总计':15s}: {total:4d}")

    print("\n" + "=" * 60)
    print("耗时统计：")
    print("=" * 60)
    print(f"  平均耗时: {avg_duration:.3f} 秒/条")
    print(f"  最快耗时: {min_duration:.3f} 秒")
    print(f"  最慢耗时: {max_duration:.3f} 秒")
    print(f"  总耗时: {overall_duration:.1f} 秒")
    if processing_times:
        print(f"  预计剩余: {(len(queries) - len(processing_times)) * avg_duration:.0f} 秒")

    # 写入汇总文本
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("查询分类统计汇总\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"总计: {total} 条查询\n")
        if args.n:
            f.write(f"随机抽样: {args.n} 条\n")

        f.write(f"\n使用模型: {model}\n")
        f.write(f"总耗时: {overall_duration:.1f} 秒\n")

        if processing_times:
            f.write(f"\n耗时统计:\n")
            f.write(f"  平均耗时: {avg_duration:.3f} 秒/条\n")
            f.write(f"  最快耗时: {min_duration:.3f} 秒\n")
            f.write(f"  最慢耗时: {max_duration:.3f} 秒\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("分类统计:\n")
        f.write("=" * 60 + "\n")

        for label in ["math_problem", "concept_explain", "greeting", "english_query",
                      "realtime_query", "general_knowledge", "chit_chat", "noise", "other", "unknown"]:
            if label in stats:
                name = LABEL_NAMES.get(label, label)
                count = stats[label]
                pct = count / total * 100
                f.write(f"{name}: {count} 条 ({pct:.1f}%)\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("各分类示例（前 5 条）：\n")
        f.write("=" * 60 + "\n\n")

        for label in ["math_problem", "concept_explain", "greeting", "realtime_query",
                      "general_knowledge", "chit_chat", "noise", "other"]:
            name = LABEL_NAMES.get(label, label)
            examples = [r for r in sorted_results if r["label"] == label][:5]
            if examples:
                f.write(f"\n【{name}】\n")
                for ex in examples:
                    f.write(f"  - {ex['query'][:80]}\n")
                    f.write(f"    理由: {ex['reason']}\n")

    print(f"\n结果已保存到: {output_file}")
    print(f"汇总已保存到: {summary_file}")
    print(f"进度文件: {progress_file}")


if __name__ == "__main__":
    main()
