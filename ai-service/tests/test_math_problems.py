"""
批量数学题目测试脚本

从 problems.json 读取题目，逐题调用 generate_openai_stream_v1 流式接口，
收集回答并与参考答案对比，输出准确率统计报告。

使用方法:
    # 测试全部题目
    python -m tests.test_math_problems

    # 测试前 10 道
    python -m tests.test_math_problems --limit 10

    # 指定题目类型
    python -m tests.test_math_problems --type single --limit 20

    # 指定题目范围
    python -m tests.test_math_problems --start 5 --limit 10

    # 只测试指定 ID 的题目
    python -m tests.test_math_problems --ids 0,3,7

    # 自定义模型和参数
    python -m tests.test_math_problems --model Qwen3-14B-AWQ --temperature 0.7 --limit 5

    # 指定 problems.json 路径
    python -m tests.test_math_problems --problems /path/to/problems.json --limit 5

    # 间隔时间（秒），避免连续请求压垮服务
    python -m tests.test_math_problems --limit 10 --interval 2

    # 遇到错误时跳过继续
    python -m tests.test_math_problems --limit 50 --skip-errors
"""

import asyncio
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import OpenAIChatRequest, OpenAIMessage
from app.api.endpoints.chat_stream_v1 import generate_openai_stream_v1
from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PROBLEMS_PATH = "/home/zj/math_model_deployment/Qwen2.5-Math/reports/problems.json"

DEFAULT_PARAMS = {
    "model": "qwen3:14b",
    "user_id": "3",
    "employee_id": "29",
    "team_id": "4",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
}


# =============================================================================
# 答案提取
# =============================================================================

def extract_choice_answer(text: str) -> Optional[str]:
    """
    从模型输出中提取选择题答案（A/B/C/D）。
    优先匹配明确的选项声明，其次匹配最后一个选项字母。
    """
    if not text:
        return None

    # 按优先级排列的正则模式
    patterns = [
        r'[答选]案?\s*[是为：:\s]*\s*([A-D])',
        r'[选择选定]\s*[是为：:\s]*\s*([A-D])',
        r'[应正]?\s*确\s*[选项的答案是为：:\s]*\s*([A-D])',
        r'所以\s*[是为]?\s*([A-D])',
        r'选\s*([A-D])\b',
        r'([A-D])\s*[选是项]',
        r'([A-D])\s*[.．、)]\s',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()

    # 回退：找文本中最后一个独立的 A-D
    standalone = re.findall(r'\b([A-D])\b', text)
    if standalone:
        return standalone[-1].upper()

    return None


def extract_multi_answer(text: str) -> Optional[str]:
    """从模型输出中提取多选题答案（如 'ABC', 'A,B,D'）。"""
    if not text:
        return None

    patterns = [
        r'[答选]案?\s*[是为：:\s]*\s*([A-D](?:\s*[,，、]\s*[A-D])*)',
        r'选\s*([A-D](?:\s*[,，、]\s*[A-D])*)\b',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            last = matches[-1]
            letters = sorted(set(re.findall(r'[A-D]', last)))
            return ''.join(letters) if letters else None

    # 回退：找所有 A-D
    letters = sorted(set(re.findall(r'\b([A-D])\b', text)))
    if letters:
        return ''.join(letters)

    return None


def extract_blank_answer(text: str) -> Optional[str]:
    """从填空题输出中提取数值/表达式答案。"""
    if not text:
        return None

    # 匹配 "答案/X=" 后面的数值
    patterns = [
        r'答案\s*[是为：:=\s]+\s*(.+?)(?:\n|$)',
        r'所以.*?=\s*(.+?)(?:\n|$|。)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()

    return text.strip()


def extract_answer(text: str, problem_type: str) -> Optional[str]:
    """根据题目类型提取答案。"""
    if problem_type == "single":
        return extract_choice_answer(text)
    elif problem_type == "multi":
        return extract_multi_answer(text)
    elif problem_type in ("blank", "unknown"):
        return extract_blank_answer(text)
    return extract_choice_answer(text)


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较。"""
    return re.sub(r'\s+', '', answer).upper()


def check_answer(model_answer: Optional[str], reference: str, problem_type: str) -> bool:
    """判断模型答案是否正确。"""
    if not model_answer:
        return False

    ref_norm = normalize_answer(reference)
    ans_norm = normalize_answer(model_answer)

    if problem_type == "single":
        return ans_norm == ref_norm
    elif problem_type == "multi":
        ref_set = set(ref_norm)
        ans_set = set(ans_norm)
        return ref_set == ans_set
    else:
        return ans_norm == ref_norm


# =============================================================================
# 核心测试逻辑
# =============================================================================

async def test_single_problem(
    problem: Dict[str, Any],
    index: int,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """测试单道题目，返回结果字典。"""
    question = problem["question"]
    reference = problem.get("reference_answer", "")
    problem_type = problem.get("_raw", {}).get("type", "single")

    result = {
        "index": index,
        "question": question[:200],
        "reference_answer": reference,
        "problem_type": problem_type,
        "model_answer": None,
        "is_correct": False,
        "full_response": "",
        "duration_ms": 0,
        "chunk_count": 0,
        "error": None,
    }

    session_id = f"sess_math_test_{params['user_id']}_{params['employee_id']}"

    request = OpenAIChatRequest(
        model=params["model"],
        messages=[OpenAIMessage(role="user", content=question)],
        stream=True,
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=params["max_tokens"],
        presence_penalty=0.0,
        frequency_penalty=0.0,
        seed=None,
        n=1,
        tools=None,
        employee_id=params["employee_id"],
        user_id=params["user_id"],
        session_id=session_id,
        channel_name=None,
        team_id=params["team_id"],
    )

    start_time = time.time()
    full_response = ""
    chunk_count = 0

    try:
        async for chunk_str in generate_openai_stream_v1(request):
            chunk_count += 1

            if chunk_str == "[DONE]":
                break

            try:
                chunk_data = json.loads(chunk_str)
                if "choices" in chunk_data and chunk_data["choices"]:
                    delta = chunk_data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_response += content
            except json.JSONDecodeError:
                pass

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error(f"[#{index}] 测试失败: {e}")
        # 出错时也返回已收集的结果

    elapsed = time.time() - start_time

    model_answer = extract_answer(full_response, problem_type)
    is_correct = check_answer(model_answer, reference, problem_type) if reference else False

    result.update({
        "model_answer": model_answer,
        "is_correct": is_correct,
        "full_response": full_response,
        "duration_ms": int(elapsed * 1000),
        "chunk_count": chunk_count,
    })

    return result


# =============================================================================
# 报告输出
# =============================================================================

def print_progress(result: Dict[str, Any], current: int, total: int, correct_count: int):
    """打印单题进度。"""
    status = "✓" if result["is_correct"] else "✗"
    ref = result["reference_answer"] or "?"
    ans = result["model_answer"] or "无答案"
    err = f" | ERROR: {result['error'][:60]}" if result["error"] else ""

    print(
        f"[{current}/{total}] {status} "
        f"题#{result['index']} ({result['problem_type']}) "
        f"参考={ref} 模型={ans} "
        f"耗时={result['duration_ms']}ms{err}"
    )


def print_report(results: List[Dict[str, Any]], params: Dict[str, Any], output_path: Optional[str]):
    """打印汇总报告并可选保存到文件。"""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    errors = sum(1 for r in results if r["error"])
    no_answer = sum(1 for r in results if not r["model_answer"] and not r["error"])
    durations = [r["duration_ms"] for r in results]

    # 按题型统计
    by_type: Dict[str, Dict[str, int]] = {}
    for r in results:
        t = r["problem_type"]
        if t not in by_type:
            by_type[t] = {"total": 0, "correct": 0, "errors": 0, "no_answer": 0}
        by_type[t]["total"] += 1
        if r["is_correct"]:
            by_type[t]["correct"] += 1
        if r["error"]:
            by_type[t]["errors"] += 1
        if not r["model_answer"] and not r["error"]:
            by_type[t]["no_answer"] += 1

    # 错题列表
    wrong = [r for r in results if not r["is_correct"] and not r["error"]]

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("数学题目批量测试报告")
    lines.append("=" * 70)
    lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模型: {params['model']} | temperature={params['temperature']} | max_tokens={params['max_tokens']}")
    lines.append(f"题目数: {total} | 正确: {correct} | 错误(程序): {errors} | 无答案: {no_answer}")
    lines.append(f"总耗时: {sum(durations)}ms | 平均: {sum(durations)//max(total,1)}ms | "
                 f"最长: {max(durations, default=0)}ms | 最短: {min(durations, default=0)}ms")

    if total > 0:
        lines.append(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")
        if total - errors > 0:
            lines.append(f"准确率(排除程序错误): {correct}/{total-errors} = {correct/(total-errors)*100:.1f}%")

    lines.append("")
    lines.append("-" * 70)
    lines.append("按题型统计:")
    lines.append("-" * 70)
    for t, stats in sorted(by_type.items()):
        rate = f"{stats['correct']}/{stats['total']}"
        pct = f"{stats['correct']/max(stats['total'],1)*100:.1f}%"
        lines.append(f"  {t:10s}: {rate:>8s} = {pct:>6s}  (程序错误={stats['errors']}, 无答案={stats['no_answer']})")

    if wrong:
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"答错题目 ({len(wrong)} 道):")
        lines.append("-" * 70)
        for r in wrong[:50]:  # 最多显示 50 道
            q_preview = r["question"][:80].replace("\n", " ")
            lines.append(
                f"  #{r['index']:3d} [{r['problem_type']}] "
                f"参考={r['reference_answer']} 模型={r['model_answer'] or '无'} | {q_preview}"
            )
        if len(wrong) > 50:
            lines.append(f"  ... 还有 {len(wrong) - 50} 道未显示")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    if output_path:
        report = {
            "generated_at": datetime.now().isoformat(),
            "params": params,
            "summary": {
                "total": total,
                "correct": correct,
                "errors": errors,
                "no_answer": no_answer,
                "accuracy": correct / total if total else 0,
                "accuracy_excl_errors": correct / (total - errors) if total - errors else 0,
                "total_duration_ms": sum(durations),
                "avg_duration_ms": sum(durations) // max(total, 1),
            },
            "by_type": by_type,
            "results": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存至: {output_path}")


# =============================================================================
# 主函数
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="批量数学题目测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--problems", "-p", type=str, default=DEFAULT_PROBLEMS_PATH,
                        help="problems.json 路径")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="测试题目数量（默认全部）")
    parser.add_argument("--start", type=int, default=0,
                        help="起始题目索引（从 0 开始）")
    parser.add_argument("--ids", type=str, default=None,
                        help="指定题目索引，逗号分隔（如 0,3,7）")
    parser.add_argument("--type", type=str, default=None, choices=["single", "multi", "blank", "unknown"],
                        help="只测试指定类型")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_PARAMS["model"],
                        help="模型名称")
    parser.add_argument("--temperature", type=float, default=DEFAULT_PARAMS["temperature"],
                        help="采样温度")
    parser.add_argument("--top-p", type=float, default=DEFAULT_PARAMS["top_p"],
                        help="核采样参数")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_PARAMS["max_tokens"],
                        help="最大生成 token 数")
    parser.add_argument("--user-id", "-u", type=str, default=DEFAULT_PARAMS["user_id"],
                        help="用户 ID")
    parser.add_argument("--employee-id", "-e", type=str, default=DEFAULT_PARAMS["employee_id"],
                        help="数字员工 ID")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="题目间隔时间（秒），默认 1.0")
    parser.add_argument("--skip-errors", action="store_true",
                        help="遇到错误跳过继续")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="结果输出 JSON 路径")

    args = parser.parse_args()

    # 加载题目
    problems_path = Path(args.problems)
    if not problems_path.exists():
        print(f"错误: 找不到题目文件 {problems_path}")
        sys.exit(1)

    with open(problems_path, "r", encoding="utf-8") as f:
        all_problems = json.load(f)

    # 筛选题目
    if args.ids:
        indices = [int(i) for i in args.ids.split(",")]
        problems = [(i, all_problems[i]) for i in indices if i < len(all_problems)]
    else:
        start = args.start
        end = args.start + args.limit if args.limit else len(all_problems)
        problems = list(enumerate(all_problems))[start:end]

    if args.type:
        problems = [(i, p) for i, p in problems if p.get("_raw", {}).get("type") == args.type]

    if not problems:
        print("没有符合条件的题目")
        sys.exit(0)

    params = {
        "model": args.model,
        "user_id": args.user_id,
        "employee_id": args.employee_id,
        "team_id": "4",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    total = len(problems)
    print(f"加载 {total} 道题目 (共 {len(all_problems)} 道)")
    print(f"参数: model={params['model']}, temperature={params['temperature']}, max_tokens={params['max_tokens']}")
    print(f"间隔: {args.interval}s | 跳过错误: {args.skip_errors}")
    print("-" * 70)

    results = []
    correct_count = 0

    for idx, (problem_index, problem) in enumerate(problems):
        result = await test_single_problem(problem, problem_index, params)
        results.append(result)

        if result["is_correct"]:
            correct_count += 1

        print_progress(result, idx + 1, total, correct_count)

        # 题目间隔
        if idx < total - 1 and args.interval > 0:
            await asyncio.sleep(args.interval)

    print_report(results, params, args.output)


if __name__ == "__main__":
    asyncio.run(main())
