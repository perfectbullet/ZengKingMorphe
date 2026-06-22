#!/usr/bin/env python3
"""
审查 ASR->LaTeX 样本数据，识别不适合作为口语转公式样本的记录。

用法：
    cd /home/zj/ZengKingMorphe
    conda activate morphe
    python ai-service/scripts/review_asr_latex_samples.py
    python ai-service/scripts/review_asr_latex_samples.py -n 50 --seed 42

环境变量从 ai-service/.env 读取：
    MATH_MODEL_BASE_URL
    MATH_MODEL_NAME
    MATH_MODEL_API_KEY
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

AI_SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_SERVICE_DIR.parent

load_dotenv(AI_SERVICE_DIR / ".env", override=False)

MODEL_BASE_URL = os.getenv("MATH_MODEL_BASE_URL", "").strip()
MODEL_NAME = os.getenv("MATH_MODEL_NAME", "").strip()
MODEL_API_KEY = os.getenv("MATH_MODEL_API_KEY", "").strip()
REVIEW_TIMEOUT = float(os.getenv("ASR_LATEX_REVIEW_TIMEOUT", "60"))
REVIEW_MAX_TOKENS = int(os.getenv("ASR_LATEX_REVIEW_MAX_TOKENS", "256"))
REVIEW_MAX_RETRIES = int(os.getenv("ASR_LATEX_REVIEW_MAX_RETRIES", "0"))
REVIEW_TRUST_ENV = os.getenv("ASR_LATEX_REVIEW_TRUST_ENV", "false").lower() == "true"

SYSTEM_PROMPT = """你是一个数学 ASR->LaTeX 数据质检员。

任务：
审查用户输入，判断它是否适合作为“口语化数学表达 -> LaTeX/数学文本”的输入。
每次只会收到一条输入文本。

重点检查两个问题：
1. before_is_oral_asr：
   - true：用户输入是中文口语/ASR 风格的数学表达，例如“x 的平方”“二的 x 次方”“数列 an”“大于等于零”。
   - false：用户输入已经是书面数学文本或 LaTeX/TeX 表达，例如包含 \\( ... \\)、\\frac、\\cdot、成段符号公式，或者明显不是口语化 ASR。

2. is_complete_math_problem：
   - true：用户输入是完整或基本完整的数学题目/问题描述，有明确求解、证明、计算、讨论、判断、取值范围等任务。
   - false：用户输入是残缺片段、半句话、缺少关键问题目标、明显 ASR 中断，例如只说到“在零点。”却没有说明求什么。

决策：
- keep：before_is_oral_asr=true 且 is_complete_math_problem=true。
- drop：before_is_oral_asr=false 或 is_complete_math_problem=false。
- decision 不需要模型输出，脚本会根据两个布尔字段判断。

返回格式：
只返回一个 JSON 对象，不要输出 Markdown 代码块、解释或其它文本。
JSON 对象格式必须是：
{
  "before_is_oral_asr": true,
  "is_complete_math_problem": true
}

要求：
- 不要改写题目，不要做 ASR->LaTeX 转换，只做数据质检判断。
- 不要输出 decision、issues、reason；只输出 before_is_oral_asr 和 is_complete_math_problem 两个布尔字段。

示例：
输入：
函数 f(x) 等于二的 x 次方加上 x 的平方，则函数呃 f(x) 在零点。

输出：
{
  "before_is_oral_asr": true,
  "is_complete_math_problem": false
}

示例：
输入：
若函数 \\( f(x) = \\frac{1 + a \\cdot e^{-x^2}}{x} \\) 的最小值为负二，则正实数 \\( a \\) 的值为多少？

输出：
{
  "before_is_oral_asr": false,
  "is_complete_math_problem": true
}

示例：
输入：
若函数 \\( f(x) = \\frac{1 + a \\cdot e^{-x^2}}{x} \\) 的最小值为负二，则正实数 \\( a \\) 的值为多少？

输出：
{
  "before_is_oral_asr": false,
  "is_complete_math_problem": true
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 LLM 审查 ASR->LaTeX JSONL 样本是否口语化且题干完整。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "asr_latex_before_after.jsonl",
        help="输入 JSONL，默认 data/asr_latex_before_after.jsonl。",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="随机抽样数量；不传则处理全部。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；设置后可复现抽样。",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="输出 JSONL 路径，默认 data/output/asr_latex_quality_review-时间戳.jsonl。",
    )
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            before = str(item.get("before") or "").strip()
            after = str(item.get("after") or "").strip()
            if before:
                samples.append(
                    {
                        "line_no": line_no,
                        "before": before,
                        "after": after,
                    }
                )
    return samples


def create_client() -> AsyncOpenAI:
    if not MODEL_BASE_URL or not MODEL_NAME or not MODEL_API_KEY:
        raise SystemExit(
            "请在 ai-service/.env 中设置 MATH_MODEL_BASE_URL、"
            "MATH_MODEL_NAME、MATH_MODEL_API_KEY"
        )

    timeout = httpx.Timeout(
        REVIEW_TIMEOUT,
        connect=min(10.0, REVIEW_TIMEOUT),
        read=REVIEW_TIMEOUT,
        write=REVIEW_TIMEOUT,
        pool=REVIEW_TIMEOUT,
    )
    http_client = httpx.AsyncClient(timeout=timeout, trust_env=REVIEW_TRUST_ENV)
    return AsyncOpenAI(
        api_key=MODEL_API_KEY,
        base_url=MODEL_BASE_URL,
        http_client=http_client,
        max_retries=REVIEW_MAX_RETRIES,
    )


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text.strip("`").strip()


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def extract_json_object(content: str) -> dict[str, Any]:
    cleaned = strip_code_fence(strip_think_tags(content)).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError("模型没有返回合法 JSON 对象")


def normalize_review(raw: dict[str, Any]) -> dict[str, Any]:
    before_is_oral_asr = raw.get("before_is_oral_asr")
    is_complete_math_problem = raw.get("is_complete_math_problem")
    if not isinstance(before_is_oral_asr, bool):
        raise ValueError("JSON 缺少布尔字段 before_is_oral_asr")
    if not isinstance(is_complete_math_problem, bool):
        raise ValueError("JSON 缺少布尔字段 is_complete_math_problem")

    decision, issues = decide_from_flags(
        before_is_oral_asr,
        is_complete_math_problem,
    )
    return {
        "before_is_oral_asr": before_is_oral_asr,
        "is_complete_math_problem": is_complete_math_problem,
        "decision": decision,
        "issues": issues,
    }


def decide_from_flags(
    before_is_oral_asr: bool,
    is_complete_math_problem: bool,
) -> tuple[str, list[str]]:
    if before_is_oral_asr and is_complete_math_problem:
        return "keep", []

    issues = []
    if not before_is_oral_asr:
        issues.append("not_oral_asr")
    if not is_complete_math_problem:
        issues.append("incomplete_problem")
    return "drop", issues


async def review_one_sample(client: AsyncOpenAI, before: str) -> dict[str, Any]:
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": before},
        ],
        temperature=0.0,
        max_tokens=REVIEW_MAX_TOKENS,
    )
    content = response.choices[0].message.content or ""
    raw = extract_json_object(content)
    return normalize_review(raw)


async def review_samples(
    samples: list[dict[str, Any]],
    output_jsonl: Path,
    processed_line_nos: set[int],
) -> list[dict[str, Any]]:
    client = create_client()
    results: list[dict[str, Any]] = []
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    try:
        for idx, sample in enumerate(samples, 1):
            if sample["line_no"] in processed_line_nos:
                print(
                    f"[{idx}/{len(samples)}] line={sample['line_no']} skipped",
                    flush=True,
                )
                continue
            print(f"[{idx}/{len(samples)}] line={sample['line_no']}...", flush=True)
            start = time.time()
            try:
                review = await review_one_sample(client, sample["before"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"  failed: {error}")
                review = {
                    "before_is_oral_asr": False,
                    "is_complete_math_problem": False,
                    "decision": "drop",
                    "issues": ["not_oral_asr", "incomplete_problem"],
                    "error": error[:160],
                }

            result = {
                **sample,
                **review,
                "elapsed_seconds": round(time.time() - start, 2),
            }
            results.append(result)
            append_jsonl(result, output_jsonl)
            processed_line_nos.add(sample["line_no"])
            print(
                f"  saved: decision={result['decision']} jsonl={output_jsonl}",
                flush=True,
            )
    finally:
        await client.close()

    return results


def default_output_path() -> Path:
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "data" / "asr_latex_quality_review.jsonl"


def load_processed_line_nos(path: Path) -> set[int]:
    processed: set[int] = set()
    if not path.exists():
        return processed

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                processed.add(int(item["line_no"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(f"忽略结果文件第 {line_no} 行: {exc}", flush=True)
    return processed


def append_jsonl(item: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    samples = load_samples(source_path)
    if not samples:
        raise SystemExit(f"样本为空: {source_path}")

    if args.num_samples is not None:
        if args.num_samples <= 0:
            raise SystemExit("--num-samples 必须大于 0")
        rng = random.Random(args.seed)
        samples = rng.sample(samples, k=min(args.num_samples, len(samples)))

    output_jsonl = (args.output_jsonl or default_output_path()).resolve()
    processed_line_nos = load_processed_line_nos(output_jsonl)

    print(f"读取 {len(samples)} 条样本，模型 {MODEL_NAME}，开始质检...")
    print(f"JSONL 输出: {output_jsonl}")
    if processed_line_nos:
        print(f"已处理 {len(processed_line_nos)} 条，将按 line_no 跳过。")
    await review_samples(samples, output_jsonl, processed_line_nos)
    print(f"JSONL 已保存: {output_jsonl}")


if __name__ == "__main__":
    asyncio.run(main())
