#!/usr/bin/env python3
"""
随机抽样测试提示词版 word_to_latex，并将结果保存为 Markdown。

用法：
    cd ai-service
    conda activate morphe
    python -m app.services.word_to_latex -n 10

环境变量从 ai-service/.env 读取：
    LLM_BASE_URL
    LLM_MODEL
    LLM_API_KEY（可选，本地 OpenAI-compatible 服务不需要时可为空）
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

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

AI_SERVICE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = AI_SERVICE_DIR.parent

load_dotenv(AI_SERVICE_DIR / ".env", override=False)

WORD_TO_LATEX_LLM_BASE_URL = os.getenv("LLM_BASE_URL")
WORD_TO_LATEX_LLM_MODEL = os.getenv("LLM_MODEL")
WORD_TO_LATEX_LLM_API_KEY = (
    os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "empty"
)
WORD_TO_LATEX_TIMEOUT = float(os.getenv("WORD_TO_LATEX_TIMEOUT", "30"))
WORD_TO_LATEX_MAX_TOKENS = int(os.getenv("WORD_TO_LATEX_MAX_TOKENS", "1024"))
WORD_TO_LATEX_MAX_RETRIES = int(os.getenv("WORD_TO_LATEX_MAX_RETRIES", "0"))

WORD_TO_LATEX_SYSTEM_PROMPT = r"""你是一个 ASR 数学文本转 LaTeX 的助手。
你的任务是把中文口语数学表达转换成适合 Markdown/KaTeX 渲染和数学推理的文本。

转换规则：
1. 只转换输入中明确的数学表达、变量、上下标、比较符号、等号、括号、分式、根式、积分、数列符号等。
2. 保留原句语义和非数学文本；不要解题，不要补充条件，不要修正题目事实，除非明显是 ASR 数学读法需要规范化。
3. 完整公式用 `$...$` 包裹；短变量可以按语境包裹为 `$x$`、`$a_n$`。
4. 幂、下标、分式、积分使用标准 LaTeX：`x^{2}`、`B_n`、`\frac{1}{4}`、`\int_{0}^{1}`。
5. 数列An、数列an要转成带花括号的数列写法，例如：数列An -> 数列{$A_n$}，数列an -> 数列{$a_n$}。
6. “B的下标n”转为 `$B_n$`。
7. 如果输入不包含需要转换的数学表达，原样返回。
8. 如果句子开头有`多选题`字样，把`多选题`用小括号包起来
9. 选择题选项标记不要当作数学变量处理。`A选项`、`a选项`、`选项A`、`选项a` 应统一转换为 `A. `；`B/C/D` 同理。不要输出 `$a$选项`、`$b$选项` 这类格式。

输出格式：
只返回转换后的完整文本。
不要返回 JSON，不要返回 Markdown 代码块，不要解释，不要输出推理过程，不要添加“输出：”等前缀。

示例 1：
输入：求解不等式X的平方减去三X加二小于零的解集
输出：
求解不等式$X^{2}-3X+2<0$的解集

示例 2：
输入：已知等差数列前N项和S N等于二N的平方，加上N求等差数列的通项公式和公差。
输出：
已知等差数列前$N$项和$S_N=2N^{2}+N$，求等差数列的通项公式和公差。

示例 3：
输入：什么是土豆什么是马铃薯
输出：
什么是土豆什么是马铃薯

示例 4：
输入：求函数f(x)等于x的平方的复合函数，在椭圆的交点坐标处求导数
输出：
求函数$f(x)=x^{2}$的复合函数，在椭圆的交点坐标处求导数

示例 5：
输入：计算积分∫(0到1) x的三次方 dx的值，结果等于四分之一
输出：
计算积分$\int_{0}^{1} x^{3} dx$的值，结果$=\frac{1}{4}$

示例 6：
输入：函数f(x) 等于 a 的 x 减二次方加二，其中 a 大于零
输出：
函数$f(x)=a^{x-2}+2,\quad a>0$

示例 7：
对任意的x属于零到正无穷的开区间
输出：
若对任意的 $x\in (0, + \infty)$

示例 8：
多选题：已知b大于零，对任意的x属于零到正无穷的开区间，
输出：
(多选题)已知$b>0$，若对任意的 $x\in (0, + \infty)$，

示例 9：
输入：多选题：若f(x)等于x平方，则a选项f(x)为偶函数。b选项f(0)=1。c选项f(x)大于0。d选项f(x)有零点。
输出：
(多选题)若$f(x)=x^{2}$，则：
A. $f(x)$为偶函数。
B. $f(0)=1$。
C. $f(x)>0$。
D. $f(x)$有零点。
"""


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text.strip("`").strip()


SUBQUESTION_INDEX_MAP = {
    "一": 1,
    "二": 2,
    "三": 3,
    "第一": 1,
    "第二": 2,
    "第三": 3,
    "1": 1,
    "2": 2,
    "3": 3,
    "１": 1,
    "２": 2,
    "３": 3,
}


def _normalize_subquestion_index(raw: str):
    return SUBQUESTION_INDEX_MAP.get(raw.strip())


SUBQUESTION_SPOKEN_RE = re.compile(
    r"""
    (?P<prefix>^|[\n。！？；;，,、]\s*)
    (?P<marker>
        第\s*(?P<num1>[一二三123１２３])\s*(?:问|小题|小问)
        |
        (?P<ord>第一|第二|第三)\s*(?:问|小题|小问)
        |
        问题\s*(?P<num2>[一二三123１２３])
    )
    [，,、。．:\s]*
    """,
    re.VERBOSE,
)


def normalize_spoken_subquestion_markers(text: str) -> str:
    if not text:
        return text

    def repl(match: re.Match) -> str:
        prefix = match.group("prefix") or ""
        raw_index = (
            match.group("num1") or match.group("ord") or match.group("num2") or ""
        )

        index = _normalize_subquestion_index(raw_index)
        if index is None:
            return match.group(0)

        # 小问前的标点分两类：
        # - 强分隔（句末标点 。！？；;）：保留原标点并换两行，作为新段落起点。
        # - 弱分隔（逗号 ，, 、顿号 、）：句内停顿，原样保留、不换行。
        # 例如：第一问求E的方程。第二问证明...  ->  (1)求E的方程。\n\n(2)证明...
        #       $，第一小问：求周期。            ->  $，(1)：求周期。
        if prefix and prefix.strip() in {"。", "！", "？", "；", ";"}:
            prefix = prefix.rstrip() + "\n\n"

        return f"{prefix}({index})"

    return SUBQUESTION_SPOKEN_RE.sub(repl, text)


CHOICE_CONTEXT_RE = re.compile(
    r"下列选项哪个正确|"
    r"下列说法正确的是|"
    r"下列结论正确的是|"
    r"下列.*?正确的是|"
    r"下列.*?错误的是|"
    r"这是一道多选题|"
    r"这是一道单选题|"
    r"多选题|"
    r"单选题|"
    r"选择题"
)


CHOICE_OPTION_MARKER_RE = re.compile(
    r"""
    (?:
        \$\s*(?P<label_math1>[A-Da-d])\s*\$\s*选项
        |
        (?P<label1>[A-Da-d])\s*选项
        |
        选项\s*\$\s*(?P<label_math2>[A-Da-d])\s*\$
        |
        选项\s*(?P<label2>[A-Da-d])
    )
    [：:，,、\s]*
    """,
    re.VERBOSE,
)


def has_full_choice_option_markers(text: str) -> bool:
    """文本中是否同时出现 A/B/C/D 四个“选项”口语化标记。"""
    if not text:
        return False

    labels = set()

    for match in CHOICE_OPTION_MARKER_RE.finditer(text):
        label = (
            match.group("label_math1")
            or match.group("label1")
            or match.group("label_math2")
            or match.group("label2")
            or ""
        ).upper()
        if label:
            labels.add(label)

    return {"A", "B", "C", "D"}.issubset(labels)


def is_choice_context(text: str) -> bool:
    """是否处于选择题语境：命中显式关键词，或同时出现 A/B/C/D 四个选项标记。"""
    if not text:
        return False

    return bool(CHOICE_CONTEXT_RE.search(text)) or has_full_choice_option_markers(text)


def _format_text_before_choice_marker(before: str) -> str:
    """规范化某个“选项X”标记之前的文本片段，决定它如何收尾并换行。"""
    if before == "":
        return ""

    stripped = before.rstrip()

    # 处理：则选项A  ->  则：
    if stripped.endswith("则"):
        return stripped[:-1] + "则：\n"

    # 处理：则，选项A / 则,选项A / 则：选项A / 则:选项A  ->  则：
    for suffix in ("则，", "则,", "则：", "则:"):
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)] + "则：\n"

    # 前面已经是换行，不重复制造多余空行。
    if stripped.endswith("\n"):
        return stripped

    # 普通情况：保留前面的标点或正文，然后换行。
    return stripped + "\n"


def normalize_choice_option_markers(text: str) -> str:
    if not text:
        return text

    if not is_choice_context(text):
        return text

    result_parts = []
    last_end = 0
    matched = False

    for match in CHOICE_OPTION_MARKER_RE.finditer(text):
        label = (
            match.group("label_math1")
            or match.group("label1")
            or match.group("label_math2")
            or match.group("label2")
            or ""
        ).upper()
        if not label:
            continue

        matched = True

        before = text[last_end : match.start()]
        result_parts.append(_format_text_before_choice_marker(before))
        result_parts.append(f"{label}. ")

        last_end = match.end()

    if not matched:
        return text

    result_parts.append(text[last_end:])

    normalized = "".join(result_parts)

    # 清理 3 个以上连续换行，避免异常空行。
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()


def clean_model_output(content: str) -> str:
    cleaned = strip_code_fence(strip_think_tags(content)).strip()
    for prefix in ("输出：", "输出:", "结果：", "结果:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()

    cleaned = cleaned.strip().strip('"').strip("'").strip()
    cleaned = normalize_spoken_subquestion_markers(cleaned)
    cleaned = normalize_choice_option_markers(cleaned)
    return cleaned.strip()


def create_client() -> AsyncOpenAI:
    timeout = httpx.Timeout(
        WORD_TO_LATEX_TIMEOUT,
        connect=min(5.0, WORD_TO_LATEX_TIMEOUT),
        read=WORD_TO_LATEX_TIMEOUT,
        write=WORD_TO_LATEX_TIMEOUT,
        pool=WORD_TO_LATEX_TIMEOUT,
    )
    http_client = httpx.AsyncClient(timeout=timeout, trust_env=False)
    return AsyncOpenAI(
        api_key=WORD_TO_LATEX_LLM_API_KEY,
        base_url=WORD_TO_LATEX_LLM_BASE_URL,
        http_client=http_client,
        max_retries=WORD_TO_LATEX_MAX_RETRIES,
    )


async def word_to_latex(text: str) -> str:
    if not text or not text.strip():
        return ""
    # 输入侧预规范化：送入 LLM 之前先把 a选项 / b选项 / 选项A 这类口语化标记
    # 转成 A. / B.，降低模型把 a/b/c/d 当作数学变量而输出 $a$选项 的概率。
    # normalize_choice_option_markers 内部带 is_choice_context 判断，非选择题
    # 语境不会改动原文。
    query = normalize_choice_option_markers(text.strip())

    client = create_client()
    try:
        response = await client.chat.completions.create(
            model=WORD_TO_LATEX_LLM_MODEL,
            messages=[
                {"role": "system", "content": WORD_TO_LATEX_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": query,
                },
            ],
            temperature=0.7,
            top_p=0.8,
            presence_penalty=0,
            max_tokens=WORD_TO_LATEX_MAX_TOKENS,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }            
        )
        content = response.choices[0].message.content or ""
        return clean_model_output(content)
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="随机抽样 ASR->LaTeX 样本，调用 word_to_latex 并保存 Markdown 结果。"
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=10,
        help="随机抽样数量，默认 10。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "asr_latex_quality_review.jsonl",
        help="样本 JSONL 路径，默认 data/asr_latex_before_after.jsonl。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "word2latex_prompt_sample.md",
        help="输出 Markdown 路径，默认 data/output/word2latex_prompt_sample-时间戳.md。",
    )
    parser.add_argument(
        "--review-jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "asr_latex_quality_review.jsonl",
        help="质检结果 JSONL 路径；decision 为 drop 的 line_no 会被跳过。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；设置后可复现抽样。",
    )
    return parser.parse_args()


def load_samples(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc

            before = (item.get("before") or "").strip()
            after = (item.get("after") or "").strip()
            if before:
                samples.append(
                    {
                        "line_no": line_no,
                        "before": before,
                        "after": after,
                    }
                )
    return samples


def load_review_line_nos(path: Path) -> tuple[set[int], set[int]]:
    reviewed_line_nos: set[int] = set()
    drop_line_nos: set[int] = set()
    if not path.exists():
        return reviewed_line_nos, drop_line_nos

    with path.open("r", encoding="utf-8") as f:
        for result_line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                line_no = int(item["line_no"])
                reviewed_line_nos.add(line_no)
                if item.get("decision") == "drop":
                    drop_line_nos.add(line_no)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(f"忽略质检结果第 {result_line_no} 行: {exc}")
    return reviewed_line_nos, drop_line_nos


async def convert_samples(samples: list[dict]) -> list[dict]:
    results = []
    total = len(samples)
    for idx, sample in enumerate(samples, 1):
        before = sample["before"]
        print(f"[{idx}/{total}] line={sample['line_no']} {before[:50]}...")
        start = time.time()
        error = ""
        try:
            generated = await word_to_latex(before)
        except Exception as exc:
            generated = ""
            error = f"{type(exc).__name__}: {exc}"
            print(f"  failed: {error}")
        results.append(
            {
                **sample,
                "generated": generated,
                "error": error,
                "elapsed": time.time() - start,
            }
        )
    return results


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "data" / "output" / f"word2latex_prompt_sample-{timestamp}.md"


def write_markdown(results: list[dict], output_path: Path, source_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# word_to_latex Prompt Sample",
        "",
        f"- Created at: `{created_at}`",
        f"- Model: `{WORD_TO_LATEX_LLM_MODEL}`",
        f"- Base URL: `{WORD_TO_LATEX_LLM_BASE_URL}`",
        f"- Source: `{source_path}`",
        f"- Samples: `{len(results)}`",
        "",
    ]

    for idx, item in enumerate(results, 1):
        lines.extend(
            [
                f"## Sample {idx} (line {item['line_no']})",
                "",
                f"- Elapsed: `{item['elapsed']:.2f}s`",
                f"- Error: `{item['error'] or 'none'}`",
                "",
                "### ASR Before",
                "",
                item["before"],
                "",
                "```text",
                item["before"],
                "```",
                "",
                "### Expected After",
                "",
                item["after"] or "(empty)",
                "",
                "```text",
                item["after"] or "",
                "```",
                "",
                "### Generated",
                "",
                item["generated"] or "(empty)",
                "",
                "```text",
                item["generated"] or "",
                "```",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples 必须大于 0")

    source_path = args.input.resolve()
    if not source_path.exists():
        raise SystemExit(f"找不到样本文件: {source_path}")

    samples = load_samples(source_path)
    if not samples:
        raise SystemExit(f"样本文件为空: {source_path}")

    review_path = args.review_jsonl.resolve()
    source_count = len(samples)
    reviewed_line_nos, drop_line_nos = load_review_line_nos(review_path)
    if reviewed_line_nos:
        samples = [
            sample for sample in samples if sample["line_no"] not in drop_line_nos
        ]
        unreviewed_count = source_count - len(reviewed_line_nos)
        print(
            f"读取质检结果: {review_path}，"
            f"已质检 {len(reviewed_line_nos)} 条，"
            f"其中 drop {len(drop_line_nos)} 条，"
            f"未质检 {unreviewed_count} 条，"
            f"可测试 {len(samples)} 条"
        )
    else:
        print(f"未发现质检结果，路径: {review_path}")

    if not samples:
        raise SystemExit("过滤 drop 样本后没有可测试样本")

    rng = random.Random(args.seed)
    selected = rng.sample(samples, k=min(args.num_samples, len(samples)))
    output_path = (args.output or default_output_path()).resolve()

    print(
        f"原始样本 {source_count} 条，过滤后可测试 {len(samples)} 条，"
        f"抽样 {len(selected)} 条，"
        f"模型 {WORD_TO_LATEX_LLM_MODEL}"
    )
    results = await convert_samples(selected)
    write_markdown(results, output_path, source_path)
    print(f"\nMarkdown 结果已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
