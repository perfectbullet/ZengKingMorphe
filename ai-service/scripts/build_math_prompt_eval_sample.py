#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学提示词评测 - 样本缓存脚本

用途:
    从完整 JSONL 数据集中, 根据 difficulty 字段生成固定评测样本缓存。

采样规则:
    困难 / 压轴: 全部保留
    中等:        随机 medium-count 道 (默认 30)
    简单:        随机 easy-count   道 (默认 20)

要求:
    1. 使用固定随机种子;
    2. 输出为 JSONL;
    3. 每条记录保留原始字段;
    4. 每条记录增加 eval_sample 元信息;
    5. 如果某个难度数量不足, 全部保留并打印 warning;
    6. 输出目录不存在时自动创建;
    7. 生成缓存后, 后续推理评测脚本只读缓存文件, 不再读完整数据集。

用法:
    python ai-service/scripts/build_math_prompt_eval_sample.py \\
      --input data/math_qa_275_20260617.mineru.filled_reference_answer-v2.jsonl \\
      --output data/math_prompt_eval/math_qa_eval_sample_difficult_all_medium30_easy20_seed20260624.jsonl \\
      --seed 20260624 --medium-count 30 --easy-count 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

# 难度 -> 处理方式
# 困难 / 压轴: 全部保留; 中等 / 简单: 按数量随机采样
KEEP_ALL_DIFFICULTIES = ("困难", "压轴")
# 采样顺序固定, 保证同一 seed 下结果稳定
SAMPLE_ORDER = ("中等", "简单")


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"解析失败 {path}:{ln}: {e}")
                raise
            records.append(obj)
    return records


def group_by_difficulty(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        diff = rec.get("difficulty", "UNKNOWN")
        groups[diff].append(rec)
    # 每个 group 按 id 排序, 保证采样顺序与输入顺序解耦, 结果仅由 seed 决定
    for diff in groups:
        groups[diff].sort(key=lambda r: str(r.get("id", "")))
    return dict(groups)


def build_sample_rule(medium_count: int, easy_count: int, keep_all_diffs: list[str]) -> str:
    parts = [f"{d}全部" for d in keep_all_diffs]
    parts.append(f"中等{medium_count}")
    parts.append(f"简单{easy_count}")
    return " + ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从完整 JSONL 数据集中生成固定评测样本缓存",
    )
    parser.add_argument("--input", required=True, help="完整 JSONL 数据集路径")
    parser.add_argument("--output", required=True, help="样本缓存 JSONL 输出路径")
    parser.add_argument("--seed", type=int, default=20260624, help="随机种子, 默认 20260624")
    parser.add_argument("--medium-count", type=int, default=30, help="中等题采样数量, 默认 30")
    parser.add_argument("--easy-count", type=int, default=20, help="简单题采样数量, 默认 20")
    args = parser.parse_args()

    in_path = args.input
    out_path = Path(args.output)
    seed = args.seed

    if not os.path.exists(in_path):
        logger.error(f"输入文件不存在: {in_path}")
        return 2

    records = load_jsonl(in_path)
    logger.info(f"读取输入: {in_path} 共 {len(records)} 条")

    groups = group_by_difficulty(records)
    for diff in sorted(groups.keys()):
        logger.info(f"  difficulty={diff!r}: {len(groups[diff])} 条")

    # 确定哪些难度全保留 (困难一定全保留; 压轴也按困难处理全保留)
    keep_all_diffs = [d for d in KEEP_ALL_DIFFICULTIES if d in groups]

    rng = random.Random(seed)
    sample_rule = build_sample_rule(args.medium_count, args.easy_count, keep_all_diffs)

    selected: list[tuple[str, dict]] = []

    # 1) 困难 / 压轴: 全部保留
    for diff in keep_all_diffs:
        logger.info(f"[KEEP-ALL] {diff}: {len(groups[diff])} 条全部保留")
        for rec in groups[diff]:
            selected.append((diff, rec))

    # 2) 中等 / 简单: 固定顺序随机采样 (共用同一个 rng, 保证 seed 决定结果)
    requested = {"中等": args.medium_count, "简单": args.easy_count}
    for diff in SAMPLE_ORDER:
        pool = groups.get(diff, [])
        want = requested[diff]
        if len(pool) < want:
            logger.warning(
                f"[WARN] {diff} 数量不足: 请求 {want} 条, 实际只有 {len(pool)} 条, 全部保留"
            )
            chosen = pool
        else:
            chosen = rng.sample(pool, want)
        logger.info(f"[SAMPLE] {diff}: 请求 {want}, 采样 {len(chosen)} 条")
        for rec in chosen:
            selected.append((diff, rec))

    # 3) 其它未覆盖的难度 (理论上不应出现), 全部保留并 warning
    covered = set(KEEP_ALL_DIFFICULTIES) | set(SAMPLE_ORDER)
    for diff, pool in groups.items():
        if diff in covered:
            continue
        logger.warning(
            f"[WARN] 未识别的 difficulty={diff!r}: {len(pool)} 条, 全部保留"
        )
        for rec in pool:
            selected.append((diff, rec))

    # 自动创建输出目录
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for diff, rec in selected:
            out_rec = dict(rec)  # 保留原始字段
            out_rec["eval_sample"] = {
                "sample_source": "cached",
                "sample_seed": seed,
                "sample_rule": sample_rule,
                "difficulty": rec.get("difficulty", diff),
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            total += 1

    logger.info(
        f"写入输出: {out_path} 共 {total} 条 (seed={seed}, sample_rule={sample_rule})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
