#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学提示词评测脚本 (离线)

目标: 离线验证两种推理策略在多问题数学题上的效果
    1. single_full_question        完整题干一次性推理
    2. sequential_two_part         两问拆开顺序推理

链路:
    读取缓存样本 JSONL
      -> 正则拆题
      -> 根据拆题结果执行推理策略
      -> 调用数学模型流式接口
      -> 累计输出
      -> 去掉 think 标签
      -> append-only 写入明细 JSONL
      -> 支持断点续跑
      -> 生成 merged 聚合 JSONL

约束:
    - 不修改线上接口逻辑
    - 不修改 ai-service/app/services/math_agent_service.py
    - 第一版串行执行 (concurrency=1)

环境变量 (直接读取, 不写进 config.py):
    MATH_MODEL_BASE_URL   必填
    MATH_MODEL_NAME       必填
    MATH_MODEL_API_KEY    可选, 默认 nokey
    MATH_MAX_TOKEN        可选, 默认 30720
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

# 让 loguru 输出到 stderr, 不污染 stdout 的 split-only 摘要
logger.remove()
logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])

# =====================================================================
# 环境加载
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
# ai-service/.env 位于 scripts/ 的上一级
CANDIDATE_ENV_PATHS = [
    SCRIPT_DIR.parent / ".env",          # ai-service/.env
    Path.cwd() / ".env",
    Path.cwd() / "ai-service" / ".env",
]


def load_env_if_present() -> None:
    """若存在 .env 文件则加载, 不覆盖已存在的环境变量。"""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:  # pragma: no cover - dotenv 不可用时的兜底
        load_dotenv = None  # type: ignore

    for env_path in CANDIDATE_ENV_PATHS:
        if not env_path.exists():
            continue
        if load_dotenv is not None:
            load_dotenv(env_path, override=False)
        else:
            _manual_load_dotenv(env_path)
        logger.debug(f"已加载 env 文件: {env_path}")
        return


def _manual_load_dotenv(env_path: Path) -> None:
    """无 python-dotenv 时的简易 KEY=VALUE 解析器 (不覆盖已有值)。"""
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


# =====================================================================
# 第三部分: 模型配置
# =====================================================================


class ModelConfigError(Exception):
    pass


def get_model_config() -> dict:
    base_url = os.getenv("MATH_MODEL_BASE_URL", "").strip()
    model_name = os.getenv("MATH_MODEL_NAME", "").strip()
    api_key = os.getenv("MATH_MODEL_API_KEY", "").strip()
    max_token = int(os.getenv("MATH_MAX_TOKEN", "30720"))

    if not base_url:
        raise ModelConfigError("MATH_MODEL_BASE_URL 未设置")
    if not model_name:
        raise ModelConfigError("MATH_MODEL_NAME 未设置")

    return {
        "base_url": base_url,
        "model_name": model_name,
        "max_tokens": max_token,
        "api_key_set": bool(api_key),
        # 实际 key 只在内存中使用, 不落盘
        "api_key": api_key or "nokey",
    }


# =====================================================================
# 第四部分: 正则拆题
# =====================================================================

# 小问标记必须在行首或换行后, 避免误切 M(1,3/2) / P(4,0) / (0,+∞) / 点（1，1）
SUBQUESTION_MARKER_RE = re.compile(
    r"""
    (?P<prefix>^|\n)
    [ \t]*
    (?P<marker>
        [（(]\s*(?P<num>[1-9]\d*)\s*[）)]
        |
        [（(]\s*(?P<roman>[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|I{1,3}|IV|V|VI{0,3}|IX|X)\s*[）)]
        |
        第\s*(?P<cn>[一二三四五六七八九十0-9０-９１２３４５６７８９]+)\s*(?:问|小题)[。．、，,]?
        |
        (?P<cn_word>第一|第二|第三|第四|第五|第六|第七|第八|第九|第十)\s*(?:问|小题)[。．、，,]?
    )
    """,
    re.MULTILINE | re.VERBOSE,
)


def _derive_marker_index(m: re.Match) -> str:
    """从匹配中提取小问序号 (仅用于信息展示, 不影响拆题判定)。"""
    if m.group("num"):
        return m.group("num")
    if m.group("roman"):
        return m.group("roman")
    if m.group("cn"):
        return m.group("cn")
    if m.group("cn_word"):
        return m.group("cn_word")
    return "?"


def split_question(question: str) -> dict:
    """正则拆题, 返回 split_result。

    marker_count == 0  -> single
    marker_count == 2  -> two_part
    marker_count 其它  -> fallback_single_due_marker_count
    """
    text = question or ""
    matches = list(SUBQUESTION_MARKER_RE.finditer(text))
    count = len(matches)

    if count == 0:
        return {
            "split_status": "single",
            "matched_marker_count": 0,
            "common_stem": text.strip(),
            "subquestions": [],
        }

    # 公共题干: 第一个标记之前的内容, 去掉行首题号 (如 "24. ")
    common_stem = text[: matches[0].start()].rstrip()
    common_stem = re.sub(r"^\s*\d+[.．、)]\s*", "", common_stem).strip()

    subquestions = []
    for i, m in enumerate(matches):
        marker = m.group("marker")
        start = m.start("marker")
        end = matches[i + 1].start() if i + 1 < count else len(text)
        raw_subquestion_text = text[start:end].strip()
        cls = classify_subquestion_type(raw_subquestion_text)
        subquestions.append(
            {
                "index": _derive_marker_index(m),
                "marker": marker,
                "type": cls["type"],
                "classification_reason": cls["classification_reason"],
                "question": raw_subquestion_text,
                "raw_subquestion_text": raw_subquestion_text,
                "clean_subquestion_text": cls["clean_subquestion_text"],
            }
        )

    status = "two_part" if count == 2 else "fallback_single_due_marker_count"
    return {
        "split_status": status,
        "matched_marker_count": count,
        "common_stem": common_stem,
        "subquestions": subquestions,
    }


# =====================================================================
# 第五部分: 小问分类
# =====================================================================

PROOF_RE = re.compile(r"证明|求证|试证|证[:：]|说明.*成立")


def strip_subquestion_marker(text: str) -> str:
    text = text.strip()

    text = re.sub(
        r"^[（(]\s*(?:[1-9]\d*|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|I{1,3}|IV|V|VI{0,3}|IX|X)\s*[）)]\s*",
        "",
        text,
    )

    text = re.sub(
        r"^第\s*[一二三四五六七八九十0-9０-９１２３４５６７８９]+\s*(?:问|小题)[。．、，,]?\s*",
        "",
        text,
    )

    text = re.sub(
        r"^(?:第一|第二|第三|第四|第五|第六|第七|第八|第九|第十)\s*(?:问|小题)[。．、，,]?\s*",
        "",
        text,
    )

    return text.strip()


def classify_subquestion_type(raw_text: str) -> dict:
    clean_text = strip_subquestion_marker(raw_text)

    if PROOF_RE.search(clean_text):
        return {
            "type": "proof",
            "clean_subquestion_text": clean_text,
            "classification_reason": "matched_proof_keyword",
        }

    return {
        "type": "non_proof",
        "clean_subquestion_text": clean_text,
        "classification_reason": "default_non_proof",
    }


# =====================================================================
# 第六部分: 提示词
# =====================================================================

# boxed 普通数学提示词 (single_full_question baseline / sequential non_proof 小问)
SYSTEM_PROMPT_BOXED = "请逐步推理，并将最终答案放在 \\boxed{} 中。请全程使用中文作答。"
PROMPT_BOXED_BASELINE = ("boxed_baseline_v1", "v1")
PROMPT_NON_PROOF_BOXED = ("non_proof_boxed_v1", "v1")

# proof 正向证明提示词 (sequential proof 小问)
SYSTEM_PROMPT_PROOF = "请给出完整、严谨的证明过程，并以证明性语言收束。请全程使用中文作答。"
PROMPT_PROOF_POSITIVE = ("proof_positive_v1", "v1")


def prompt_for_subquestion_type(sq_type: str) -> tuple[str, str, str]:
    """按小问类型返回 (system_prompt, prompt_name, prompt_version)。"""
    if sq_type == "proof":
        name, ver = PROMPT_PROOF_POSITIVE
        return SYSTEM_PROMPT_PROOF, name, ver
    name, ver = PROMPT_NON_PROOF_BOXED
    return SYSTEM_PROMPT_BOXED, name, ver


# =====================================================================
# 第九部分: think 内容清理
# =====================================================================

THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think\s*>",
    re.IGNORECASE | re.DOTALL,
)


def strip_think_blocks(text: str) -> tuple[str, bool]:
    if not text:
        return "", False

    has_think = bool(THINK_BLOCK_RE.search(text)) or bool(
        re.search(r"<think\b[^>]*>", text, re.IGNORECASE)
    )

    cleaned = THINK_BLOCK_RE.sub("", text)

    # 兼容异常情况: 出现未闭合 <think>, 这类内容不落盘
    cleaned = re.sub(
        r"<think\b[^>]*>.*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return cleaned.strip(), has_think


# =====================================================================
# 第十部分: 断点续跑 - run_key
# =====================================================================


def question_hash(question: str) -> str:
    return hashlib.sha256((question or "").encode("utf-8")).hexdigest()


def make_run_key(components: dict) -> str:
    text = json.dumps(components, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_run_key_components(
    *,
    record_id: str,
    q_hash: str,
    strategy: str,
    task_type: str,
    subquestion_index,
    prompt_name: str,
    prompt_version: str,
    model_config: dict,
) -> dict:
    return {
        "id": record_id,
        "question_hash": q_hash,
        "strategy": strategy,
        "task_type": task_type,
        "subquestion_index": subquestion_index,
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "model_base_url": model_config["base_url"],
        "model_name": model_config["model_name"],
        "max_tokens": model_config["max_tokens"],
    }


# =====================================================================
# 明细 JSONL append-only 写入
# =====================================================================


class AppendStore:
    """append-only 明细 JSONL, 写完 flush + fsync。"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # append 模式打开, 持续写入
        self._fh = open(self.path, "a", encoding="utf-8")

    def append(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def load_completed_run_keys(path: str) -> set[str]:
    """读取明细 JSONL, 仅收集 status==completed 的 run_key。"""
    completed: set[str] = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("status") == "completed" and obj.get("run_key"):
                completed.add(obj["run_key"])
    return completed


# =====================================================================
# 第八部分: 流式模型调用
# =====================================================================


async def stream_chat(
    client,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    max_tokens: int,
    timeout: float,
) -> tuple[str, bool]:
    """流式调用, 边收边累计。返回 (raw_output, reasoning_content_seen)。

    不在这里清理 think, 统一在输出结束后清理。
    """
    stream = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    raw_parts: list[str] = []
    reasoning_content_seen = False

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        content = getattr(delta, "content", None)
        if content:
            raw_parts.append(content)

        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            reasoning_content_seen = True

        # 兼容部分服务把 reasoning_content 放在 model_extra
        extra = getattr(delta, "model_extra", None) or {}
        if extra.get("reasoning_content"):
            reasoning_content_seen = True

    return "".join(raw_parts), reasoning_content_seen


# =====================================================================
# 耗时工具
# =====================================================================


def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def make_timing(monotonic_start: float, started_at: str | None = None) -> dict:
    return {
        "started_at": started_at,
        "finished_at": now_iso(),
        "duration_ms": int((time.monotonic() - monotonic_start) * 1000),
    }


# =====================================================================
# 第七部分 + 第十二部分: 推理策略与明细记录
# =====================================================================

STRATEGY_SINGLE = "single_full_question"
STRATEGY_SEQUENTIAL = "sequential_two_part"

TASK_SINGLE = "single_full_question"
TASK_SUBQUESTION = "sequential_two_part_subquestion"
TASK_FINAL = "sequential_two_part_final"


def _model_config_public_view(model_config: dict) -> dict:
    """落盘时只暴露是否设置 key, 不暴露原文。"""
    return {
        "base_url": model_config["base_url"],
        "model_name": model_config["model_name"],
        "max_tokens": model_config["max_tokens"],
        "api_key_set": model_config["api_key_set"],
    }


def _output_block(raw_output: str, reasoning_content_seen: bool) -> dict:
    cleaned, has_think = strip_think_blocks(raw_output)
    return {
        "output": cleaned,
        "output_chars": len(cleaned),
        "raw_output_chars": len(raw_output),
        "think_removed": has_think,
        "reasoning_content_seen": reasoning_content_seen,
    }


def _base_record(
    *,
    run_key: str,
    run_key_components: dict,
    status: str,
    record: dict,
    q_hash: str,
    strategy: str,
    task_type: str,
    prompt_name: str,
    prompt_version: str,
    model_config: dict,
    split_result: dict,
) -> dict:
    return {
        "run_key": run_key,
        "run_key_components": run_key_components,
        "status": status,
        "id": record.get("id"),
        "difficulty": record.get("difficulty"),
        "question_hash": q_hash,
        "strategy": strategy,
        "task_type": task_type,
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "model_config": _model_config_public_view(model_config),
        "question": record.get("question"),
        "reference_answer": record.get("reference_answer"),
        "split_result": split_result,
        "error": None,
    }


async def run_single_full_question(
    *,
    client,
    record: dict,
    q_hash: str,
    split_result: dict,
    model_config: dict,
    store: AppendStore,
    completed: set[str],
    force: bool,
    timeout: float,
) -> dict | None:
    """完整题干一次性推理。返回明细记录 dict (或 None 表示跳过)。"""
    prompt_name, prompt_version = PROMPT_BOXED_BASELINE
    system_prompt = SYSTEM_PROMPT_BOXED
    user_prompt = record.get("question", "")

    components = build_run_key_components(
        record_id=record.get("id"),
        q_hash=q_hash,
        strategy=STRATEGY_SINGLE,
        task_type=TASK_SINGLE,
        subquestion_index=None,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        model_config=model_config,
    )
    run_key = make_run_key(components)

    if not force and run_key in completed:
        logger.info(f"[SKIP] completed run_key={run_key[:16]}... id={record.get('id')}")
        return None

    rec = _base_record(
        run_key=run_key,
        run_key_components=components,
        status="running",
        record=record,
        q_hash=q_hash,
        strategy=STRATEGY_SINGLE,
        task_type=TASK_SINGLE,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        model_config=model_config,
        split_result=split_result,
    )
    rec["input"] = {"system_prompt": system_prompt, "user_prompt": user_prompt}

    mono = time.monotonic()
    started_at = now_iso()
    try:
        raw_output, reasoning_content_seen = await stream_chat(
            client, system_prompt, user_prompt,
            model_config["model_name"], model_config["max_tokens"], timeout,
        )
        rec.update(_output_block(raw_output, reasoning_content_seen))
        rec["timing"] = make_timing(mono, started_at)
        rec["status"] = "completed"
        logger.info(
            f"[DONE] single id={record.get('id')} "
            f"output_chars={rec['output_chars']} duration_ms={rec['timing']['duration_ms']}"
        )
    except Exception as e:
        rec["timing"] = make_timing(mono, started_at)
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        logger.error(f"[ERROR] single id={record.get('id')}: {e}")

    store.append(rec)
    return rec


async def run_sequential_two_part(
    *,
    client,
    record: dict,
    q_hash: str,
    split_result: dict,
    model_config: dict,
    store: AppendStore,
    completed: set[str],
    force: bool,
    timeout: float,
) -> dict | None:
    """两问顺序推理: 第1问 -> 第2问(带第1问完整解答) -> final 合并。"""
    subs = split_result["subquestions"]
    common_stem = split_result["common_stem"]
    if len(subs) != 2:
        # 调用方应保证仅 two_part 进入此函数
        logger.warning(
            f"[WARN] sequential_two_part 进入但 subquestions={len(subs)}, 跳过 id={record.get('id')}"
        )
        return None

    sub_outputs: list[dict] = []  # 每问 {index, run_key, output, output_chars, duration_ms}
    answer_1 = None

    seq_mono = time.monotonic()

    # ---------- 第 1 问 ----------
    sq1 = subs[0]
    sys1, name1, ver1 = prompt_for_subquestion_type(sq1["type"])
    user_prompt_1 = (
        f"公共题干：\n{common_stem}\n\n现在解答第（1）问：\n{sq1['raw_subquestion_text']}"
    )
    comp1 = build_run_key_components(
        record_id=record.get("id"),
        q_hash=q_hash,
        strategy=STRATEGY_SEQUENTIAL,
        task_type=TASK_SUBQUESTION,
        subquestion_index="1",
        prompt_name=name1,
        prompt_version=ver1,
        model_config=model_config,
    )
    rk1 = make_run_key(comp1)

    rec1 = _base_record(
        run_key=rk1,
        run_key_components=comp1,
        status="running",
        record=record,
        q_hash=q_hash,
        strategy=STRATEGY_SEQUENTIAL,
        task_type=TASK_SUBQUESTION,
        prompt_name=name1,
        prompt_version=ver1,
        model_config=model_config,
        split_result=split_result,
    )
    rec1["subquestion_index"] = "1"
    rec1["subquestion_type"] = sq1["type"]
    rec1["previous_outputs_included"] = []
    rec1["input"] = {"system_prompt": sys1, "user_prompt": user_prompt_1}

    mono1 = time.monotonic()
    started1 = now_iso()
    if not force and rk1 in completed:
        logger.info(f"[SKIP] completed run_key={rk1[:16]}... id={record.get('id')} sq=1")
        # 续跑: 从历史明细还原第1问输出
        restored = _restore_subquestion_output(store.path, rk1)
        if restored is None:
            logger.error(
                f"[ERROR] 无法还原第1问历史输出, 终止该题 sequential id={record.get('id')}"
            )
            return None
        answer_1 = restored["output"]
        sub_outputs.append(
            {
                "index": "1",
                "run_key": rk1,
                "output_chars": restored["output_chars"],
                "duration_ms": restored["duration_ms"],
            }
        )
        sq1_duration_ms = restored["duration_ms"]
    else:
        try:
            raw1, rc1 = await stream_chat(
                client, sys1, user_prompt_1,
                model_config["model_name"], model_config["max_tokens"], timeout,
            )
            ob1 = _output_block(raw1, rc1)
            rec1.update(ob1)
            rec1["timing"] = make_timing(mono1, started1)
            rec1["status"] = "completed"
            answer_1 = ob1["output"]
            sq1_duration_ms = rec1["timing"]["duration_ms"]
            sub_outputs.append(
                {
                    "index": "1",
                    "run_key": rk1,
                    "output_chars": ob1["output_chars"],
                    "duration_ms": sq1_duration_ms,
                }
            )
            logger.info(
                f"[DONE] seq sq=1 id={record.get('id')} type={sq1['type']} "
                f"output_chars={ob1['output_chars']} duration_ms={sq1_duration_ms}"
            )
        except Exception as e:
            rec1["timing"] = make_timing(mono1, started1)
            rec1["status"] = "error"
            rec1["error"] = f"{type(e).__name__}: {e}"
            store.append(rec1)
            logger.error(f"[ERROR] seq sq=1 id={record.get('id')}: {e}")
            return None
        store.append(rec1)

    # ---------- 第 2 问 (带第1问完整解答过程) ----------
    sq2 = subs[1]
    sys2, name2, ver2 = prompt_for_subquestion_type(sq2["type"])
    user_prompt_2 = (
        f"公共题干：\n{common_stem}\n\n"
        f"第（1）问的完整解答过程如下：\n{answer_1}\n\n"
        f"现在解答第（2）问：\n{sq2['raw_subquestion_text']}"
    )
    comp2 = build_run_key_components(
        record_id=record.get("id"),
        q_hash=q_hash,
        strategy=STRATEGY_SEQUENTIAL,
        task_type=TASK_SUBQUESTION,
        subquestion_index="2",
        prompt_name=name2,
        prompt_version=ver2,
        model_config=model_config,
    )
    rk2 = make_run_key(comp2)

    rec2 = _base_record(
        run_key=rk2,
        run_key_components=comp2,
        status="running",
        record=record,
        q_hash=q_hash,
        strategy=STRATEGY_SEQUENTIAL,
        task_type=TASK_SUBQUESTION,
        prompt_name=name2,
        prompt_version=ver2,
        model_config=model_config,
        split_result=split_result,
    )
    rec2["subquestion_index"] = "2"
    rec2["subquestion_type"] = sq2["type"]
    rec2["previous_outputs_included"] = ["1"]
    rec2["input"] = {"system_prompt": sys2, "user_prompt": user_prompt_2}

    mono2 = time.monotonic()
    started2 = now_iso()
    answer_2 = None
    if not force and rk2 in completed:
        logger.info(f"[SKIP] completed run_key={rk2[:16]}... id={record.get('id')} sq=2")
        restored2 = _restore_subquestion_output(store.path, rk2)
        if restored2 is None:
            logger.error(
                f"[ERROR] 无法还原第2问历史输出, 终止该题 sequential id={record.get('id')}"
            )
            return None
        answer_2 = restored2["output"]
        sq2_duration_ms = restored2["duration_ms"]
        sub_outputs.append(
            {
                "index": "2",
                "run_key": rk2,
                "output_chars": restored2["output_chars"],
                "duration_ms": sq2_duration_ms,
            }
        )
    else:
        try:
            raw2, rc2 = await stream_chat(
                client, sys2, user_prompt_2,
                model_config["model_name"], model_config["max_tokens"], timeout,
            )
            ob2 = _output_block(raw2, rc2)
            rec2.update(ob2)
            rec2["timing"] = make_timing(mono2, started2)
            rec2["status"] = "completed"
            answer_2 = ob2["output"]
            sq2_duration_ms = rec2["timing"]["duration_ms"]
            sub_outputs.append(
                {
                    "index": "2",
                    "run_key": rk2,
                    "output_chars": ob2["output_chars"],
                    "duration_ms": sq2_duration_ms,
                }
            )
            logger.info(
                f"[DONE] seq sq=2 id={record.get('id')} type={sq2['type']} "
                f"output_chars={ob2['output_chars']} duration_ms={sq2_duration_ms}"
            )
        except Exception as e:
            rec2["timing"] = make_timing(mono2, started2)
            rec2["status"] = "error"
            rec2["error"] = f"{type(e).__name__}: {e}"
            store.append(rec2)
            logger.error(f"[ERROR] seq sq=2 id={record.get('id')}: {e}")
            return None
        store.append(rec2)

    # ---------- final 合并 (不调用模型) ----------
    final_components = build_run_key_components(
        record_id=record.get("id"),
        q_hash=q_hash,
        strategy=STRATEGY_SEQUENTIAL,
        task_type=TASK_FINAL,
        subquestion_index=None,
        prompt_name="sequential_two_part_final",
        prompt_version="v1",
        model_config=model_config,
    )
    rk_final = make_run_key(final_components)
    final_output = f"（1）\n{answer_1}\n\n（2）\n{answer_2}"

    final_rec = {
        "run_key": rk_final,
        "run_key_components": final_components,
        "status": "completed",
        "id": record.get("id"),
        "difficulty": record.get("difficulty"),
        "question_hash": q_hash,
        "strategy": STRATEGY_SEQUENTIAL,
        "task_type": TASK_FINAL,
        "prompt_name": "sequential_two_part_final",
        "prompt_version": "v1",
        "model_config": _model_config_public_view(model_config),
        "final_output": final_output,
        "subquestion_outputs": sub_outputs,
        "timing": {
            "duration_ms": int((time.monotonic() - seq_mono) * 1000),
            "subquestion_1_duration_ms": sub_outputs[0]["duration_ms"],
            "subquestion_2_duration_ms": sub_outputs[1]["duration_ms"],
        },
        "error": None,
    }

    if not (force or rk_final in completed):
        store.append(final_rec)
    elif rk_final in completed and not force:
        logger.info(f"[SKIP] completed run_key={rk_final[:16]}... (final) id={record.get('id')}")
    else:
        store.append(final_rec)

    logger.info(
        f"[DONE] seq final id={record.get('id')} "
        f"final_chars={len(final_output)} duration_ms={final_rec['timing']['duration_ms']}"
    )
    return final_rec


def _restore_subquestion_output(store_path: Path, run_key: str) -> dict | None:
    """断点续跑: 从明细 JSONL 中按 run_key 还原某小问的 output。"""
    if not store_path.exists():
        return None
    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("run_key") == run_key and obj.get("status") == "completed":
                timing = obj.get("timing", {}) or {}
                return {
                    "output": obj.get("output", ""),
                    "output_chars": obj.get("output_chars", 0),
                    "duration_ms": timing.get("duration_ms", 0),
                }
    return None


# =====================================================================
# 第十三部分: merged 聚合
# =====================================================================


def build_merged(store_path: str, merged_path: str, sample_records: list[dict]) -> None:
    """根据明细 JSONL 生成 per-question 聚合结果。

    同一 (id, strategy, task_type, subquestion_index) 保留最后一条 completed
    (兼容 --force 重复跑)。
    """
    # 读明细, 索引到 per-id 的最新 completed 记录
    latest: dict[tuple, dict] = {}
    order: list[str] = []
    if os.path.exists(store_path):
        with open(store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("status") != "completed":
                    continue
                key = (
                    obj.get("id"),
                    obj.get("strategy"),
                    obj.get("task_type"),
                    obj.get("subquestion_index"),
                )
                latest[key] = obj
                if obj.get("id") not in order:
                    order.append(obj.get("id"))

    # 按 sample 顺序输出 (保证可读); 明细里有但 sample 没有的也补在后面
    sample_ids = [r.get("id") for r in sample_records]
    out_ids = [i for i in sample_ids if i is not None]
    for i in order:
        if i not in out_ids:
            out_ids.append(i)

    Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(merged_path, "w", encoding="utf-8") as f:
        for qid in out_ids:
            # 取该 id 的 question / reference_answer / split_result (任一记录即可)
            meta = next(
                (
                    latest[k]
                    for k in latest
                    if k[0] == qid
                ),
                None,
            )
            if meta is None:
                continue

            results = []

            # single_full_question
            single_key = (qid, STRATEGY_SINGLE, TASK_SINGLE, None)
            if single_key in latest:
                s = latest[single_key]
                results.append(
                    {
                        "strategy": STRATEGY_SINGLE,
                        "output": s.get("output"),
                        "duration_ms": (s.get("timing") or {}).get("duration_ms"),
                        "error": s.get("error"),
                    }
                )

            # sequential_two_part
            sq1_key = (qid, STRATEGY_SEQUENTIAL, TASK_SUBQUESTION, "1")
            sq2_key = (qid, STRATEGY_SEQUENTIAL, TASK_SUBQUESTION, "2")
            final_key = (qid, STRATEGY_SEQUENTIAL, TASK_FINAL, None)
            if final_key in latest:
                seq_entry: dict = {"strategy": STRATEGY_SEQUENTIAL, "runs": []}
                for sq_key, idx in ((sq1_key, "1"), (sq2_key, "2")):
                    if sq_key in latest:
                        sq = latest[sq_key]
                        seq_entry["runs"].append(
                            {
                                "subquestion_index": idx,
                                "subquestion_type": sq.get("subquestion_type"),
                                "output": sq.get("output"),
                                "duration_ms": (sq.get("timing") or {}).get("duration_ms"),
                            }
                        )
                fin = latest[final_key]
                seq_entry["final_output"] = fin.get("final_output")
                seq_entry["duration_ms"] = (fin.get("timing") or {}).get("duration_ms")
                seq_entry["error"] = fin.get("error")
                results.append(seq_entry)

            merged_obj = {
                "id": qid,
                "difficulty": meta.get("difficulty"),
                "question": meta.get("question"),
                "reference_answer": meta.get("reference_answer"),
                "split_result": meta.get("split_result", {}),
                "results": results,
            }
            f.write(json.dumps(merged_obj, ensure_ascii=False) + "\n")
            n += 1

    logger.info(f"写入 merged: {merged_path} 共 {n} 题")


# =====================================================================
# split-only 预览
# =====================================================================


def do_split_only(input_path: str, limit: int | None, ids: list[str]) -> int:
    records = _read_sample(input_path)
    if ids:
        id_set = set(ids)
        records = [r for r in records if r.get("id") in id_set]
    if limit is not None:
        records = records[:limit]

    status_counter: dict[str, int] = {}
    print(f"{'idx':>4}  {'id':<12} {'difficulty':<6} {'status':<36} markers")
    for i, rec in enumerate(records, 1):
        sr = split_question(rec.get("question", ""))
        status = sr["split_status"]
        status_counter[status] = status_counter.get(status, 0) + 1
        print(
            f"{i:>4}  {str(rec.get('id')):<12} "
            f"{str(rec.get('difficulty')):<6} {status:<36} {sr['matched_marker_count']}"
        )
    print("\n=== split_status 汇总 ===")
    for k, v in sorted(status_counter.items()):
        print(f"  {k}: {v}")
    return 0


# =====================================================================
# 工具: 读缓存样本
# =====================================================================


def _read_sample(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# =====================================================================
# 主流程
# =====================================================================


async def run_eval(args: argparse.Namespace) -> int:
    # --merge-only: 不调用模型, 只生成 merged
    if args.merge_only:
        sample = _read_sample(args.input)
        merged_path = args.merged_output or _default_merged_path(args.output)
        build_merged(args.output, merged_path, sample)
        return 0

    sample = _read_sample(args.input)

    # --id 过滤
    if args.id:
        id_set = set(args.id)
        sample = [r for r in sample if r.get("id") in id_set]
    if args.limit is not None:
        sample = sample[:args.limit]

    # 校验模型配置 (实际需要调用模型时)
    try:
        model_config = get_model_config()
    except ModelConfigError as e:
        logger.error(f"模型配置错误: {e}")
        return 2
    logger.info(
        f"模型配置: base_url={model_config['base_url']} "
        f"model={model_config['model_name']} max_tokens={model_config['max_tokens']} "
        f"api_key_set={model_config['api_key_set']}"
    )

    # 断点续跑: 读取已完成 run_key
    completed: set[str] = set()
    if args.resume and not args.force:
        completed = load_completed_run_keys(args.output)
        logger.info(f"断点续跑: 已完成 {len(completed)} 个最小任务, 将跳过")

    # 创建 OpenAI 兼容流式客户端
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=model_config["base_url"],
        api_key=model_config["api_key"],
    )

    store = AppendStore(args.output)

    total = len(sample)
    for idx, rec in enumerate(sample, 1):
        qid = rec.get("id")
        question = rec.get("question", "")
        q_hash = question_hash(question)
        split_result = split_question(question)
        status = split_result["split_status"]

        logger.info(
            f"=== [{idx}/{total}] id={qid} difficulty={rec.get('difficulty')} "
            f"split_status={status} markers={split_result['matched_marker_count']} ==="
        )

        if status == "two_part":
            # 1) baseline: 完整题干一次性推理
            await run_single_full_question(
                client=client, record=rec, q_hash=q_hash, split_result=split_result,
                model_config=model_config, store=store, completed=completed,
                force=args.force, timeout=args.timeout,
            )
            # 2) 两问顺序推理
            await run_sequential_two_part(
                client=client, record=rec, q_hash=q_hash, split_result=split_result,
                model_config=model_config, store=store, completed=completed,
                force=args.force, timeout=args.timeout,
            )
        else:
            # single 或 fallback_single_due_marker_count: 仅完整题干单次推理
            await run_single_full_question(
                client=client, record=rec, q_hash=q_hash, split_result=split_result,
                model_config=model_config, store=store, completed=completed,
                force=args.force, timeout=args.timeout,
            )

    store.close()

    # 生成 merged
    merged_path = args.merged_output or _default_merged_path(args.output)
    build_merged(args.output, merged_path, sample)
    logger.info(f"评测完成。明细: {args.output}  聚合: {merged_path}")
    return 0


def _default_merged_path(output_path: str) -> str:
    p = Path(output_path)
    if p.suffix == ".jsonl":
        return str(p.with_suffix("")) + ".merged.jsonl"
    return output_path + ".merged.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="数学提示词离线评测 (串行)")
    parser.add_argument("--input", required=True, help="缓存样本 JSONL, 必填")
    parser.add_argument("--output", required=True, help="append-only 明细 JSONL, 必填")
    parser.add_argument("--merged-output", default=None, help="聚合结果 JSONL, 可选")
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="默认启用: 读取历史 output, 跳过已完成任务",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="不读取历史 output",
    )
    parser.add_argument("--force", action="store_true", help="忽略已有 completed, 重新跑")
    parser.add_argument("--limit", type=int, default=None, help="限制处理条数")
    parser.add_argument("--id", action="append", default=None, help="只跑指定题目 id, 可重复传")
    parser.add_argument("--split-only", action="store_true", help="只拆题, 不调用模型")
    parser.add_argument("--merge-only", action="store_true", help="只根据 output 生成 merged-output")
    parser.add_argument("--timeout", type=float, default=600.0, help="单次模型请求超时(秒), 默认 600")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    load_env_if_present()

    if args.split_only:
        return do_split_only(args.input, args.limit, args.id or [])

    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return 2

    try:
        return asyncio.run(run_eval(args))
    except KeyboardInterrupt:
        logger.warning("已中断 (Ctrl+C)")
        return 130


if __name__ == "__main__":
    sys.exit(main())
