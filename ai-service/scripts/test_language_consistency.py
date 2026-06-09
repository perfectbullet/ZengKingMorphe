#!/usr/bin/env python3
"""
回答语言与本轮查询语言一致性回归测试。

复现用户报告的场景：在同一 session 内交替提问中英文，
验证“本轮回复语言只跟当前问句语言相关，不被历史语种带偏”。

预期：
- EN 问 → EN 答
- ZH 问 → ZH 答
- 任何方向，body 不出现“另一语种主导”或“中英混合”输出
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

import requests


def detect_lang(text: str) -> str:
    """body 主导语言检测：'zh' / 'en' / 'mixed' / 'unknown'。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cjk == 0 and latin == 0:
        return "unknown"
    if cjk > 0 and latin == 0:
        return "zh"
    if latin > 0 and cjk == 0:
        return "en"
    # 同时含两种字符 → 检查是否“显著混合”
    # 阈值：另一语种字符数占主语种字符数 30% 以上即认为是混合污染。
    if cjk >= latin:
        ratio = latin / max(cjk, 1)
        return "mixed" if ratio > 0.30 else "zh"
    else:
        ratio = cjk / max(latin, 1)
        return "mixed" if ratio > 0.10 else "en"


# preface 由本服务在 chat_stream_v1.py 注入（"Got it—let me think..." / "好的，我正在梳理..."），
# 它本身已按 prefer_zh_output 选择语种，不参与 LLM 输出，需要在判定前剥离。
_PREFACE_PATTERNS = [
    r"^Got it[^\n]*\n+",
    r"^One sec[^\n]*\n+",
    r"^好的，我正在梳理[^\n]*\n+",
    r"^等我一小下下[^\n]*\n+",
]


def strip_preface(text: str) -> str:
    out = text
    for p in _PREFACE_PATTERNS:
        out = re.sub(p, "", out, count=1)
    return out.strip()


def run_query(host: str, session_id: str, query: str, timeout: int = 120) -> str:
    url = host.rstrip("/") + "/api/chat/v1/chat/completions"
    body = {
        "model": "qwen3:14b",
        "messages": [{"role": "user", "content": query}],
        "stream": True,
        "employee_id": "68",
        "user_id": "2373824",
        "session_id": session_id,
        "team_id": "4",
    }
    full = ""
    with requests.post(url, json=body, stream=True, timeout=(5, timeout)) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if not line:
                continue
            if line == "[DONE]":
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("object") != "chat.completion.chunk":
                continue
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    full += content
                if choice.get("finish_reason") == "stop":
                    return full
    return full


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8100")
    parser.add_argument("--session-id", default=None,
                        help="会话 ID（默认每次随机，避免历史污染）")
    parser.add_argument("--repeat", type=int, default=1, help="整体跑几遍")
    args = parser.parse_args()

    test_plan = [
        ("Q1 EN", "what is the most popular hobby among young people", "en"),
        ("Q2 EN", "what is the most popular hobby among young people", "en"),
        ("Q3 ZH", "我想了解一下今天成都堵不堵？", "zh"),
        ("Q4 EN", "what is the most popular hobby among young people", "en"),
        ("Q5 EN", "what is the most popular hobby among young people", "en"),
    ]

    total_fail = 0
    for r_idx in range(args.repeat):
        sid = args.session_id or f"sess_lang_test_{int(time.time())}_{r_idx}"
        print(f"\n========== ROUND {r_idx + 1}/{args.repeat}  session_id={sid} ==========")
        for tag, query, expected in test_plan:
            print(f"\n--- {tag} | expected={expected} | query={query!r} ---")
            try:
                raw = run_query(args.host, sid, query)
            except Exception as e:
                print(f"  ERROR: {e}")
                total_fail += 1
                continue
            body = strip_preface(raw)
            lang = detect_lang(body)
            ok = (lang == expected)
            status = "PASS" if ok else "FAIL"
            if not ok:
                total_fail += 1
            preview = body.replace("\n", " ")[:240]
            print(f"  [{status}] detected={lang}")
            print(f"  body: {preview}")

    print()
    print("=" * 60)
    print(f"FINAL: total_fail={total_fail}")
    print("=" * 60)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
