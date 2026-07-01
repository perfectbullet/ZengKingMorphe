#!/usr/bin/env python3
"""Independent local review tool for math concept JSONL data."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel


REQUIRED_SOURCE_TYPE = "manual_math_concept"
# None 表示未复核（默认）；correct/incorrect 表示人工判定结果。
REVIEW_STATUSES = {None, "correct", "incorrect"}
DEFAULT_REVIEW_STATUS = None
FILTER_STATUSES = {"all", "unreviewed", "correct", "incorrect"}
DEFAULT_JSONL_PATH = Path(
    "ai-service/data/math_concepts/"
    "05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl"
)


def load_records(jsonl_path: Path) -> list[dict]:
    """逐行读取 JSONL，跳过空白行；解析失败或非 JSON 对象时抛出带行号的 ValueError。

    不自动改写 md_content、不删除已知字段；仅剔除运行时注入的 ``index`` 字段；
    并为缺少 ``review_status`` 的记录补默认值 ``None``（未复核）。
    """
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as source:
        for line_no, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_no} 行解析失败: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 第 {line_no} 行必须是 JSON 对象")
            record.pop("index", None)
            record.setdefault("review_status", DEFAULT_REVIEW_STATUS)
            records.append(record)
    return records


def ensure_backup_once(jsonl_path: Path) -> None:
    """在原文件旁创建一次 ``<file>.bak``，已存在则不覆盖，写一半失败会清理坏文件。"""
    backup_path = Path(f"{jsonl_path}.bak")
    try:
        backup_fd = os.open(
            backup_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o666,
        )
    except FileExistsError:
        return

    try:
        with os.fdopen(backup_fd, "wb") as backup:
            with jsonl_path.open("rb") as source:
                shutil.copyfileobj(source, backup)
            backup.flush()
            os.fsync(backup.fileno())
    except Exception:
        backup_path.unlink(missing_ok=True)
        raise


def save_records_atomic(jsonl_path: Path, records: list[dict]) -> None:
    """先备份一次，再原子替换 JSONL；保持一行一个 JSON、中文不转义、顺序与未知字段不变。"""
    ensure_backup_once(jsonl_path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=jsonl_path.parent,
            prefix=f".{jsonl_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for record in records:
                persisted_record = {
                    key: value for key, value in record.items() if key != "index"
                }
                temporary.write(
                    json.dumps(persisted_record, ensure_ascii=False) + "\n"
                )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, jsonl_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def validate_records(records: list[dict]) -> dict:
    """校验全部记录，返回 ``{ok, total, errors}``。"""
    errors: list[dict] = []
    seen_doc_ids: dict[str, int] = {}
    for index, record in enumerate(records):
        for field in ("doc_id", "concept_name", "md_content"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    {
                        "index": index,
                        "field": field,
                        "message": f"{field} 不能为空",
                    }
                )
        if record.get("source_type") != REQUIRED_SOURCE_TYPE:
            errors.append(
                {
                    "index": index,
                    "field": "source_type",
                    "message": f"source_type 必须为 {REQUIRED_SOURCE_TYPE}",
                }
            )
        if record.get("review_status") not in REVIEW_STATUSES:
            errors.append(
                {
                    "index": index,
                    "field": "review_status",
                    "message": "review_status 必须为 null、correct 或 incorrect",
                }
            )
        doc_id = record.get("doc_id")
        if isinstance(doc_id, str) and doc_id.strip():
            if doc_id in seen_doc_ids:
                errors.append(
                    {
                        "index": index,
                        "field": "doc_id",
                        "message": (
                            f"doc_id 重复: {doc_id}"
                            f"（首次出现在第 {seen_doc_ids[doc_id]} 条）"
                        ),
                    }
                )
            else:
                seen_doc_ids[doc_id] = index
    return {"ok": not errors, "total": len(records), "errors": errors}


def build_stats(records: list[dict]) -> dict:
    """统计各复核状态的数量，供工具栏展示。"""
    return {
        "total": len(records),
        "unreviewed": sum(r.get("review_status") is None for r in records),
        "correct": sum(r.get("review_status") == "correct" for r in records),
        "incorrect": sum(r.get("review_status") == "incorrect" for r in records),
    }


def filter_records(records: list[dict], status: str) -> list[dict]:
    """按 UI 状态过滤记录，返回值为 ``records`` 的子集（同对象引用，保留原 index 定位）。"""
    if status not in FILTER_STATUSES:
        raise ValueError(f"不支持的过滤状态: {status}")
    if status == "all":
        return list(records)
    if status == "unreviewed":
        return [record for record in records if record.get("review_status") is None]
    return [record for record in records if record.get("review_status") == status]


def to_lightweight(record: dict, index: int) -> dict:
    """生成轻量记录摘要，供列表接口使用（不含完整 md_content）。"""
    md_content = record.get("md_content", "")
    if not isinstance(md_content, str):
        md_content = str(md_content)
    return {
        "index": index,
        "doc_id": record.get("doc_id"),
        "concept_name": record.get("concept_name"),
        "source_type": record.get("source_type"),
        "review_status": record.get("review_status"),
        "content_chars": len(md_content),
        "content_preview": md_content[:120],
    }


class RecordInput(BaseModel):
    doc_id: str
    concept_name: str
    md_content: str
    source_type: str = REQUIRED_SOURCE_TYPE
    review_status: str | None = DEFAULT_REVIEW_STATUS


_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>数学概念 JSONL 人工复核</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; color: #17202a; background: #f4f6f7; }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body { display: flex; flex-direction: column; margin: 0; overflow: hidden; }
    header { flex: none; padding: 14px 22px; color: white; background: #1f3a4d; }
    header h1 { margin: 0 0 5px; font-size: 19px; }
    header p { margin: 0; overflow-wrap: anywhere; color: #d6e4ec; font-size: 12px; }
    .toolbar { flex: none; display: flex; flex-wrap: wrap; align-items: center; gap: 18px; padding: 10px 22px; background: #e9eef1; border-bottom: 1px solid #c9d2d8; font-size: 13px; color: #52636d; }
    .toolbar b { font-variant-numeric: tabular-nums; color: #253943; }
    .toolbar .spacer { margin-left: auto; color: #41525b; }
    .app-layout { display: flex; flex: 1; min-height: 0; }
    .sidebar { display: flex; flex: 0 0 380px; flex-direction: column; min-width: 0; border-right: 1px solid #c9d2d8; background: white; }
    .sidebar-controls { flex: none; padding: 12px 14px; border-bottom: 1px solid #dce3e7; }
    .sidebar-controls input { width: 100%; min-height: 34px; padding: 7px 10px; border: 1px solid #9aa8b0; border-radius: 6px; font: inherit; }
    .sidebar-controls input:focus { border-color: #28789b; outline: 2px solid #d9edf5; }
    .record-list { flex: 1; min-height: 0; overflow-y: auto; padding: 8px; }
    .record-item { display: block; width: 100%; margin: 0 0 7px; padding: 10px 12px; overflow: hidden; text-align: left; border: 1px solid #d8e0e4; border-left: 4px solid transparent; border-radius: 6px; background: white; cursor: pointer; }
    .record-item:hover { border-color: #9db7c5; background: #f3f8fa; }
    .record-item.active { border-color: #8bb3c7; border-left-color: #28789b; background: #eaf5fa; }
    .record-title { display: flex; align-items: center; gap: 7px; min-width: 0; margin-bottom: 4px; font-size: 12px; color: #52636d; }
    .record-number { font-weight: 700; color: #253943; }
    .record-meta { color: #71818a; font-size: 11px; }
    .record-concept { font-size: 14px; font-weight: 600; color: #17202a; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .record-docid { color: #28708f; font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .record-preview { display: -webkit-box; overflow: hidden; color: #41525b; font-size: 12px; line-height: 1.5; overflow-wrap: anywhere; -webkit-box-orient: vertical; -webkit-line-clamp: 2; margin-top: 4px; }
    .empty-list { padding: 24px 12px; text-align: center; color: #71818a; font-size: 13px; }
    .main-panel { flex: 1; min-width: 0; overflow-y: auto; padding: 18px 22px 30px; }
    .detail-header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 14px; }
    .position { color: #52636d; font-size: 13px; font-variant-numeric: tabular-nums; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 14px; margin-bottom: 14px; }
    .field { display: flex; flex-direction: column; gap: 5px; }
    .field.full { grid-column: 1 / -1; }
    .field label { color: #344952; font-size: 12px; font-weight: 700; }
    .field input { min-height: 36px; padding: 7px 10px; border: 1px solid #9aa8b0; border-radius: 6px; font: inherit; }
    .field input:focus, textarea:focus { border-color: #28789b; outline: 2px solid #d9edf5; }
    .field .hint { color: #71818a; font-size: 11px; }
    .editor-section { padding: 14px; border: 1px solid #ced7dc; border-radius: 8px; background: white; }
    .editor-section h2 { margin: 0 0 4px; font-size: 14px; }
    .editor-section .count { color: #71818a; font-size: 11px; margin: 0 0 6px; }
    textarea { width: 100%; min-height: 360px; resize: vertical; padding: 11px; border: 1px solid #9aa8b0; border-radius: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; line-height: 1.6; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    button { min-height: 36px; padding: 7px 14px; border: 1px solid #9aa8b0; border-radius: 6px; background: white; cursor: pointer; font: inherit; }
    button:hover { background: #edf3f6; }
    button.primary { color: white; border-color: #176b4d; background: #176b4d; }
    button:disabled { opacity: .45; cursor: default; }
    #message { min-height: 22px; margin: 12px 0 0; color: #176b4d; font-size: 13px; }
    #message.error { color: #a93226; }
    #validation { margin: 8px 0 0; padding: 0; white-space: pre-wrap; overflow-wrap: anywhere; color: #a93226; font-size: 12px; line-height: 1.6; }
    #validation.ok { color: #176b4d; }
    @media (max-width: 900px) { .sidebar { flex-basis: 320px; } .form-grid { grid-template-columns: 1fr; } .field.full { grid-column: auto; } }
    @media (max-width: 680px) { body { overflow: auto; } .app-layout { display: block; } .sidebar { width: 100%; height: 45vh; border-right: 0; border-bottom: 1px solid #c9d2d8; } .main-panel { overflow: visible; padding: 14px; } }
    .md-workspace { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; align-items: stretch; margin-top: 12px; }
    .md-editor-panel, .md-preview-panel { min-width: 0; border: 1px solid #ced7dc; border-radius: 8px; background: white; padding: 14px; }
    .section-title-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
    .section-title-row h2 { margin: 0; font-size: 15px; }
    .md-editor-panel textarea { width: 100%; min-height: 520px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; line-height: 1.65; }
    .preview-hint { color: #71818a; font-size: 12px; }
    .markdown-preview { min-height: 520px; max-height: calc(100vh - 360px); overflow: auto; padding: 14px; border: 1px solid #d8e0e4; border-radius: 6px; background: #fff; line-height: 1.8; overflow-wrap: anywhere; white-space: pre-wrap; }
    .markdown-preview h1, .markdown-preview h2, .markdown-preview h3 { margin-top: 1em; margin-bottom: .6em; }
    .markdown-preview p { margin: .65em 0; }
    .markdown-preview ul, .markdown-preview ol { margin: .5em 0; padding-left: 1.6em; }
    .markdown-preview table { border-collapse: collapse; width: 100%; margin: 12px 0; }
    .markdown-preview th, .markdown-preview td { border: 1px solid #cbd5dc; padding: 6px 8px; vertical-align: top; }
    .markdown-preview pre { overflow: auto; padding: 10px; border-radius: 6px; background: #f4f6f7; }
    .markdown-preview code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .markdown-preview img { max-width: 100%; height: auto; }
    .markdown-preview .preview-error { color: #a93226; white-space: pre-wrap; }
    .katex-display { overflow-x: auto; overflow-y: hidden; padding: .25em 0; }
    .status-badge { display: inline-flex; align-items: center; min-height: 22px; padding: 2px 9px; border: 1px solid transparent; border-radius: 999px; white-space: nowrap; font-size: 12px; font-variant-numeric: tabular-nums; }
    .status-badge.unreviewed { color: #435661; border-color: #c6d2d8; background: #edf2f4; }
    .status-badge.correct { color: #176b4d; border-color: #a9d5c2; background: #e7f5ee; }
    .status-badge.incorrect { color: #a93226; border-color: #e3b5b0; background: #faecea; }
    .filter-group { display: inline-flex; gap: 4px; }
    .filter-group button { min-height: 28px; padding: 3px 10px; font-size: 12px; }
    .filter-group button.active { color: white; border-color: #276678; background: #276678; }
    .status-group { display: inline-flex; gap: 4px; padding-left: 6px; margin-left: 6px; border-left: 1px solid #c9d2d8; }
    .status-group button { min-height: 36px; padding: 7px 12px; }
    .status-group button[data-status="correct"] { color: #176b4d; border-color: #a9d5c2; background: #e7f5ee; }
    .status-group button[data-status="incorrect"] { color: #a93226; border-color: #e3b5b0; background: #faecea; }
    .status-group button.active { outline: 2px solid #28789b; outline-offset: 1px; }
    .record-list .status-badge { margin-left: auto; }
    @media (max-width: 1100px) { .md-workspace { grid-template-columns: 1fr; } .markdown-preview { max-height: none; } }
  </style>
</head>
<body>
  <header>
    <h1>数学概念 JSONL 人工复核</h1>
    <p>当前文件：__FILE_PATH__</p>
  </header>
  <div class="toolbar">
    <span>总数：<b id="total">0</b></span>
    <span>未复核：<b id="stat-unreviewed">0</b></span>
    <span>正确：<b id="stat-correct">0</b></span>
    <span>错误：<b id="stat-incorrect">0</b></span>
    <span class="filter-group" role="group">
      <button type="button" data-filter="all" class="active">全部</button>
      <button type="button" data-filter="unreviewed">未复核</button>
      <button type="button" data-filter="correct">正确</button>
      <button type="button" data-filter="incorrect">错误</button>
    </span>
    <span>位置：<b id="position">- / -</b></span>
    <span id="status-line" class="spacer">就绪</span>
  </div>
  <div class="app-layout">
    <aside class="sidebar">
      <div class="sidebar-controls">
        <input id="search" type="search" placeholder="搜索 concept_name / doc_id / 预览">
      </div>
      <div id="record-list" class="record-list"></div>
    </aside>
    <main class="main-panel">
      <div class="detail-header">
        <span class="position">当前编辑记录</span>
        <span id="current-status" class="status-badge unreviewed">未复核</span>
        <b id="position-detail" class="position">- / -</b>
      </div>
      <div class="form-grid">
        <div class="field">
          <label for="doc_id">doc_id</label>
          <input id="doc_id" type="text">
        </div>
        <div class="field">
          <label for="concept_name">concept_name</label>
          <input id="concept_name" type="text">
        </div>
        <div class="field full">
          <label for="source_type">source_type</label>
          <input id="source_type" type="text">
          <span class="hint">必须为 manual_math_concept</span>
        </div>
      </div>
      <div class="md-workspace">
        <section class="md-editor-panel">
          <div class="section-title-row">
            <h2>md_content 编辑</h2>
            <span id="md-count">字符数：0</span>
          </div>
          <textarea id="md_content" spellcheck="false"></textarea>
        </section>
        <section class="md-preview-panel">
          <div class="section-title-row">
            <h2>md_content 渲染预览</h2>
            <span class="preview-hint">Markdown + LaTeX</span>
          </div>
          <div id="md-content-preview" class="markdown-preview"></div>
        </section>
      </div>
      <div class="actions">
        <button id="save" class="primary">保存（Ctrl+S）</button>
        <button id="prev">上一条（Alt+←）</button>
        <button id="next">下一条（Alt+→）</button>
        <span class="status-group" id="status-group">
          <button type="button" data-status="" title="标记为未复核并保存，跳到下一条">未复核</button>
          <button type="button" data-status="correct" title="标记为正确并保存，跳到下一条">✓ 正确</button>
          <button type="button" data-status="incorrect" title="标记为错误并保存，跳到下一条">✗ 错误</button>
        </span>
        <button id="create-btn">新增记录</button>
        <button id="create-submit" class="primary" hidden>创建</button>
        <button id="create-cancel" hidden>取消</button>
        <button id="validate">校验</button>
        <button id="export">下载 JSONL</button>
      </div>
      <p id="message"></p>
      <pre id="validation"></pre>
    </main>
  </div>
  <script>
    const state = { records: [], total: 0, currentIndex: null, loaded: null, search: "", filter: "all", creating: false };
    const $ = (id) => document.getElementById(id);
    function normalizeStatus(value) { return value === "correct" || value === "incorrect" ? value : null; }
    function statusLabel(value) { return value === "correct" ? "正确" : value === "incorrect" ? "错误" : "未复核"; }
    function statusClass(value) { return value === "correct" ? "correct" : value === "incorrect" ? "incorrect" : "unreviewed"; }

    function setMessage(text, isError) {
      const el = $("message");
      el.textContent = text;
      el.classList.toggle("error", !!isError);
    }
    function setValidation(result) {
      const el = $("validation");
      if (!result) { el.textContent = ""; el.className = ""; return; }
      el.className = result.ok ? "ok" : "";
      if (result.ok) {
        el.textContent = "校验通过：共 " + result.total + " 条记录";
      } else {
        const lines = result.errors.map(e => "#" + e.index + " [" + e.field + "] " + e.message);
        el.textContent = "校验未通过：" + result.errors.length + " 个问题（共 " + result.total + " 条）\\n" + lines.join("\\n");
      }
    }
    function isDirty() {
      if (!state.loaded) return false;
      return $("doc_id").value !== state.loaded.doc_id
          || $("concept_name").value !== state.loaded.concept_name
          || $("source_type").value !== state.loaded.source_type
          || $("md_content").value !== state.loaded.md_content;
    }
    function confirmSwitch() {
      if (state.creating && createHasContent()) {
        return window.confirm("当前正在新建记录且已填写内容，确定要放弃吗？");
      }
      return !isDirty() || window.confirm("当前记录有未保存修改，确定要切换吗？");
    }
    function currentPosition() {
      return state.records.findIndex(r => r.index === state.currentIndex);
    }
    function updateMdCount() {
      $("md-count").textContent = "字符数：" + $("md_content").value.length;
    }
    function renderMarkdownPreview() {
      const textarea = document.getElementById("md_content");
      const preview = document.getElementById("md-content-preview");
      if (!textarea || !preview) return;
      const backslash = String.fromCharCode(92);
      // 参考 tools/asr_latex_review/app.py 的 renderPreview：把原始文本直接放入
      // textContent，再让 KaTeX auto-render 扫描定界符。不走 Markdown 解析，避免
      // 跨行块级公式 $$...$$ 被破坏（之前的根因：行内 $m$ 单行可渲染，但 $$...$$
      // 跨行会被 Markdown 解析器拆分/转义，导致 auto-render 匹配失败）。
      preview.textContent = textarea.value || "";
      if (window.renderMathInElement) {
        renderMathInElement(preview, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: backslash + "[", right: backslash + "]", display: true },
            { left: "$", right: "$", display: false },
            { left: backslash + "(", right: backslash + ")", display: false }
          ],
          throwOnError: false
        });
      }
    }
    function updatePosition() {
      if (state.creating) {
        $("position").textContent = "新建 / " + (state.records.length || "-");
        $("position-detail").textContent = "新建记录";
        return;
      }
      const pos = currentPosition();
      const text = (pos >= 0 ? (pos + 1) : "-") + " / " + (state.records.length || "-");
      $("position").textContent = text;
      $("position-detail").textContent = text;
    }
    function renderList() {
      const list = $("record-list");
      list.replaceChildren();
      const q = state.search.trim().toLowerCase();
      const filtered = q ? state.records.filter(r => {
        const docId = String(r.doc_id || "").toLowerCase();
        const name = String(r.concept_name || "").toLowerCase();
        const preview = String(r.content_preview || "").toLowerCase();
        return name.includes(q) || docId.includes(q) || preview.includes(q);
      }) : state.records;
      if (!filtered.length) {
        const empty = document.createElement("div");
        empty.className = "empty-list";
        empty.textContent = state.records.length ? "没有匹配的记录" : "当前没有记录";
        list.appendChild(empty);
        return;
      }
      filtered.forEach(r => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "record-item" + (r.index === state.currentIndex ? " active" : "");
        item.addEventListener("click", () => openRecord(r.index));
        const title = document.createElement("div");
        title.className = "record-title";
        const num = document.createElement("span"); num.className = "record-number"; num.textContent = "#" + r.index; title.appendChild(num);
        const chars = document.createElement("span"); chars.className = "record-meta"; chars.textContent = r.content_chars + " 字"; title.appendChild(chars);
        const badge = document.createElement("span"); badge.className = "status-badge " + statusClass(r.review_status); badge.textContent = statusLabel(r.review_status); title.appendChild(badge);
        item.appendChild(title);
        const concept = document.createElement("div"); concept.className = "record-concept"; concept.textContent = r.concept_name || "(无 concept_name)"; item.appendChild(concept);
        const docId = document.createElement("div"); docId.className = "record-docid"; docId.textContent = r.doc_id || "(无 doc_id)"; item.appendChild(docId);
        if (r.content_preview) {
          const pre = document.createElement("div"); pre.className = "record-preview"; pre.textContent = r.content_preview; item.appendChild(pre);
        }
        list.appendChild(item);
      });
      const active = list.querySelector(".record-item.active");
      if (active) active.scrollIntoView({ block: "nearest" });
    }
    function applyRecord(record, index) {
      state.creating = false;
      state.currentIndex = index;
      const reviewStatus = normalizeStatus(record.review_status);
      state.loaded = {
        doc_id: record.doc_id || "",
        concept_name: record.concept_name || "",
        source_type: record.source_type || "",
        md_content: record.md_content || "",
        review_status: reviewStatus,
      };
      $("doc_id").value = state.loaded.doc_id;
      $("concept_name").value = state.loaded.concept_name;
      $("source_type").value = state.loaded.source_type;
      $("md_content").value = state.loaded.md_content;
      renderStatus(reviewStatus);
      updateCreateMode();
      updateMdCount();
      renderMarkdownPreview();
      updatePosition();
      setMessage("已加载第 " + index + " 条");
    }
    function renderStatus(value) {
      const status = normalizeStatus(value);
      const badge = $("current-status");
      badge.textContent = statusLabel(status);
      badge.className = "status-badge " + statusClass(status);
      document.querySelectorAll(".status-group [data-status]").forEach(button => {
        const bs = button.dataset.status === "" ? null : button.dataset.status;
        button.classList.toggle("active", bs === status);
      });
    }
    function updateCreateMode() {
      const creating = state.creating;
      $("save").hidden = creating;
      $("prev").hidden = creating;
      $("next").hidden = creating;
      $("status-group").style.display = creating ? "none" : "";
      $("create-btn").hidden = creating;
      $("create-submit").hidden = !creating;
      $("create-cancel").hidden = !creating;
    }
    function createHasContent() {
      return !!($("doc_id").value.trim() || $("concept_name").value.trim() || $("md_content").value.trim());
    }
    async function loadList() {
      const res = await fetch("/api/records?status=" + encodeURIComponent(state.filter));
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      state.records = data.records;
      state.total = data.total;
      $("total").textContent = data.total;
      const stats = data.stats || {};
      $("stat-unreviewed").textContent = stats.unreviewed ?? 0;
      $("stat-correct").textContent = stats.correct ?? 0;
      $("stat-incorrect").textContent = stats.incorrect ?? 0;
      document.querySelectorAll(".filter-group [data-filter]").forEach(button => {
        button.classList.toggle("active", button.dataset.filter === state.filter);
      });
      renderList();
      updatePosition();
      return data;
    }
    async function openRecord(index) {
      if (index === state.currentIndex) return;
      if (!confirmSwitch()) return;
      const res = await fetch("/api/records/" + index);
      if (!res.ok) { setMessage("加载失败：" + await res.text(), true); return; }
      const data = await res.json();
      applyRecord(data.record, index);
      renderList();
    }
    function buildPayload(reviewStatus) {
      return {
        doc_id: $("doc_id").value,
        concept_name: $("concept_name").value,
        md_content: $("md_content").value,
        source_type: $("source_type").value || "manual_math_concept",
        review_status: normalizeStatus(reviewStatus),
      };
    }
    async function saveRecord(status, advance) {
      if (state.creating) { setMessage("正在新建记录，请点上方创建按钮", true); return; }
      if (state.currentIndex === null) { setMessage("没有可保存的记录", true); return; }
      const reviewStatus = status === undefined ? state.loaded.review_status : normalizeStatus(status);
      const payload = buildPayload(reviewStatus);
      setValidation(null);
      const res = await fetch("/api/records/" + state.currentIndex, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) { setMessage("保存失败：" + await res.text(), true); return; }
      const data = await res.json();
      state.loaded = {
        doc_id: payload.doc_id,
        concept_name: payload.concept_name,
        source_type: payload.source_type,
        md_content: payload.md_content,
        review_status: payload.review_status,
      };
      setMessage(advance ? "已保存，已跳到下一条" : "已保存");
      setValidation(data.validation);
      await loadList();
      if (advance) { goNext(); } else { renderList(); renderStatus(payload.review_status); }
    }
    function enterCreate() {
      if (!confirmSwitch()) return;
      state.creating = true;
      state.currentIndex = null;
      state.loaded = null;
      $("doc_id").value = "";
      $("concept_name").value = "";
      $("source_type").value = "manual_math_concept";
      $("md_content").value = "";
      renderStatus(null);
      updateCreateMode();
      updateMdCount();
      renderMarkdownPreview();
      updatePosition();
      renderList();
      setMessage("填写完成后点上方创建按钮");
      $("doc_id").focus();
    }
    async function submitCreate() {
      const payload = buildPayload(null);
      if (!payload.doc_id.trim()) { setMessage("doc_id 不能为空", true); return; }
      if (!payload.concept_name.trim()) { setMessage("concept_name 不能为空", true); return; }
      if (!payload.md_content.trim()) { setMessage("md_content 不能为空", true); return; }
      setValidation(null);
      const res = await fetch("/api/records", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) { setMessage("创建失败：" + await res.text(), true); return; }
      const data = await res.json();
      setMessage("已创建");
      setValidation(data.validation);
      await loadList();
      await openRecord(data.index);
    }
    function cancelCreate() {
      state.creating = false;
      setMessage("已取消新建");
      if (state.records.length) { openRecord(state.records[0].index); }
      else { updateCreateMode(); }
    }
    function goPrev() {
      const pos = currentPosition();
      if (pos <= 0) { setMessage("已经是第一条"); return; }
      openRecord(state.records[pos - 1].index);
    }
    function goNext() {
      const pos = currentPosition();
      if (pos < 0 || pos >= state.records.length - 1) { setMessage("已经是最后一条"); return; }
      openRecord(state.records[pos + 1].index);
    }
    async function validateAll() {
      setValidation(null);
      const res = await fetch("/api/validate", { method: "POST" });
      if (!res.ok) { setMessage("校验失败：" + await res.text(), true); return; }
      const data = await res.json();
      setValidation(data);
      setMessage(data.ok ? "校验通过" : "校验未通过，详见下方", !data.ok);
    }
    function exportJsonl() {
      window.location.href = "/api/export";
    }
    $("md_content").addEventListener("input", () => { updateMdCount(); renderMarkdownPreview(); });
    $("search").addEventListener("input", (e) => { state.search = e.target.value; renderList(); });
    $("save").addEventListener("click", () => saveRecord().catch(err => setMessage(err.message, true)));
    $("prev").addEventListener("click", goPrev);
    $("next").addEventListener("click", goNext);
    $("create-btn").addEventListener("click", enterCreate);
    $("create-submit").addEventListener("click", () => submitCreate().catch(err => setMessage(err.message, true)));
    $("create-cancel").addEventListener("click", cancelCreate);
    document.querySelectorAll(".status-group [data-status]").forEach(button => {
      button.addEventListener("click", () => {
        const value = button.dataset.status === "" ? null : button.dataset.status;
        saveRecord(value, true).catch(err => setMessage(err.message, true));
      });
    });
    document.querySelectorAll(".filter-group [data-filter]").forEach(button => {
      button.addEventListener("click", () => {
        state.filter = button.dataset.filter;
        loadList().then(() => {
          if (state.records.length) openRecord(state.records[0].index);
          else { state.currentIndex = null; state.loaded = null; updatePosition(); renderList(); }
        }).catch(err => setMessage(err.message, true));
      });
    });
    $("validate").addEventListener("click", () => validateAll().catch(err => setMessage(err.message, true)));
    $("export").addEventListener("click", exportJsonl);
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && (e.key === "s" || e.key === "S")) {
        e.preventDefault();
        if (state.creating) submitCreate().catch(err => setMessage(err.message, true));
        else saveRecord().catch(err => setMessage(err.message, true));
      } else if (!state.creating && e.altKey && e.key === "ArrowLeft") {
        e.preventDefault();
        goPrev();
      } else if (!state.creating && e.altKey && e.key === "ArrowRight") {
        e.preventDefault();
        goNext();
      }
    });
    window.addEventListener("load", async () => {
      try {
        await loadList();
        if (state.records.length) await openRecord(state.records[0].index);
        renderMarkdownPreview();
      } catch (err) { setMessage(err.message, true); }
    });
  </script>
</body>
</html>
"""


def _page_html(jsonl_path: Path) -> str:
    file_path = html.escape(str(jsonl_path), quote=True)
    return _PAGE_TEMPLATE.replace("__FILE_PATH__", file_path)


def create_app(jsonl_path: Path) -> FastAPI:
    jsonl_path = jsonl_path.expanduser().resolve()
    records = load_records(jsonl_path)
    app = FastAPI(title="Math Concepts Review Tool", docs_url=None, redoc_url=None)
    app.state.jsonl_path = jsonl_path
    app.state.records = records
    app.state.save_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    async def review_page() -> str:
        return _page_html(jsonl_path)

    @app.get("/api/records")
    async def list_records(status: str = Query(default="all")) -> dict[str, Any]:
        try:
            with app.state.save_lock:
                app.state.records = load_records(app.state.jsonl_path)
                snapshot = list(app.state.records)
                stats = build_stats(snapshot)
            visible = filter_records(snapshot, status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # visible 是 snapshot 的子集（同对象引用），用 id() 还原原文件 index。
        index_by_id = {id(record): index for index, record in enumerate(snapshot)}
        return {
            "jsonl_path": str(app.state.jsonl_path),
            "total": len(snapshot),
            "stats": stats,
            "records": [
                to_lightweight(record, index_by_id[id(record)])
                for record in visible
            ],
        }

    @app.get("/api/records/{index}")
    async def get_record(index: int) -> dict[str, Any]:
        if index < 0 or index >= len(app.state.records):
            raise HTTPException(status_code=404, detail="记录不存在")
        return {"index": index, "record": app.state.records[index]}

    @app.put("/api/records/{index}")
    async def update_record(index: int, update: RecordInput) -> dict[str, Any]:
        if index < 0 or index >= len(app.state.records):
            raise HTTPException(status_code=404, detail="记录不存在")
        if not update.doc_id.strip():
            raise HTTPException(status_code=400, detail="doc_id 不能为空")
        if not update.concept_name.strip():
            raise HTTPException(status_code=400, detail="concept_name 不能为空")
        if not update.md_content.strip():
            raise HTTPException(status_code=400, detail="md_content 不能为空")
        if update.source_type != REQUIRED_SOURCE_TYPE:
            raise HTTPException(
                status_code=400,
                detail=f"source_type 必须为 {REQUIRED_SOURCE_TYPE}",
            )
        if update.review_status not in REVIEW_STATUSES:
            raise HTTPException(
                status_code=400,
                detail="review_status 必须为 null、correct 或 incorrect",
            )

        with app.state.save_lock:
            for i, rec in enumerate(app.state.records):
                if i != index and rec.get("doc_id") == update.doc_id:
                    raise HTTPException(
                        status_code=409,
                        detail=f"doc_id 已存在: {update.doc_id}",
                    )
            previous_record = dict(app.state.records[index])
            app.state.records[index]["doc_id"] = update.doc_id
            app.state.records[index]["concept_name"] = update.concept_name
            app.state.records[index]["md_content"] = update.md_content
            app.state.records[index]["source_type"] = update.source_type
            app.state.records[index]["review_status"] = update.review_status
            try:
                save_records_atomic(app.state.jsonl_path, app.state.records)
            except Exception as exc:
                app.state.records[index] = previous_record
                raise HTTPException(status_code=500, detail="保存失败") from exc

        lightweight = to_lightweight(app.state.records[index], index)
        return {
            "ok": True,
            "index": index,
            "record": lightweight,
            "validation": validate_records(app.state.records),
        }

    @app.post("/api/records")
    async def create_record(create: RecordInput) -> dict[str, Any]:
        if not create.doc_id.strip():
            raise HTTPException(status_code=400, detail="doc_id 不能为空")
        if not create.concept_name.strip():
            raise HTTPException(status_code=400, detail="concept_name 不能为空")
        if not create.md_content.strip():
            raise HTTPException(status_code=400, detail="md_content 不能为空")
        if create.source_type != REQUIRED_SOURCE_TYPE:
            raise HTTPException(
                status_code=400,
                detail=f"source_type 必须为 {REQUIRED_SOURCE_TYPE}",
            )
        if create.review_status not in REVIEW_STATUSES:
            raise HTTPException(
                status_code=400,
                detail="review_status 必须为 null、correct 或 incorrect",
            )

        with app.state.save_lock:
            if any(rec.get("doc_id") == create.doc_id for rec in app.state.records):
                raise HTTPException(
                    status_code=409,
                    detail=f"doc_id 已存在: {create.doc_id}",
                )
            new_record = {
                "doc_id": create.doc_id,
                "concept_name": create.concept_name,
                "md_content": create.md_content,
                "source_type": create.source_type,
                "review_status": create.review_status,
            }
            app.state.records.append(new_record)
            new_index = len(app.state.records) - 1
            try:
                save_records_atomic(app.state.jsonl_path, app.state.records)
            except Exception as exc:
                app.state.records.pop()
                raise HTTPException(status_code=500, detail="保存失败") from exc

        lightweight = to_lightweight(app.state.records[new_index], new_index)
        return {
            "ok": True,
            "index": new_index,
            "record": lightweight,
            "validation": validate_records(app.state.records),
        }

    @app.post("/api/validate")
    async def validate_endpoint() -> dict[str, Any]:
        with app.state.save_lock:
            snapshot = list(app.state.records)
        return validate_records(snapshot)

    @app.get("/api/export")
    async def export_jsonl():
        return FileResponse(
            app.state.jsonl_path,
            media_type="application/x-ndjson",
            filename=app.state.jsonl_path.name,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地数学概念 JSONL 人工复核工具")
    parser.add_argument(
        "--jsonl-path",
        default=DEFAULT_JSONL_PATH,
        type=Path,
        help="待复核的数学概念 JSONL 文件路径",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.jsonl_path.is_file():
        raise SystemExit(f"JSONL 文件不存在: {args.jsonl_path}")
    app = create_app(args.jsonl_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
