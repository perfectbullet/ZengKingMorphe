#!/usr/bin/env python3
"""Independent local review tool for ASR-to-LaTeX JSONL data."""

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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


REVIEW_STATUSES = {None, "correct", "incorrect"}
FILTER_STATUSES = {"all", "unreviewed", "correct", "incorrect"}


def load_records(jsonl_path: Path) -> list[dict]:
    """Load JSONL records and add defaults required by the review UI."""
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as source:
        for line_no, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL 第 {line_no} 行解析失败: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 第 {line_no} 行必须是 JSON 对象")
            record.pop("index", None)
            record.setdefault("id", None)
            record.setdefault("before", "")
            record.setdefault("after", "")
            record.setdefault("review_status", None)
            records.append(record)
    return records


def ensure_backup_once(jsonl_path: Path) -> None:
    """Create ``<file>.bak`` once, without replacing an existing backup."""
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
    """Back up once, then atomically replace the JSONL file."""
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


def build_stats(records: list[dict]) -> dict:
    """Build review counters for the complete data set."""
    correct = sum(record.get("review_status") == "correct" for record in records)
    incorrect = sum(
        record.get("review_status") == "incorrect" for record in records
    )
    return {
        "total": len(records),
        "unreviewed": len(records) - correct - incorrect,
        "correct": correct,
        "incorrect": incorrect,
    }


def filter_records(records: list[dict], status: str) -> list[dict]:
    """Filter records by a supported UI status."""
    if status not in FILTER_STATUSES:
        raise ValueError(f"不支持的过滤状态: {status}")
    if status == "all":
        return list(records)
    if status == "unreviewed":
        return [record for record in records if record.get("review_status") is None]
    return [record for record in records if record.get("review_status") == status]


class RecordUpdate(BaseModel):
    id: str | None = None
    after: str
    review_status: str | None


def _page_html(jsonl_path: Path) -> str:
    file_path = html.escape(str(jsonl_path), quote=True)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ASR -> LaTeX 人工标注</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #17202a; background: #f4f6f7; }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{ display: flex; flex-direction: column; margin: 0; overflow: hidden; }}
    header {{ flex: none; padding: 14px 22px; color: white; background: #1f3a4d; }}
    header h1 {{ margin: 0 0 5px; font-size: 19px; }}
    header p {{ margin: 0; overflow-wrap: anywhere; color: #d6e4ec; font-size: 12px; }}
    .app-layout {{ display: flex; flex: 1; min-height: 0; }}
    .sidebar {{ display: flex; flex: 0 0 360px; flex-direction: column; min-width: 0; border-right: 1px solid #c9d2d8; background: white; }}
    .sidebar-controls {{ flex: none; padding: 14px; border-bottom: 1px solid #dce3e7; }}
    .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; margin-bottom: 12px; font-size: 13px; color: #52636d; }}
    .stats span {{ font-variant-numeric: tabular-nums; }}
    .filters, .actions, .status-actions {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }}
    .filters {{ display: grid; grid-template-columns: 1fr 1fr; }}
    button {{ min-height: 36px; padding: 7px 12px; border: 1px solid #9aa8b0; border-radius: 6px; background: white; cursor: pointer; }}
    button:hover {{ background: #edf3f6; }}
    button.active {{ color: white; border-color: #276678; background: #276678; }}
    button.primary {{ color: white; border-color: #176b4d; background: #176b4d; }}
    button:disabled {{ opacity: .45; cursor: default; }}
    .record-list {{ flex: 1; min-height: 0; overflow-y: auto; padding: 8px; }}
    .record-item {{ display: block; width: 100%; min-height: 0; margin: 0 0 7px; padding: 11px 12px; overflow: hidden; text-align: left; border: 1px solid #d8e0e4; border-left: 4px solid transparent; border-radius: 6px; background: white; }}
    .record-item:hover {{ border-color: #9db7c5; background: #f3f8fa; }}
    .record-item.active {{ color: #17202a; border-color: #8bb3c7; border-left-color: #28789b; background: #eaf5fa; }}
    .record-title {{ display: flex; align-items: center; gap: 7px; min-width: 0; margin-bottom: 6px; font-size: 12px; color: #52636d; }}
    .record-number {{ font-weight: 700; color: #253943; }}
    .record-id {{ max-width: 105px; overflow: hidden; color: #28708f; text-overflow: ellipsis; white-space: nowrap; }}
    .record-time {{ margin-left: auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .record-before-preview {{ display: -webkit-box; overflow: hidden; color: #26373f; font-size: 13px; line-height: 1.5; overflow-wrap: anywhere; -webkit-box-orient: vertical; -webkit-line-clamp: 3; }}
    .status-badge, .meta-badge {{ display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border: 1px solid transparent; border-radius: 999px; white-space: nowrap; font-size: 12px; }}
    .status-badge.unreviewed {{ color: #435661; border-color: #c6d2d8; background: #edf2f4; }}
    .status-badge.correct {{ color: #176b4d; border-color: #a9d5c2; background: #e7f5ee; }}
    .status-badge.incorrect {{ color: #a93226; border-color: #e3b5b0; background: #faecea; }}
    .empty-list {{ padding: 24px 12px; text-align: center; color: #71818a; font-size: 13px; }}
    .main-panel {{ flex: 1; min-width: 0; overflow-y: auto; padding: 18px 22px 30px; }}
    .detail-header {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 14px; }}
    .meta-badges {{ display: flex; flex-wrap: wrap; gap: 7px; }}
    .meta-badge {{ color: #344952; border-color: #cbd7dc; background: white; }}
    .meta-badge.status-correct {{ color: #176b4d; border-color: #a9d5c2; background: #e7f5ee; }}
    .meta-badge.status-incorrect {{ color: #a93226; border-color: #e3b5b0; background: #faecea; }}
    .position {{ color: #52636d; font-size: 13px; font-variant-numeric: tabular-nums; }}
    .id-editor {{ display: grid; grid-template-columns: auto minmax(180px, 420px); align-items: center; gap: 10px; margin-bottom: 14px; }}
    .id-editor label {{ color: #344952; font-size: 13px; font-weight: 700; }}
    .id-editor input {{ width: 100%; min-height: 38px; padding: 7px 10px; border: 1px solid #9aa8b0; border-radius: 6px; font: inherit; }}
    .id-editor input:focus, textarea:focus {{ border-color: #28789b; outline: 2px solid #d9edf5; }}
    .workspace {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    section {{ min-width: 0; padding: 16px; border: 1px solid #ced7dc; border-radius: 8px; background: white; }}
    section.full {{ grid-column: 1 / -1; }}
    h2 {{ margin: 0 0 10px; font-size: 15px; }}
    .text-block, .preview {{ min-height: 150px; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.7; }}
    textarea {{ width: 100%; min-height: 210px; resize: vertical; padding: 11px; border: 1px solid #9aa8b0; border-radius: 6px; font: inherit; line-height: 1.6; }}
    .footer {{ justify-content: space-between; margin-top: 14px; }}
    #message {{ min-height: 24px; margin: 10px 0 0; color: #176b4d; }}
    #message.error {{ color: #a93226; }}
    @media (max-width: 900px) {{ .sidebar {{ flex-basis: 320px; }} .workspace {{ grid-template-columns: 1fr; }} section.full {{ grid-column: auto; }} }}
    @media (max-width: 680px) {{ body {{ overflow: auto; }} .app-layout {{ display: block; }} .sidebar {{ width: 100%; height: 45vh; border-right: 0; border-bottom: 1px solid #c9d2d8; }} .main-panel {{ overflow: visible; padding: 14px; }} }}
  </style>
</head>
<body>
  <header><h1>ASR -> LaTeX 人工标注</h1><p>当前文件：{file_path}</p></header>
  <div class="app-layout">
    <aside class="sidebar">
      <div class="sidebar-controls">
        <div class="stats">
          <span>total: <b id="total">0</b></span><span>unreviewed: <b id="unreviewed">0</b></span>
          <span>correct: <b id="correct">0</b></span><span>incorrect: <b id="incorrect">0</b></span>
        </div>
        <div id="filters" class="filters">
          <button data-filter="all" class="active">全部</button><button data-filter="unreviewed">未判断</button>
          <button data-filter="correct">correct</button><button data-filter="incorrect">incorrect</button>
        </div>
      </div>
      <div id="record-list" class="record-list"></div>
    </aside>
    <main class="main-panel">
      <div class="detail-header">
        <div class="meta-badges">
          <span id="meta-id" class="meta-badge">id: 未设置</span>
          <span id="meta-time" class="meta-badge">time: -</span>
          <span id="meta-duration" class="meta-badge">duration: null</span>
          <span id="meta-index" class="meta-badge">index: -</span>
          <span id="meta-status" class="meta-badge">status: 未判断</span>
        </div>
        <b id="position" class="position">0 / 0</b>
      </div>
      <div class="id-editor">
        <label for="record-id-input">ID</label>
        <input id="record-id-input" type="text" placeholder="可选，例如 MATH-001">
      </div>
      <div class="workspace">
        <section><h2>before</h2><div id="before" class="text-block"></div></section>
        <section><h2>after rendered preview</h2><div id="preview" class="preview"></div></section>
        <section class="full"><h2>after</h2><textarea id="after" spellcheck="false"></textarea></section>
      </div>
      <div class="actions footer">
        <div class="actions"><button id="previous">上一条</button><button id="next">下一条</button></div>
        <div class="status-actions">
          <button data-status="">未判断</button><button data-status="correct">correct</button>
          <button data-status="incorrect">incorrect</button><button id="save" class="primary">保存</button>
        </div>
      </div>
      <p id="message"></p>
    </main>
  </div>
  <script>
    const state = {{ records: [], stats: {{}}, currentPosition: 0, filter: "all" }};
    const $ = (id) => document.getElementById(id);

    function statusLabel(value) {{ return value === null ? "未判断" : value; }}
    function statusClass(value) {{ return value === "correct" ? "correct" : value === "incorrect" ? "incorrect" : "unreviewed"; }}
    function idLabel(value) {{ return typeof value === "string" && value.trim() ? value : "未设置"; }}
    function current() {{ return state.records[state.currentPosition] || null; }}
    function setMessage(text, isError = false) {{ $("message").textContent = text; $("message").classList.toggle("error", isError); }}
    function addTextElement(parent, tag, className, text) {{
      const element = document.createElement(tag);
      element.className = className;
      element.textContent = text;
      parent.appendChild(element);
      return element;
    }}
    function renderPreview() {{
      const preview = $("preview");
      preview.textContent = $("after").value;
      if (window.renderMathInElement) {{
        renderMathInElement(preview, {{
          delimiters: [
            {{left: "$$", right: "$$", display: true}}, {{left: "$", right: "$", display: false}},
            {{left: "\\\\(", right: "\\\\)", display: false}}, {{left: "\\\\[", right: "\\\\]", display: true}}
          ],
          throwOnError: false
        }});
      }}
    }}
    function renderStats() {{ ["total", "unreviewed", "correct", "incorrect"].forEach(k => $(k).textContent = state.stats[k] ?? 0); }}
    function renderSidebar() {{
      const list = $("record-list");
      list.replaceChildren();
      if (!state.records.length) {{
        addTextElement(list, "div", "empty-list", "当前过滤条件下没有记录");
        return;
      }}
      state.records.forEach((record, position) => {{
        const item = document.createElement("button");
        item.type = "button";
        item.className = `record-item${{position === state.currentPosition ? " active" : ""}}`;
        item.addEventListener("click", () => {{ state.currentPosition = position; renderRecord(); }});
        const title = addTextElement(item, "div", "record-title", "");
        addTextElement(title, "span", "record-number", `${{position + 1}} / ${{state.records.length}}`);
        addTextElement(title, "span", "record-index", `#${{record.index}}`);
        addTextElement(title, "span", "record-id", `ID: ${{idLabel(record.id)}}`);
        addTextElement(title, "span", `status-badge ${{statusClass(record.review_status)}}`, statusLabel(record.review_status));
        if (record.time) addTextElement(title, "span", "record-time", String(record.time));
        addTextElement(item, "div", "record-before-preview", record.before ?? "");
        list.appendChild(item);
      }});
      requestAnimationFrame(() => list.querySelector(".record-item.active")?.scrollIntoView({{block: "nearest"}}));
    }}
    function renderMeta(record) {{
      $("meta-id").textContent = `id: ${{idLabel(record?.id)}}`;
      $("meta-time").textContent = `time: ${{record?.time ?? "-"}}`;
      $("meta-duration").textContent = `duration: ${{record?.duration ?? "null"}}`;
      $("meta-index").textContent = `index: ${{record?.index ?? "-"}}`;
      $("meta-status").textContent = `status: ${{statusLabel(record?.review_status ?? null)}}`;
      $("meta-status").className = `meta-badge status-${{statusClass(record?.review_status)}}`;
    }}
    function renderRecord() {{
      const record = current();
      $("position").textContent = record ? `${{state.currentPosition + 1}} / ${{state.records.length}}` : "0 / 0";
      $("before").textContent = record?.before ?? "";
      $("record-id-input").value = record?.id ?? "";
      $("after").value = record?.after ?? "";
      document.querySelectorAll("[data-status]").forEach(button => button.classList.toggle("active", (button.dataset.status || null) === (record?.review_status ?? null)));
      $("previous").disabled = !record || state.currentPosition === 0;
      $("next").disabled = !record || state.currentPosition >= state.records.length - 1;
      $("save").disabled = !record;
      renderMeta(record);
      renderSidebar();
      renderPreview();
    }}
    async function loadRecords(filter = state.filter, preferredIndex = null, fallbackPosition = 0, initialLoad = false) {{
      const response = await fetch(`/api/records?status=${{encodeURIComponent(filter)}}`);
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      state.filter = filter; state.records = data.records; state.stats = data.stats;
      let target = preferredIndex === null ? -1 : state.records.findIndex(record => record.index === preferredIndex);
      if (target < 0 && initialLoad && filter === "all") target = state.records.findIndex(record => record.review_status === null);
      if (target < 0 && state.records.length) target = Math.min(Math.max(fallbackPosition, 0), state.records.length - 1);
      state.currentPosition = target >= 0 ? target : 0;
      document.querySelectorAll("[data-filter]").forEach(button => button.classList.toggle("active", button.dataset.filter === filter));
      renderStats(); renderRecord();
    }}
    async function save(status, advance = false) {{
      const record = current(); if (!record) return;
      const savedIndex = record.index;
      const savedPosition = state.currentPosition;
      const nextIndex = advance ? state.records[savedPosition + 1]?.index ?? null : savedIndex;
      const response = await fetch(`/api/records/${{record.index}}`, {{
        method: "POST", headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          id: $("record-id-input").value.trim() || null,
          after: $("after").value,
          review_status: status
        }})
      }});
      if (!response.ok) throw new Error(await response.text());
      setMessage("保存成功");
      await loadRecords(state.filter, nextIndex, savedPosition);
    }}
    $("after").addEventListener("input", renderPreview);
    $("previous").addEventListener("click", () => {{ state.currentPosition--; renderRecord(); }});
    $("next").addEventListener("click", () => {{ state.currentPosition++; renderRecord(); }});
    $("save").addEventListener("click", () => save(current().review_status).catch(error => setMessage(error.message, true)));
    document.querySelectorAll("[data-status]").forEach(button => button.addEventListener("click", () => save(button.dataset.status || null, true).catch(error => setMessage(error.message, true))));
    document.querySelectorAll("[data-filter]").forEach(button => button.addEventListener("click", () => loadRecords(button.dataset.filter, null, 0).catch(error => setMessage(error.message, true))));
    window.addEventListener("load", () => loadRecords("all", null, 0, true).catch(error => setMessage(error.message, true)));
  </script>
</body>
</html>"""


def create_app(jsonl_path: Path) -> FastAPI:
    jsonl_path = jsonl_path.expanduser().resolve()
    records = load_records(jsonl_path)
    app = FastAPI(title="ASR LaTeX Review Tool", docs_url=None, redoc_url=None)
    app.state.jsonl_path = jsonl_path
    app.state.records = records
    app.state.save_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    async def review_page() -> str:
        return _page_html(jsonl_path)

    @app.get("/api/records")
    async def get_records(
        status: str = Query(default="all"),
    ) -> dict[str, Any]:
        try:
            indexed_records = [
                {**record, "index": index}
                for index, record in enumerate(app.state.records)
            ]
            visible_records = filter_records(indexed_records, status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "records": visible_records,
            "stats": build_stats(app.state.records),
        }

    @app.post("/api/records/{index}")
    async def update_record(index: int, update: RecordUpdate) -> dict[str, Any]:
        if update.review_status not in REVIEW_STATUSES:
            raise HTTPException(status_code=400, detail="非法 review_status")
        if index < 0 or index >= len(app.state.records):
            raise HTTPException(status_code=404, detail="记录不存在")

        with app.state.save_lock:
            previous_record = dict(app.state.records[index])
            if "id" in update.model_fields_set:
                normalized_id = update.id.strip() if update.id is not None else ""
                app.state.records[index]["id"] = normalized_id or None
            app.state.records[index]["after"] = update.after
            app.state.records[index]["review_status"] = update.review_status
            try:
                save_records_atomic(app.state.jsonl_path, app.state.records)
            except Exception as exc:
                app.state.records[index] = previous_record
                raise HTTPException(status_code=500, detail="保存失败") from exc

        return {
            "ok": True,
            "index": index,
            "record": app.state.records[index],
            "stats": build_stats(app.state.records),
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 ASR -> LaTeX 人工标注工具")
    parser.add_argument("--file", required=True, type=Path, help="待标注 JSONL 文件")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8899, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.file.is_file():
        raise SystemExit(f"JSONL 文件不存在: {args.file}")
    app = create_app(args.file)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
