# 数学概念 JSONL 人工复核工具

这是一个独立的本地 FastAPI 工具，用于在浏览器中逐条查看、修改并保存数学概念 JSONL 数据。每条记录包含 `doc_id`、`concept_name`、`md_content`、`source_type`、`review_status` 五个核心字段，工具会原样保留记录中的未知字段、记录顺序和中文字符，并支持逐条标记复核状态。

实现风格参考同目录上一层的 `tools/asr_latex_review/app.py`：单文件 `app.py`、内嵌 HTML/CSS/JS、原子保存 + 一次性 `.bak` 备份。

## 默认 JSONL 路径

```
ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl
```

## 启动

> 所有 Python 命令请在 conda 环境 `morphe` 中执行。

从仓库根目录执行：

```bash
cd ZengKingMorphe

python tools/math_concepts_review/app.py \
  --jsonl-path ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl \
  --host 127.0.0.1 \
  --port 8766
```

不传 `--jsonl-path` 时使用上面的默认路径。浏览器打开：

```
http://127.0.0.1:8766
```

局域网复核时可将 `--host` 改为 `0.0.0.0`。

## 依赖

依赖 `fastapi`、`uvicorn`、`pydantic`，均已包含在 `ai-service/requirements.txt` 中，无需额外安装：

```bash
pip install fastapi uvicorn pydantic
```

## 页面功能

- 顶部：工具标题、当前 JSONL 文件路径、复核统计（总数 / 未复核 / 正确 / 错误）、状态筛选（全部 / 未复核 / 正确 / 错误）、当前位置、保存状态。
- 左侧：记录列表（显示 index、concept_name、doc_id 简写、字符数、复核状态徽章、预览），上方提供搜索框，可按 `concept_name` / `doc_id` / `content_preview` 过滤（第一版不搜索完整 `md_content`）。
- 右侧：编辑区，包含 `doc_id`、`concept_name`、`source_type` 输入框、当前复核状态徽章，以及 `md_content` 的左右两栏工作区。
  - 左栏：`md_content` 编辑文本框（等宽字体）。保存的是原始 Markdown 文本，不自动格式化、不自动改写教材原文。
  - 右栏：`md_content` 渲染预览，渲染 LaTeX 公式（纯文本 + KaTeX，不走 Markdown 解析）；修改左栏后右栏立即刷新。
  - 预览仅用于复核，**保存时仍写入原始文本**，不会把渲染后的 KaTeX 结果写入 JSONL。
  - 实时显示 `md_content` 字符数。
- 按钮：保存、上一条、下一条、一键判定（未复核 / ✓ 正确 / ✗ 错误，点击即保存并跳到下一条）、新增记录、校验、下载 JSONL。

### LaTeX 渲染预览

- `md_content` 区域采用左编辑 / 右预览的左右对照布局；窄屏（宽度 ≤ 1100px）自动转为上下布局。
- 右侧预览参考 `tools/asr_latex_review/app.py`：把原始文本直接写入预览区 `textContent`，再由 KaTeX `auto-render` 扫描定界符渲染公式（页面仅通过 CDN 加载 KaTeX，不再使用 marked / DOMPurify）。
- 不走 Markdown 解析，因此跨行块级公式 `$$...$$` 不会被破坏；普通文字按纯文本展示（不渲染标题 / 列表 / 表格等 Markdown 样式）。
- 修改左侧 textarea 后右侧预览实时刷新（input 事件）。
- 支持的 LaTeX 定界符：`$...$`（行内）、`$$...$$`（块级）、`\(...\)`（行内）、`\[...\]`（块级）。
- 预览仅用于人工复核；保存（按钮或 `Ctrl+S`）写入的始终是左侧 textarea 的原始文本，**不会保存渲染结果，也不会替换或转义 LaTeX 符号、中文**。
- 图片相对路径（如 `![](images/xxx.jpg)`）不保证显示，本工具重点是文字与公式复核，未做 `images` 静态目录映射。

### 复核状态

每条记录的 `review_status` 只允许以下三个值：

- `null`：未复核（缺少该字段时的默认值）；
- `correct`：内容正确；
- `incorrect`：内容有误 / 需修正。

工具栏的统计与筛选、列表项右侧的状态徽章、详情区的状态徽章均实时反映该字段。点击详情区的「未复核 / ✓ 正确 / ✗ 错误」按钮会立即写入该状态、保存当前编辑内容并跳到下一条，便于逐条快速判定。新建记录默认为未复核，创建后可再判定。

### 快捷键

| 快捷键 | 功能 |
| --- | --- |
| `Ctrl + S` | 保存当前记录（新建态下改为创建新记录） |
| `Alt + ←` | 上一条 |
| `Alt + →` | 下一条 |

当当前记录存在未保存修改时，切换到其他记录会弹窗确认：`当前记录有未保存修改，确定要切换吗？`；新建记录已填写内容时切换会弹窗确认：`当前正在新建记录且已填写内容，确定要放弃吗？`。

## API 简介

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 返回复核页面（内嵌 HTML） |
| `GET` | `/api/records` | 返回轻量记录列表（含 `total`、`stats`、每条的 `content_chars` 与 `content_preview`，不含完整 `md_content`）；可用 `?status=all/unreviewed/correct/incorrect` 过滤 |
| `GET` | `/api/records/{index}` | 返回单条完整记录；`index` 越界返回 404 |
| `PUT` | `/api/records/{index}` | 更新单条记录，覆盖五个核心字段（含 `review_status`），保留未知字段；返回更新后的轻量记录与全量校验结果 |
| `POST` | `/api/records` | 在末尾新增一条记录（五个核心字段），`doc_id` 重复返回 409；返回新记录轻量信息与全量校验结果 |
| `POST` | `/api/validate` | 校验当前内存中的全部记录 |
| `GET` | `/api/export` | 下载当前 JSONL 文件（保持原文件名） |

## 校验规则

`POST /api/validate` 与保存/创建接口返回的 `validation` 遵循以下规则：

1. 每条必须有非空 `doc_id`；
2. 每条必须有非空 `concept_name`；
3. 每条必须有非空 `md_content`；
4. 每条 `source_type` 必须等于 `manual_math_concept`；
5. 每条 `review_status` 必须为 `null`、`correct` 或 `incorrect` 之一；
6. `doc_id` 不允许重复。

返回示例（通过）：

```json
{ "ok": true, "total": 41, "errors": [] }
```

返回示例（失败）：

```json
{
  "ok": false,
  "total": 41,
  "errors": [
    { "index": 3, "field": "concept_name", "message": "concept_name 不能为空" }
  ]
}
```

## 保存与备份

- 保存采用原子替换：先写入临时文件并 `fsync`，再用 `os.replace` 覆盖原 JSONL，保存后仍保持一行一个 JSON 对象，**不会变成 JSON 数组**。
- 第一次保存前会在原文件旁创建一次 `<原文件名>.bak`；如果 `.bak` 已存在，**不会覆盖**。
- 保存使用 `json.dumps(record, ensure_ascii=False)`，**中文不会被转义**。
- 保存**不删除未知字段、不改变记录顺序、不自动格式化 `md_content`**。

## 注意事项

- 不要多人同时对同一个 JSONL 文件启动本工具，否则相互覆盖会丢失复核结果。
- 请在编辑前确认传入了正确的 `--jsonl-path`，并妥善保留 `.bak` 备份。
- 本工具不修改 `tools/asr_latex_review`、`ai-service` 业务代码以及 LightRAG / RAGAnything 核心代码。
