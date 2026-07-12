# MinerU 单 PDF 解析脚本

MinerU 精准解析 API 单个 PDF 的官方限制为 **200 页、200 MB**。本脚本会使用 `pypdf` 读取页数和文件大小；只要任一限制超过，就将 PDF 物理拆分为临时文件，按原页码顺序串行解析，避免通过 `page_ranges` 绕过服务端限制。分片会同时满足页数和 200 MB 限制；若按页数切出的分片仍过大，会继续二分，因此实际分片页数可能少于 200 页。单独一页仍超过 200 MB 时会在上传前明确失败。

## 安装与运行

```bash
/home/zj/miniconda3/envs/morphe/bin/python -m pip install -r tools/call_online_mineru/requirements.txt
export MINERU_API_TOKEN='...'
python tools/call_online_mineru/extract_mineru_sdk.py \
  --input /path/to/book.pdf \
  --output /path/to/output \
  --model_version vlm \
  --chunk_pages 200 \
  --result_timeout 1800
```

`--chunk_pages` 默认 200，范围为 1 到 200；`--poll_interval` 和 `--result_timeout` 必须大于 0。仅当 PDF 同时不超过 200 页和 200 MB 时，才保持原有单任务行为。上传请求保留原始文件名（包括中文名），但不会提交 `data_id`，避免中文、空格或超长文件名违反该字段的服务端约束。

## 输出

普通 PDF：

```text
output/
├── document.pdf.zip
└── document/
    ├── document_full.md
    ├── document_content_list_v2.json
    ├── images/
    └── ...
```

超过 200 页的 PDF：

```text
output/
├── document/
│   ├── document_full.md
│   ├── document_content_list_v2.json
│   ├── images/part_0001/, images/part_0002/, ...
│   └── merge_manifest.json
└── document_parts/
    ├── document_part_0001_p0001-p0200.pdf.zip
    ├── document_part_0001_p0001-p0200/
    └── ...
```

大文件结果仅结构化合并正文 Markdown、`content_list_v2.json` 和图片；`model.json`、`middle.json`、`layout.pdf`、`span.pdf` 等原始调试产物不会拼接，仍保留在 `document_parts/` 中以便排查。图片以 `images/part_XXXX/` 隔离，Markdown 和 JSON 中的相对图片引用会同步改写。

任何分片失败都会立即停止，异常包含分片编号、原始页码范围和已获得的 `batch_id`；已成功分片的 ZIP/解压结果会保留在 `document_parts/`，最终统一目录不会被伪造为完成状态。
