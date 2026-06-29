# ASR -> LaTeX 人工标注工具

这是一个独立的本地 FastAPI 工具，用于逐条查看和修订 ASR 转 LaTeX 的 JSONL 数据。页面展示 `before`、`after` 和 KaTeX 渲染预览，并可将记录标记为未判断、`correct` 或 `incorrect`。

## 启动

从仓库根目录执行：

```bash
python tools/asr_latex_review/app.py \
  --file /path/to/asr_to_latex_after_20260622_012245.jsonl \
  --host 127.0.0.1 \
  --port 8899
```

浏览器访问 `http://127.0.0.1:8899`。局域网访问时可将 `--host` 改为 `0.0.0.0`。

## JSONL 字段

输入文件保持 JSONL 格式，每条记录占一行。工具读取并展示 `before` 和 `after`，保留其他原有字段，并新增 `review_status`。接口中的 `index` 仅用于定位原文件记录，不会写回 JSONL。

`review_status` 只允许以下值：

- `null`：未判断，也是缺少该字段时的默认值；
- `correct`：转换正确；
- `incorrect`：转换错误。

## 保存和备份

保存会原子覆盖原 JSONL 文件，仍保持每行一个 JSON 对象，不会转换为 JSON 数组。第一次保存前会在原文件旁创建 `<原文件名>.bak`；如果该 `.bak` 已存在，工具不会覆盖它。

请在编辑前确认传入了正确的文件路径，并妥善保留备份。
