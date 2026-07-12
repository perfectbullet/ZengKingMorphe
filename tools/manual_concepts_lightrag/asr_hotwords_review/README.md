# ASR 热词管理

```bash
python app.py --data /path/to/BOOK.asr_hotwords.jsonl --host 0.0.0.0 --port 8012
```

该独立应用提供 JSON API：新增、编辑、删除、启停、按风险/状态过滤与纯文本导出。热词权重强制为 1～10，首次写入创建一次 `.bak`，不使用 SQLite 或 ASR 引擎适配器。
