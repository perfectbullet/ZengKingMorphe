# 实体人工审核

```bash
python app.py --data /path/to/BOOK.entity_review.jsonl --host 0.0.0.0 --port 8011
```

该独立 FastAPI 应用只读写 JSONL。`GET /api/records` 支持 `q`、`entity_type`、`conflict`、`decision` 过滤；`PUT /api/records/{index}` 保存人工结论。首次保存会创建一次 `.bak`，之后原子替换 JSONL。
