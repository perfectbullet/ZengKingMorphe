# Markdown Graph Step 0～12

该流水线以单本教材 meta 为唯一业务配置载体。配置优先级统一为：**CLI > meta > `.env` > 代码默认值**。所有中间数据与人工数据均为 JSON/JSONL；JSONL 写入采用临时文件加 `os.replace()`，审核应用首次保存时创建一次 `.bak`。

## 初始化

```bash
bash run_pipeline.sh --init-meta /path/to/02珠宝手绘效果图表现_full.md \
  --domain industrial_training --subject jewelry_hand_rendering
```

这会在教材目录生成 `02珠宝手绘效果图表现_meta.json`，并在 `prompts/entity_type/` 创建该教材专属 YAML。只需人工填写 meta 的 `toc_start_line`、`toc_end_line`、`body_start_line`。

## 运行与人工检查点

```bash
bash run_pipeline.sh --meta /path/to/02珠宝手绘效果图表现_meta.json -step all
bash run_pipeline.sh --meta /path/to/02珠宝手绘效果图表现_meta.json --confirm-step 4
bash run_pipeline.sh --meta /path/to/02珠宝手绘效果图表现_meta.json \
  --working-dir /path/to/lightrag_book -step all
```

第一次 `all` 只运行 1～4 并停在验证报告。确认 Step 4 后，第二次会运行 5～11 并停在人工实体审核准备。完成审核后运行 `-step 12`。

```bash
python ../entity_review/app.py --data outputs/11_review/BOOK.entity_review.jsonl --port 8011
bash run_pipeline.sh --meta /path/to/BOOK_meta.json -step 12
python ../asr_hotwords_review/app.py --data outputs/12_asr/BOOK.asr_hotwords.jsonl --port 8012
```

Step 5 的 `content_list_v2` 仅用于图片 evidence；`content` 保持检索正文，`kg_content` 仅供实体抽取，避免业务头部、路径和图片噪声进入 KG。Step 9 仅通过 LightRAG 存储抽象接口导出，绝不读取 GraphML。

在线 Step 10 可加 `--dry-run`；它不会调用 Qwen 或 DeepSeek。
