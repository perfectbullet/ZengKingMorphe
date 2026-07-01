# manual_concepts_lightrag

独立的“手动教材数学概念导入 LightRAG”工具，不依赖 `ai-service` 业务模块，也不修改 LightRAG 源码。

## 数据与实体

默认输入：

```text
/home/zj/ZengKingMorphe/ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl
```

JSONL 每条记录本身就是一个数学概念：

- `concept_name`：主实体名；
- `doc_id`：文档 ID；
- `md_content`：概念原文，同时用于 chunk 检索和实体 description；
- `review_status`：默认只导入 `correct`，使用 `--include-non-correct` 可显式放开。

每条有效记录最终 upsert 为 `MANUAL_MATH_CONCEPT` 实体，并保存 `doc_id`、`file_path`、`source_type`、`review_status`、`strict`、`aliases` 等 GraphML 安全元数据。

## 严格实体范围

实体白名单位于：

```text
tools/manual_concepts_lightrag/entity_whitelist_draft.txt
```

当前只保留教材数学概念，不保留公式、变量、数字、符号、运算词、人名、例子对象、解题步骤或推导过程。默认行为是：

1. `entity_types=["MANUAL_MATH_CONCEPT"]` 收窄 LLM 抽取类型；
2. 非白名单 `concept_name` 不导入；
3. 文档插入后，使用 LightRAG 公开 API 删除图中非白名单实体；
4. 查询只将白名单内的手动概念实体判为 HIT。

当前 LightRAG 版本只从 `addon_params` 读取 `language` 和 `entity_types`，没有自定义实体抽取 prompt 或禁用实体抽取的公开配置 key。脚本保留了严格 prompt 策略常量，但不硬编码不存在的参数；白名单过滤和导入后清理是主要控制手段。原有无效的 `entity_types_guidance` 已移除。

## 使用

```bash
cd /home/zj/ZengKingMorphe/tools/manual_concepts_lightrag
conda activate morphe

# 默认覆盖导入
./run_import.sh

# 严格查询并依据对应 md_content 回答
QUERY="请帮我讲解二项式定理" MODE=local ./run_test.sh --answer
QUERY="请帮我讲解分类加法计数原理" MODE=local ./run_test.sh --answer
```

配置均可通过环境变量覆盖：

```bash
CONFIG=/path/concepts.jsonl \
WORKING_DIR=/path/lightrag_data \
ENTITY_WHITELIST=/path/whitelist.txt \
./run_import.sh
```

临时关闭白名单可传 `--disable-whitelist`。从配置重新生成白名单：

```bash
python import_manual_concepts_lightrag.py \
  --config "$CONFIG" \
  --working-dir "$WORKING_DIR" \
  --entity-whitelist entity_whitelist_draft.txt \
  --generate-whitelist-only
```

保存会继续使用现有文件存储后端（JsonKV、NanoVectorDB、NetworkX），默认数据目录为 `/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts`。
