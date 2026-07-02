# manual_concepts_lightrag

独立的「人工概念 → LightRAG」导入工具集。把人工审核后的概念 JSONL 导入 LightRAG，为
`ai-service` 的 `concept_explain` 意图提供概念上下文召回数据。本子项目**不直接接入**
`ai-service`，只产出可供 `ai-service` 读取的 LightRAG `working_dir` 数据。

- 第一批 `domain = math`（高中数学选择性必修第三册）；
- 后续可扩展 `industrial_training`（工业实训）等领域；
- 命名上已泛化为 `concept` / `MANUAL_CONCEPT`，不写死 `math`。

不依赖 `ai-service` 业务模块，也不修改 LightRAG 源码。

---

## 数据与实体

默认输入：

```text
/home/zj/ZengKingMorphe/ai-service/data/math_concepts/05_selective3_math_concepts_definition_blocks_with_concept_name_20260630.jsonl
```

JSONL 每条记录本身就是一个概念：

- `concept_name`：主实体名；
- `doc_id`：文档 ID；
- `md_content`：概念原文，同时用于 chunk 检索和实体 description；
- `review_status`：默认只导入 `correct`，使用 `--include-non-correct` 可显式放开；
- `domain`（可选）：覆盖默认 domain，否则取 `CONCEPT_RETRIEVAL_DOMAIN`（默认 `math`）。

每条有效记录最终写入：

- `entity_type = MANUAL_CONCEPT`（默认；可经 `CONCEPT_RETRIEVAL_ENTITY_TYPE` 覆盖）；
- `domain`（默认 math）；
- `source_type = manual_concept`；
- `legacy_source_type`：若原 `source_type == manual_math_concept`，则保留为 `manual_math_concept`；
- 以及 `doc_id`、`file_path`、`md_path`、`review_status`、`strict`、`aliases` 等 GraphML 安全元数据。

> 兼容 legacy：旧数据写入的 `entity_type = MANUAL_MATH_CONCEPT` / `source_type = manual_math_concept`
> 在 `test_manual_concept_lightrag.py` 的 HIT 判定中仍被识别（详见下文）。

---

## LightRAG main 适配（重要）

本子项目当前针对 **LightRAG main 分支** 适配：

- 新版默认使用 `addon_params["entity_types_guidance"]`（字符串）注入 entity_extraction prompt 的
  `{entity_types_guidance}` 占位符；
- 可选 `addon_params["entity_type_prompt_file"]`（指向 prompt 文件），与 `entity_types_guidance` 可同时存在；
- **不再使用** 旧的 `addon_params["entity_types"]`（列表）机制；
- 因此 `.env` 中**不应再设置** `ENTITY_TYPES` 环境变量——若 LightRAG 检测到会报错，请从 `.env`
  删除并改用 `CONCEPT_RETRIEVAL_ENTITY_TYPE`（控制默认 entity_type）或通过
  `CONCEPT_RETRIEVAL_ENTITY_TYPE_PROMPT_FILE` 指定完整 prompt 文件。

`entity_types_guidance` 只是 prompt 软约束，硬控制仍来自：

1. `review_status == correct` 过滤；
2. `concept_name` 白名单过滤；
3. `prune_non_whitelisted_entities`（旧 ainsert 路径专用）；
4. 阶段 3 手动 upsert `MANUAL_CONCEPT` 实体。

---

## 两种导入路径

| 路径 | 脚本 | 是否触发 LLM 抽取 | 默认 working_dir | 用途 |
|---|---|---|---|---|
| 主路径 | `import_manual_concepts_lightrag.py` | 是（`ainsert`） | `lightrag_manual_concepts` | 主路径 |
| 历史实验性 | `import_manual_concepts_custom_kg.py` | 否（`ainsert_custom_kg`） | `lightrag_manual_concepts_custom_kg` | historical/experimental fallback |

主路径 `import_manual_concepts_lightrag.py`：

1. 使用 `rag.ainsert`，会触发默认 LLM 实体抽取；
2. 通过 `prune_non_whitelisted_entities` + 白名单 + 阶段 3 手动 upsert `MANUAL_CONCEPT` 实体尽量清洗脏实体；
3. 可能产生 `0!`、`A_n^m`、`C(n,0)`、`第1类方案`、`步骤` 等脏实体，但通过白名单和手动 upsert 进行清洗。

历史实验性路径 `import_manual_concepts_custom_kg.py` 保留为 experimental fallback：使用 `rag.ainsert_custom_kg`，不走默认 LLM 实体抽取，图中只保留人工概念实体。

---

## 使用

```bash
cd /home/zj/ZengKingMorphe/tools/manual_concepts_lightrag
conda activate morphe   # 或直接用 /home/zj/miniconda3/envs/morphe/bin/python

# 主路径：ainsert 导入（默认指向 lightrag_manual_concepts）
./run_import.sh

# 严格查询并依据对应 md_content 回答（指向 lightrag_manual_concepts）
QUERY="请帮我讲解二项式定理" MODE=local ./run_test.sh --answer

# （可选）历史实验性路径：custom KG 导入（实验性 fallback）
./run_import_custom_kg.sh --dry-run
./run_import_custom_kg.sh
```

### 环境变量

所有变量优先级统一为：**调用方传入 > `CONCEPT_RETRIEVAL_*` > legacy 旧变量 > 默认值**。

新变量（推荐）：

| 变量 | 说明 | 默认 |
|---|---|---|
| `CONCEPT_RETRIEVAL_ENABLED` | 是否启用概念召回（占位，便于 ai-service 复用） | `true` |
| `CONCEPT_RETRIEVAL_DOMAIN` | 领域 | `math` |
| `CONCEPT_RETRIEVAL_ENTITY_TYPE` | 实体类型 | `MANUAL_CONCEPT` |
| `CONCEPT_RETRIEVAL_CONFIG` | 概念 JSONL 路径 | `ai-service/data/math_concepts/...jsonl` |
| `CONCEPT_RETRIEVAL_WHITELIST` | 白名单路径 | `entity_whitelist_draft.txt` |
| `CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR` | LightRAG 数据目录 | `lightrag_manual_concepts` |
| `CONCEPT_RETRIEVAL_ENTITY_TYPE_PROMPT_FILE` | 可选 entity extraction prompt 文件 | 空 |

legacy 变量（仍兼容，但建议迁移）：`CONFIG` / `WORKING_DIR` / `ENTITY_WHITELIST` / `BATCH_SIZE` / `QUERY` / `MODE`。

覆盖示例：

```bash
# 主路径示例
CONCEPT_RETRIEVAL_CONFIG=/path/concepts.jsonl \
CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR=/path/lightrag_manual_concepts \
CONCEPT_RETRIEVAL_DOMAIN=industrial_training \
CONCEPT_RETRIEVAL_ENTITY_TYPE=MANUAL_CONCEPT \
./run_import.sh

# （可选）历史实验性路径示例
CONCEPT_RETRIEVAL_CONFIG=/path/concepts.jsonl \
CONCEPT_RETRIEVAL_LIGHTRAG_WORKING_DIR=/path/lightrag_custom_kg \
CONCEPT_RETRIEVAL_DOMAIN=industrial_training \
CONCEPT_RETRIEVAL_ENTITY_TYPE=MANUAL_CONCEPT \
./run_import_custom_kg.sh
```

切换领域示例（不动代码）：

```bash
# 工业实训概念（主路径）
CONCEPT_RETRIEVAL_DOMAIN=industrial_training \
CONCEPT_RETRIEVAL_CONFIG=/path/industrial_training_concepts.jsonl \
./run_import.sh

# （可选）历史实验性路径
CONCEPT_RETRIEVAL_DOMAIN=industrial_training \
CONCEPT_RETRIEVAL_CONFIG=/path/industrial_training_concepts.jsonl \
./run_import_custom_kg.sh
```

常用参数：
- 主路径（`run_import.sh`）：`--replace`、`--include-non-correct`、`--disable-whitelist`、`--domain`、`--entity-type`、`--generate-whitelist-only`
- 历史实验性路径（`run_import_custom_kg.sh`）：`--dry-run`、`--dump-custom-kg out.json`、`--batch-size`、`--include-non-correct`、`--disable-whitelist`、`--domain`、`--entity-type`

---

## 命中判定（test_manual_concept_lightrag.py）

`--answer` 命中判定已收紧，避免脏实体误判：

- **强 HIT**：`entity_type ∈ {MANUAL_CONCEPT, MANUAL_MATH_CONCEPT(legacy)}` **且**
  `entity_name` 命中 config 中 `concept_name`；
- **不能单独判 HIT**：`source_type ∈ {manual_concept, manual_math_concept}` 与 chunk `file_path` 命中
  仅作为辅助 reason；
- 公式 / 变量 / 符号 / `Unknown` 类型节点一律不算 HIT（被 entity_type 门槛过滤）；
- `matched` 输出包含 `domain` / `entity_type` / `source_type`；
- `--answer` 内容优先取 `matched["md_content"]`，其次才读 `matched["md_path"]` 文件，
  确保 custom KG 模式无 `md_path` 时也能依据 JSONL 正文回答；
- `STRICT_SYSTEM_PROMPT` 已泛化为「概念讲解助手」（不再写死「数学概念」），保持严格依据资料、
  不引入文档外知识。

---

## 存储

默认使用文件后端（`JsonKV` / `NanoVectorDB` / `NetworkX`），不设
`LIGHTRAG_*_STORAGE` 环境变量即可。默认数据目录：

- 主路径：`/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts`
- 历史实验性：`/home/zj/ZengKingMorphe/ai-service/data/lightrag_manual_concepts_custom_kg`

---

## 与 ai-service 的边界

本子项目只产出 LightRAG `working_dir` 数据。后续 `ai-service` 的 `QueryClassifier`
命中 `concept_explain` 意图后，会读取该 `working_dir` 召回 `concept_context`，
**命中手动概念库后完全跳过 RAGAnything**，只依据 `concept_context` 生成答案。
本任务不修改 `ai-service`。
