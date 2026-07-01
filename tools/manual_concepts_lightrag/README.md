# manual_concepts_lightrag

手动数学概念 → LightRAG 导入 / 测试 / 可视化的独立子项目。

## 数据布局

```
tools/manual_concepts_lightrag/          ← 本子项目（脚本 + .env，自包含配置）
  ├─ import_manual_concepts_lightrag.py  导入手动概念（Markdown 全文 + MANUAL_MATH_CONCEPT 实体）
  ├─ test_manual_concept_lightrag.py     结构化召回 + HIT/MISS 路由 + 严格回答
  ├─ .env                                LLM / embedding / 数据路径配置
  └─ run_*.sh                            便捷启动脚本

ai-service/data/lightrag_manual_concepts/   ← LightRAG 数据目录（复用，勿删）
ai-service/data/math_concepts/*.jsonl       ← config 源文件（绝对路径引用）
```

存储后端：**文件后端**（`JsonKV` / `NanoVectorDB` / `NetworkX`），与现有数据一致；
**故意不设** `LIGHTRAG_*_STORAGE`，避免走 ai-service/.env 里的 Mongo+Milvus+Neo4j。

## 模型配置（见 .env）

- LLM：`Qwen3-32B-AWQ @ http://192.168.100.202:8200/v1`（内网无鉴权）
- Embedding：`BAAI/bge-m3 @ http://192.168.8.233:8092/v1`，dim=1024

## 使用

```bash
# 环境
conda activate morphe

# 1) 导入（默认 --replace，覆盖重建）
./run_import.sh
# 或自定义: CONFIG=...jsonl WORKING_DIR=... ./run_import.sh

# 2) 测试召回 / 严格回答
QUERY="请帮我讲解二项式定理" MODE=local ./run_test.sh --answer
# local 未命中可换 MODE=hybrid / mix

# 3) 可视化（需先构建 WebUI，见下方）
./run_server.sh
# 浏览器: http://localhost:9621
```

## WebUI 可视化（首次需构建）

`lightrag-server` 的 WebUI 前端不会随 pip 包预装，首次使用需构建并部署：

```bash
# 构建
cd /home/zj/RAG-Anything/thirdpart_reps/LightRAG/lightrag_webui
bun install --frozen-lockfile
bun run build
# 部署到 site-packages（供已安装的 lightrag-server 读取）
cp -r dist/* /home/zj/miniconda3/envs/morphe/lib/python3.10/site-packages/lightrag/api/webui/
```

构建完成后 `./run_server.sh`，浏览器打开 http://localhost:9621 即可看到
`ai-service/data/lightrag_manual_concepts` 里的知识图谱（41 个 MANUAL_MATH_CONCEPT 实体
及 LLM 抽取的关联实体/关系）。

## 文件后端 vs 数据库

当前 41 条（衍生 ~311 实体、graph ~370KB）规模远未到数据库的必要阈值（十万级向量、
TB 级数据、多机共享）。文件后端开箱即用、可移植（打包 working_dir 即可迁移）。
若将来扩到上千条且需多实例共享，再拨 `LIGHTRAG_*_STORAGE` 接 Mongo+Milvus+Neo4j。
