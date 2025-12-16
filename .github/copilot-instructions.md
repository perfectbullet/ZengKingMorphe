# 数字员工项目 AI 代理指南

> 依据 `docs/需求补充与完善文档.md`、`requirements.md` 与 `docs/数字员工项目-AI研发工作规划指南.md` 重新梳理，以下指引面向所有 AI 编程助手，确保落地实现与最新需求同步。

## 项目总览
- **形态**：Vue 前端 + Python AI 引擎（FastAPI + LangGraph）；Java 侧仅提供词库管理 API，不参与对话链路。
- **AI 引擎职责**：REST/SSE 对话接口、会话与上下文管理、知识库检索、任务编排、实时联网检索、敏感词检测、日志记录。
- **核心指标**：回答准确性、实时性（CRAG/联网检索）、合规性（敏感词/审计）。

## 技术架构（简化）
```
Vue (SPA)
  │ REST / SSE / WebSocket
FastAPI (Auth + Rate Limit + Error Spec)
  │
  ├─ 对话管理服务：会话/上下文/多轮
  ├─ 知识库管理服务：文档上传→分块→向量化→检索接口
  ├─ 任务编排服务：异步调度、Webhook、后台作业
  │
LangGraph 工作流
  │→ Intent & 实体识别 → 多知识库路由 → 检索融合 (Chroma + ES)
  │→ CRAG/联网检索 → LLM 生成 → 敏感词审核 → SSE 推流
  │
存储：MongoDB（会话/文档/FAQ）、Chroma（向量）、ElasticSearch（全文）
外部：OpenAI/Claude/DeepSeek/Qwen、Java 词库 API、联网搜索（HTTP 工具）
```

## API 与接口规范
- **REST**：`POST /api/chat/message`、`POST /api/chat/stream`(SSE)、`GET/DELETE /api/chat/session/{id}`。所有接口需：
  - JWT + API Key 双模式；签名校验；IP 白名单（可配置）。
  - 限流：令牌桶（每用户/每会话维度）、并发会话上限。
  - 统一错误体：`{code,message,error{type,details}}`，覆盖 400/401/403/404/429/500/503。
- **SSE**：唯一流式通道，事件顺序 fixed（start→token→done/error），需支持客户端中断与异常重试。

## 会话与对话流程
1. 请求校验 → 安全过滤（XSS/SQL/长度/必填）。
2. LangGraph 工作流：
   - 意图/实体识别（Few-shot + 关键词），判断是否实时查询。
   - 多知识库路由：按意图、问题分类或手工优先级选择/并行查询，结果融合 + 去重 + 加权排序。
   - 多轮上下文：MongoDB 存最近 5-10 轮，可配置窗口，超时 30min 自动结束。
3. 检索策略：
   - RAG：Chroma + ElasticSearch → RRF 排序 → 相关性阈值 (<0.75) 触发降级/联网。
   - CRAG/联网：`check_realtime_query` 先行判断（时间/天气/股价等关键词），命中则跳过知识库直接联网；未命中但 RAG 低相关再二次联网。
4. 生成阶段：LLM（OpenAI/Claude/DeepSeek/Qwen）+ 模板注入 + 引用拼接；流式推送。
5. 敏感词/问题检测：
   - Java 平台提供列表接口与 webhook（敏感/专业词）。
   - AI 侧使用 AC 自动机/DFA 做输入、输出双向检测；Critical 阻断、Warning 替换、Notice 仅记录。
6. 记录：对话、检索命中、敏感词触发、联网调用、评估指标全部写入 MongoDB，支持审计与 BI。

## 数据与词库集成
- **MongoDB**：`conversations`、`knowledge_docs`（含 chunks→vector_id）、`faqs`、`intents`、`custom_dictionaries` 等集合必须按文档 schema 写入。
- **Chroma**：`faq_collection`、`doc_collection` 用 text-embedding-3-small；向量 ID 回写到 Mongo。
- **ElasticSearch**：`faq_index`、`doc_index`，启用 IK 分词 + 字段权重 (`question^2`)。
- **词库/敏感词**：
  - 拉取：`GET /api/java/sensitive-words/list`、`GET /api/java/professional-words/list`（过滤参数见文档）。
  - Webhook：`POST /api/ai/sensitive-words/sync-notify` / `professional-words/sync-notify`。需实现缓存、增量更新、审计日志。
  - 专业词库用于查询扩展、实体识别、召回加权。

## 实时检索（CRAG 强化）
- `check_realtime_query`：规则 + 关键词库（时间/天气/股价/...）；命中后直接联网，绕过知识库。
- 联网工具：HTTP 客户端 + JSON 模板，需缓存/限频，并记录失败重试策略。
- 答案融合策略：
  1. 仅联网结果、
  2. 仅知识库、
  3. 混合（需标注来源）。

## 安全与合规
- 输入过滤：必填字段校验、query ≤ 1000 字符、特殊字符转义、路径遍历防护。
- 输出过滤：敏感词/敏感问题、引用合法性检查。
- 审计：保存敏感拦截记录、阈值配置、人工复核入口。

## 代码与实现约定
- **LangGraph**：工作流节点至少包含：`intent_recognition → route_strategy → (rag_search|faq_search|realtime_search) → rerank → answer_generation → quality_check → sensitive_filter`，并暴露 `should_route_realtime`、`need_human` 等条件。
- **FastAPI**：使用 `Depends` 注入鉴权、限流；SSE 用 `EventSourceResponse`；所有接口 async。
- **检索融合**：实现 RRF，加上专业词扩展 & 置信度阈值；低于阈值的答案走转人工话术。
- **任务编排**：长耗时任务走 Celery/Redis（可选）或 asyncio 后台任务；Webhook 通知需幂等。

## 开发阶段（建议）
1. **Phase 1**：FastAPI 框架 + 会话/上下文 + LangGraph 骨架 + SSE。 
2. **Phase 2**：知识库管理（上传/切片/入库）、FAQ/Mongo/Chroma/ES 管道 + 混合检索。 
3. **Phase 3**：敏感/专业词集成、联网检索、质量评估、日志与监控。 
4. **Phase 4**：性能优化、容器化、前端联调、监控告警、部署脚本。

## 参考文件
- `docs/需求补充与完善文档.md`（最新需求、接口、流程）
- `requirements.md`（初版需求）
- `docs/数字员工项目-AI研发工作规划指南.md`（架构规划）
- `docs/关于FAQ问答库.md`、`docs/关于Agent对话风格分类.md`

## AI 代理操作提示
- 任何实现前先查对 `docs/需求补充与完善文档.md` 的字段/接口详情。
- 生成代码需覆盖：鉴权、限流、错误体、日志、SSE、异步/await、敏感词钩子、知识库融合等必备逻辑。
- 回答问题时优先引用具体文件/段落，提供 JSON/HTTP 示例或 LangGraph 代码片段。
- 容器化与部署脚本必须包含 Vue + ai-service + Mongo + Chroma + ES。
- 若需求存在歧义，记录假设并在输出中显式说明，以便后续复核。
