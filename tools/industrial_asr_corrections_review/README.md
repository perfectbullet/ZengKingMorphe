# 工业实训 ASR 纠错规则维护工具

这是一个独立、本地使用的 FastAPI 小工具，用浏览器结构化维护：

```text
ai-service/app/services/conversation/industrial_asr_corrections.json
```

工具位于 `tools/industrial_asr_corrections_review/`，不会注册到
`ai-service/main.py`，不访问 MongoDB、ElasticSearch、LightRAG 或模型服务。

## 启动

从仓库根目录执行：

```bash
/home/zj/miniconda3/envs/morphe/bin/python \
  tools/industrial_asr_corrections_review/app.py \
  --json-path ai-service/app/services/conversation/industrial_asr_corrections.json \
  --host 127.0.0.1 \
  --port 8767
```

默认 JSON 路径就是上面的文件；默认监听 `127.0.0.1:8767`。浏览器打开
`http://127.0.0.1:8767`。局域网临时使用时，将 `--host` 改为 `0.0.0.0`。

启动前工具会检查文件存在、为普通文件、可按 UTF-8 读取、可解析为 JSON object，且
顶层包含 `version` 和 `rules`；不满足时会直接以清晰错误退出，不会启动空页面。

## 页面功能

- 左侧按 `id`、`canonical`、`variants`、`context.any/all/none` 搜索，并可筛选启用状态和匹配模式。
- 结构化编辑 `id`、启用开关、标准术语、错误词、匹配模式、优先级和上下文条件。
- 支持新建、复制、删除、上一条/下一条、完整校验、从磁盘重新加载和下载 JSON。
- 切换规则、重新加载或关闭页面前，若存在未保存表单修改会要求确认；`Ctrl+S` 保存当前规则。
- 不显示的顶层字段、规则未知字段会随完整对象保留，规则和列表顺序也不会调整。

新建 direct 规则不写 `context`；contextual 规则会使用 `any/all/none` 空列表初始化。
文本框一行一个值，保存时去掉空行、去重并保留首次出现顺序。

## JSON 字段与校验

- `direct`：错误词精确命中即替换。
- `contextual`：除精确命中外，还须根据 `context.any/all/none` 判断。
- `priority`：确定性规则冲突的优先级。

错误（例如重复 ID、空 canonical、非法 variants 或 contextual 缺少 context）会阻止保存。
警告（例如重复 variant、direct/contextual 重复 variant、重复 canonical、禁用规则）仅展示，不自动合并或改名。

## 保存与备份

每次正式保存均先完整校验。校验通过后，工具会先在原 JSON 同目录创建：

```text
bak_industrial_asr_corrections/
```

备份文件格式为：

```text
industrial_asr_corrections_YYYYMMDD_HHMMSS_ffffff.json
```

备份使用保存前的磁盘内容；备份失败时不会覆盖原文件。工具不自动删除备份，也没有网页回滚功能。人工恢复示例：

```bash
cp \
  ai-service/app/services/conversation/bak_industrial_asr_corrections/industrial_asr_corrections_20260715_183012_123456.json \
  ai-service/app/services/conversation/industrial_asr_corrections.json
```

保存后的 JSON 为 UTF-8、中文不转义、缩进为两个空格并以换行结尾。

## 注意事项

- 此工具按单人临时维护设计，不支持多人同时编辑、冲突合并、网页回滚或历史审计。
- 无登录、Token、API Key 或权限控制；仅应在可信本地/测试网络使用。
- 修改 JSON 后，`ai-service` 当前的进程重启加载规则；本工具不为业务服务增加热加载机制。
- 工具不修改 `ai-service` 业务路由、认证、数据库或启动流程。
