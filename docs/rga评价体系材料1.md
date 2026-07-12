## 1. RAG 召回效果评价

**结论：召回效果是好的，约 8/10。**

这次 RAG 已经把关键材料召回来了，不是召回失败。你提供的 context 显示，本次问题为“平铺珐琅工艺的基本制作流程是什么？”，`mode=hybrid`，最终上下文包含 **13 个实体、11 条关系、7 个最终文档块**。

最关键的是，召回到了真正能回答问题的核心 chunk：

```text
01珐琅工艺_full.md#L1087-L1090
```

其中明确写到：

```text
金属底板经过凸形处理和酸洗后，就可以正式开始珐琅的烧制。
平铺珐琅的烧制过程包括背釉的烧制、底釉的烧制和正面图案的烧制几个步骤。
```

这个 chunk 对“基本制作流程”是直接命中的。

另外，还召回了：

```text
4.2.1 背釉和底釉的烧制
4.2.2 正面图案的制作 part_01
4.2.2 正面图案的制作 part_02
3.1 平铺珐琅工艺
```

这些能够补足“背釉/底釉/正面图案制作”的操作细节。

所以这次 RAG 的核心判断是：

```text
实体命中：好
关键词命中：好
核心 chunk 命中：好
rerank 排序：好
引用完整性：好
上下文可回答性：好
```

### 小问题

召回里仍然有一点噪声。例如第 6 个 chunk 是“珐琅工艺的起源和发展”，对“平铺珐琅基本流程”不是核心 chunk，只是背景材料。

另外，KG entity 描述里有一些泛化或可能误导的内容，比如“平铺珐琅相对简单，通常只需一次填色和烧制即可完成”这种描述，容易影响模型生成。但这不是 chunk 召回问题，而是 KG 摘要质量问题。

---

## 2. 模型回复质量评价

**结论：模型回复质量很差，约 2/10。**

你提供的最终回答是：

```text
1. 制胎
2. 设计图案
3. 勾线
4. 填珐琅
5. 烧制
6. 打磨与抛光
7. 镀层处理
```



这个回答的问题很严重：

### 第一，混入了教材外通用工艺

“制胎”“勾线”“打磨与抛光”“镀层处理”不是这次召回上下文支持的平铺珐琅基本流程。

尤其是 **打磨与抛光** 明显错误，因为召回 chunk 明确说平铺珐琅工艺“烧制完成后不进行打磨”。

### 第二，没有利用最关键的召回内容

正确流程应该基于教材整理为：

```text
金属底板经过凸形处理和酸洗
→ 背釉烧制
→ 底釉烧制
→ 正面图案制作和烧制
```

正面图案制作内部又包括：

```text
绘制或拓印图案线条
→ 挑选、研磨、清洗釉料
→ 按图案填放釉料
→ 晾干
→ 入炉烧制
```

这些都在召回 chunk 中出现了，但模型没有忠实提取。

### 第三，回答没有引用意识

它说“不同地区或工艺流派可能会有差异”，这句话没有来自当前教材上下文，是典型的泛化补全。

所以这次不是“资料不足”，也不是“召回差”，而是：

```text
模型生成阶段没有被有效约束住。
它看到了上下文，但仍然按通用珐琅/传统工艺常识回答。
```

---

## 3. 总体判断

这次 case 可以这样定性：

| 项目           |  评价 | 说明                  |
| ------------ | --: | ------------------- |
| 路由           |  合格 | 进入了 lightrag_file   |
| entity 命中    |  良好 | 命中平铺珐琅工艺、平铺珐琅、背釉烧制等 |
| chunk 召回     |  良好 | 核心章节都召回了            |
| rerank       |  良好 | 关键 chunk 排在前面       |
| context 可回答性 |  良好 | 上下文足够回答             |
| 模型忠实性        |  很差 | 加入了教材外步骤            |
| 模型完整性        |  一般 | 给了流程，但不是教材流程        |
| 幻觉控制         |  很差 | 打磨、抛光、镀层属于明显幻觉      |
| 最终可用性        | 不合格 | 不能交给用户              |

一句话：

```text
RAG 召回已经达标，模型回答没有达标。
下一步重点不是继续调召回，而是建立“回答质量评价体系”和“生成约束/后处理检查”。
```

---

# 4. 你需要建立什么评价体系？

你应该把评价拆成两套：**RAG 召回评价** 和 **模型回答评价**。不要混在一起。

## A. RAG 召回评价体系

每个问题记录这些字段：

```text
question
expected_books
expected_keywords
expected_source_hint
entered_lightrag_file
matched_entities
low_level_keywords
high_level_keywords
final_chunks_count
final_chunk_paths
rerank_scores
retrieved_books
cross_book_suspicious
```

然后人工或脚本打分。

### RAG 召回评分建议，满分 10 分

| 维度          | 分值 | 说明                    |
| ----------- | -: | --------------------- |
| 路由正确        |  1 | 工训题进入 LightRAG，通识题不进入 |
| entity 命中   |  2 | 是否命中核心术语              |
| 关键词合理       |  1 | hl/ll keywords 是否符合问题 |
| 核心 chunk 命中 |  3 | 是否召回能直接回答问题的原文        |
| rerank 排序   |  1 | 核心 chunk 是否排在前 5      |
| 引用完整        |  1 | citations 是否非空且可追溯    |
| 跨书污染控制      |  1 | 总库后是否召回错误教材           |

这次 RAG 大概：

```text
8/10
```

扣分点主要是有少量背景 chunk 和 KG 摘要噪声。

---

## B. 模型回答评价体系

每个回答记录这些字段：

```text
answer
must_include_hit
must_not_include_hit
groundedness_score
completeness_score
faithfulness_score
hallucination_score
format_score
citation_usage
final_pass
```

### 回答质量评分建议，满分 10 分

| 维度     | 分值 | 说明                     |
| ------ | -: | ---------------------- |
| 直接回答问题 |  1 | 是否正面回答用户问题             |
| 依据上下文  |  2 | 是否能被召回材料支撑             |
| 关键点完整  |  2 | must_include 是否覆盖      |
| 无错误补全  |  2 | must_not_include 是否未出现 |
| 结构清晰   |  1 | 是否条理清楚                 |
| 术语准确   |  1 | 是否使用教材术语               |
| 不误拒答   |  1 | 有材料时不能说资料不足            |

这次模型回答大概：

```text
2/10
```

主要问题是命中了多个 `must_not_include`：

```text
制胎
打磨
抛光
镀层
```

这些在你的 QA 文件里本来就应该列为禁止项。模型最终回答正好踩雷。

---

# 5. 你接下来应该做什么？

## 第一步：把 QA 文件做成“可自动判分”的格式

比如：

```text
## QA-001
question: 平铺珐琅工艺的基本制作流程是什么？
type: 流程类
expected_route: lightrag_file
expected_books: 01
allow_cross_book: false
expected_keywords: 平铺珐琅工艺, 背釉, 底釉, 正面图案烧制
must_include: 凸形处理, 酸洗, 背釉烧制, 底釉烧制, 正面图案烧制
must_not_include: 制胎, 打磨, 抛光, 镀层, 包装
source_hint: 01珐琅工艺_full.md#L1087-L1090; 01珐琅工艺_full.md#L1091-L1139; 01珐琅工艺_full.md#L1179-L1302
```

这次回答只要脚本扫一下，就能判定：

```text
must_include 命中不足
must_not_include 命中：制胎、打磨、抛光、镀层
final_pass=false
```

---

## 第二步：报告里必须分开给两个分数

建议每题报告输出：

```text
retrieval_score: 8/10
answer_score: 2/10
final_pass: false
failure_type: generation_hallucination
```

失败类型建议固定成这些：

```text
route_error
entity_miss
keyword_error
chunk_miss
rerank_error
cross_book_pollution
generation_hallucination
generation_incomplete
generation_over_refusal
citation_error
knowledge_gap
```

这次 case 的 failure_type 是：

```text
generation_hallucination
```

不是：

```text
chunk_miss
rerank_error
knowledge_gap
```

---

## 第三步：加一个“答案后处理审查”

生成完以后，用规则检查：

```text
如果 answer 命中 must_not_include，则标记高风险；
如果 answer 没命中 must_include，则标记不完整；
如果 context 有核心 chunk，但 answer 说“资料不足”，标记 over_refusal；
```

不要一开始就用 LLM Judge。先用规则，稳定、便宜、可解释。

---

## 第四步：再考虑 LLM Judge

等规则体系跑通后，可以加 LLM Judge 做补充，判断：

```text
回答是否被上下文支持？
有没有教材外推断？
有没有遗漏核心步骤？
```

但 LLM Judge 只做辅助，不要替代 `must_include/must_not_include/source_hint`。

---

# 6. 当前案例的最终评价

```text
问题：平铺珐琅工艺的基本制作流程是什么？

RAG 召回：良好，8/10
原因：命中了 4.2、4.2.1、4.2.2 等核心章节，引用完整，rerank 分数高，上下文足够回答。

模型回复：不合格，2/10
原因：混入制胎、勾线、打磨抛光、镀层处理等教材外流程；尤其“打磨与抛光”与平铺珐琅“不进行打磨”的教材内容冲突。

主要失败类型：
generation_hallucination

下一步优化重点：
不是召回，而是回答质量评测、must_include/must_not_include 自动检查、生成后审查。
```
