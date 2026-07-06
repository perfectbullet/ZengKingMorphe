# 工业实训教材统一 KG 抽取 Base Prompt

schema_version: industrial_training_kg_schema.v1

你正在从工业实训教材中抽取统一知识图谱。所有教材共享同一个 schema 和 working_dir。教材级 profile 只能补充术语、判定口径、正反例及更严格的禁抽规则，不能新增实体类型或关系类型。

## 1. 抽取目标

按以下优先级抽取：

1. 保证问答准确；
2. 结构化工艺流程；
3. 清晰表达工艺、材料、工具、设备、缺陷、产品之间的知识关系；
4. 保留 source evidence 以便溯源；
5. 支持图文 evidence 检索；
6. 支持多本教材在同一图谱中融合；
7. 避免图号、图片路径、页码、编号、人名和单独数值污染 KG。

只抽取输入文本明确表达的知识，不补充文档外知识。

## 2. 允许实体类型

只允许以下十种实体类型：

- CONCEPT：概念、术语、定义对象、分类对象、工艺总称、设计原则、理论概念。
- METHOD：命名工艺、技法、操作方法、工艺技术方案。命名工艺和方法优先标为 METHOD。
- PROCESS：具体工艺过程、连续操作序列、生产流程、工序链。
- MATERIAL：材料、原料、介质、化学材料、半成品材料。
- TOOL：手工工具、辅助器具、量具、夹具、绘图工具。
- EQUIPMENT：设备、机器、炉、机床、加工装置。
- PROPERTY：性能、性质、技术要求、参数类别、质量指标。单独数值写入 description/evidence，不作为实体。
- DEFECT：缺陷、问题、失效现象、质量问题。
- PRODUCT：产品、作品、零件、工件、工艺样件、首饰品类。
- SAFETY_REQUIREMENT：安全要求、注意事项、防护要求、操作禁忌。

类型判定示例：

- 平铺珐琅工艺、干筛法、爪镶、雕蜡 -> METHOD
- 釉料研磨和清洗过程、金属底板预处理、宝石镶嵌流程 -> PROCESS
- 珐琅釉料、红铜、蜡、宝石、焊药 -> MATERIAL
- 研钵、锉刀、蜡刀、游标卡尺 -> TOOL
- 珐琅炉、压片机、车床、铸造机 -> EQUIPMENT
- 烧成温度、透明度、尺寸精度、透视准确性 -> PROPERTY
- 砂眼、开裂、毛刺、错位、透视错误 -> DEFECT
- 珐琅戒指、蜡模、玉雕作品、首饰效果图 -> PRODUCT
- 高温操作注意事项、通风要求、用电安全 -> SAFETY_REQUIREMENT

## 3. 禁止实体类型

禁止输出：

- FIGURE
- PERSON
- AUTHOR
- ORGANIZATION
- LOCATION
- DATE
- PAGE
- STEP
- NUMBER
- UNKNOWN

如果无法判断实体类型，不要输出该实体，绝不能用 UNKNOWN 兜底。

## 4. 允许关系类型

只允许以下十六种关系类型：

- 属于：分类、归属、上下位关系。
- 包括：整体包含子类、组成部分或知识点。
- 组成：结构组成、材料组成、部件组成。
- 用于：工具、材料、设备的用途。
- 需要：工艺、过程、产品对材料、工具、条件、要求的依赖。
- 影响：性能、参数、环境对结果的影响。
- 导致：原因导致缺陷、问题或结果。
- 适用于：方法、材料、工具的适用对象。
- 步骤：工艺到步骤或流程到子步骤。
- 前置步骤：步骤顺序中的前置关系。
- 后续步骤：步骤顺序中的后续关系。
- 对比：方法、材料、效果之间的比较。
- 注意事项：工艺或操作对应的注意事项与安全要求。
- 产生：工艺、过程产生效果、产品或缺陷。
- 改善：措施对质量、缺陷、性能的改善。
- 防止：措施防止缺陷或风险。

关系示例：

- 平铺珐琅工艺 - 属于 -> 珐琅工艺
- 珐琅工艺 - 包括 -> 掐丝珐琅工艺
- 珐琅炉 - 用于 -> 烧制
- 平铺珐琅工艺 - 需要 -> 珐琅釉料
- 清洁度 - 影响 -> 附着性
- 杂质 - 导致 -> 砂眼
- 平铺珐琅工艺 - 步骤 -> 金属底板预处理
- 清洗 - 前置步骤 -> 施釉
- 施釉 - 后续步骤 -> 烧制
- 透明釉料 - 对比 -> 不透明釉料
- 清洗 - 改善 -> 附着性
- 玻璃罩 - 防止 -> 灰尘污染

不得创造 schema 之外的关系类型。

## 5. 禁止抽取对象

### 5.1 图号与图片

纯图号（图3-1、图 8-50）不是实体。不得创建 FIGURE 实体。图号只写入 evidence.figure_no，图注写入 evidence.caption/raw_caption，图片路径写入 metadata。

图片路径、URL、hash 文件名不是实体，例如：

- images/xxx.jpg
- ./images/xxx.png
- https://cdn-mineru...
- 5e705c21a05b6fb9ad8f7c8ffa5bf684.jpg

### 5.2 章节、页码与步骤编号

第3章、CHAPTER 03、3.1、10.3.2、043、页码、STEP 01、操作步骤1、步骤1均不得作为实体。目录信息写入 chunk metadata，步骤编号可写入 description/evidence。

### 5.3 人名与组织

作者、艺术家、设计师、感谢名单、译者、编者、出版社人员姓名默认不抽取。工业实训 KG 第一版不做人名、作者、组织、地点知识图谱。

### 5.4 单独数值

850摄氏度、900℃、40%、1:1、0.5mm、043 等单独温度、百分比、比例、尺寸、页码不得作为实体。

正确做法：抽取“烧制温度”“酸液浓度”“线描稿比例”等 PROPERTY，在 description 或 evidence 中保存具体数值。

### 5.5 过泛实体

避免抽取“工艺、方法、材料、工具、设备、作品、图案、效果、操作、注意事项”等过泛名称，优先选择上下文中的具体稳定名称。

## 6. Evidence 规则

evidence 是实体、关系和回答的证据来源，不是核心 KG 实体。可包含：

- chunk_id
- doc_id
- file_path
- catalog_index
- catalog_title
- heading_path
- source_md_path
- figure_no
- caption
- raw_caption
- image_path
- page_idx
- bbox
- source_missing_status

最低限度必须保留 source_id 和 file_path。图号只进入 evidence.figure_no，图注只进入 caption/raw_caption，图片路径只进入 evidence.image_path 或 chunk.images metadata。不得把 evidence 字段值抽成实体节点。

## 7. source_missing 规则

如果目录项标记为 source_missing：

1. 不得用相邻章节补全；
2. 不得用同名或近似章节替代；
3. 回答时必须说明该目录项正文缺失；
4. 可以提示存在相关但不同章节的内容；
5. 不得将相关章节说成缺失目录项本身。

## 8. 输出质量要求

1. 实体名必须短、准、稳定；
2. 只输出允许的实体类型和关系类型；
3. 不输出 UNKNOWN、FIGURE、人名、图片路径、纯图号、单独数值或目录编号实体；
4. 命名工艺优先 METHOD，连续过程优先 PROCESS；
5. TOOL 与 EQUIPMENT 必须严格区分；
6. 产品、作品、工件和工艺样件可以抽为 PRODUCT；
7. 所有知识必须来自输入文本，不补充文档外知识；
8. 图注信息进入 evidence；
9. source_missing 不得被自动补全；
10. 严格保留 LightRAG 要求的实体/关系输出格式，不要改变 tuple delimiter、record delimiter、completion delimiter 或字段顺序。
