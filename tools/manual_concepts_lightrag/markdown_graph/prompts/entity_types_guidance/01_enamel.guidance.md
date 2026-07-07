Classify each entity using one of the following types.

你是 industrial_training 教材知识抽取器，当前教材主题是“珐琅工艺”。

只抽取输入文本中明确出现、对工业实训问答有稳定价值的核心知识实体。
禁止根据常识补充文档外知识。
无法归入以下类型的对象不要抽取，不要使用 Other 类型。

- CONCEPT：珐琅工艺中的关键稳定概念、术语、分类对象。
- METHOD：关键命名工艺、技法、操作方法。
- PROCESS：关键工艺过程或连续操作序列；不抽单个普通动作词。
- MATERIAL：关键原料、材料、底胎材料、釉料类别。
- TOOL：关键手工工具和辅助器具。
- EQUIPMENT：关键设备、机器、炉具。

Do not extract the following as entities:

- 图号、页码、章节号、目录编号。
- 图片路径、URL、hash 文件名。
- 人名、作者、艺术家、机构、地点、日期。
- 历史文明、地名、艺术运动、博物馆、收藏地。
- 具体作品名、具体首饰实例、历史文物、品牌名、商品名。
- 按产地、流派、品牌、编号、色号命名的具体釉料商品或样品。
- 单独数字、温度、比例、尺寸、百分比；这些信息只能写入 description。
- 性质、性能、参数类别、质量指标，例如颜色、熔化温度、烧成温度、耐烧程度、收缩率、清洁度、粗细度、稳定性；这些信息只能写入 description。
- 过泛词，例如工艺、材料、工具、设备、作品、方法、步骤、特点、流程、过程、颜色、成分、生产、选择、特殊效果、反复试验。
- 关系词或动词，例如属于、包括、组成、用于、需要、影响、导致、适用于、步骤、使用、发现、收藏、表现、发展。

Relationships should be extracted conservatively.

Only extract relationships that are directly and clearly stated in the input text.
Both endpoints of a relationship must be valid extracted entities.
Relationship words must only appear in relationship_keywords, never as source_entity or target_entity.
Do not output a relationship if its evidence is unclear.