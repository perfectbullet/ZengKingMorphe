# MinerU JSON 解析与 RAG 利用方案

## 用户需求确认
- **图片处理**: 使用VLM生成图片描述，参与向量化检索
- **输出格式**: 既要CLI工具（独立运行），又要集成到项目（文档上传时自动调用）

---

## 一、JSON 文件结构分析

### 1.1 顶层结构

```json
{
    "pdf_info": [...],      // 数组，每页一个元素
    "_backend": "vlm",      // 后端类型
    "_version_name": "2.6.4" // MinerU版本
}
```

### 1.2 页面结构 (pdf_info[i])

| 字段 | 类型 | 说明 |
|------|------|------|
| `page_idx` | int | 页码(从0开始) |
| `page_size` | [int, int] | 页面尺寸 [宽, 高] |
| `para_blocks` | array | 段落数组 |
| `discarded_blocks` | array | 被丢弃的块 |

### 1.3 段落块结构 (para_blocks[i])

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 块类型: `text`/`title`/`image`/`list`/`table`/`footer` |
| `bbox` | [x1,y1,x2,y2] | 边界框坐标(页面上的位置) |
| `index` | int | 块索引 |
| `angle` | float | 旋转角度 |
| `lines` | array | 文本行(非image类型) |
| `blocks` | array | 嵌套块(image类型特有) |

### 1.4 文本内容结构

```json
"lines": [
    {
        "bbox": [x1, y1, x2, y2],
        "spans": [
            {
                "bbox": [x1, y1, x2, y2],
                "type": "text",
                "content": "实际文本内容"
            }
        ]
    }
]
```

### 1.5 图片内容结构

```json
{
    "type": "image",
    "bbox": [x1, y1, x2, y2],
    "blocks": [
        {
            "type": "image_body",
            "lines": [
                {
                    "spans": [
                        {
                            "type": "image",
                            "image_path": "https://cdn-mineru.openxlab.org.cn/..."
                        }
                    ]
                }
            ]
        }
    ]
}
```

### 1.6 数据统计（实际文件）

| 类型 | 数量 | 说明 |
|------|------|------|
| 总页数 | 150 | |
| text | 623 | 普通文本段落 |
| image | 474 | 图片 |
| title | 268 | 标题 |
| list | 50 | 列表 |

---

## 二、字段含义解释

| 字段 | 含义 | RAG价值 |
|------|------|---------|
| `type` | 元素类型 | 高-用于结构化分块、保持上下文 |
| `content` | 文本内容 | 核心-向量嵌入的基础 |
| `bbox` | 页面位置坐标 | 中-可用于布局感知检索 |
| `page_idx` | 页码 | 高-文档溯源、引用 |
| `image_path` | 图片URL | 高-多模态RAG |
| `title` | 标题类型 | 高-章节结构、文档导航 |
| `list` | 列表类型 | 中-保持列表完整性 |

---

## 三、实现方案

### 3.1 脚本文件结构

```
ai-service/app/services/
├── mineru_json_parser.py      # JSON解析器（新增）
└── mineru_aware_chunking.py   # 结构化分块（新增）
```

### 3.2 JSON解析器功能

```python
class MinerUJsonParser:
    """MinerU JSON解析器"""

    def parse_file(self, json_path: str) -> MinerUDocument:
        """解析JSON文件为结构化对象"""

    def extract_text_by_page(self, page_idx: int) -> str:
        """提取指定页的文本内容"""

    def extract_titles(self) -> List[Dict]:
        """提取所有标题及其层级"""

    def extract_images(self) -> List[Dict]:
        """提取所有图片信息"""

    def to_enhanced_markdown(self) -> str:
        """生成带结构标记的Markdown"""
```

### 3.3 RAG 利用方案

#### 方案1: 结构化分块（推荐）

利用 `type` 字段进行智能分块：
- 按 `title` 边界切分文档
- 保持 `list` 完整性
- 关联 `image` 与前后文

#### 方案2: 多模态RAG

1. 提取图片URL和上下文
2. 使用VLM生成图片描述（可选）
3. 将图片描述与文本一起向量化
4. 检索时返回文本+图片

#### 方案3: 元数据增强

在ChromaDB/ElasticSearch中存储额外元数据：
- `page_idx`: 页码
- `block_types`: 包含的块类型
- `has_images`: 是否包含图片
- `title_path`: 标题路径

---

## 四、实现步骤

### Step 1: 创建JSON解析服务模块

**文件**: `ai-service/app/services/mineru_json_parser.py`

**功能**:
- 读取并解析MinerU JSON文件
- 提取文本、标题、图片等结构化数据
- 提供便捷的查询接口

### Step 2: 创建图片描述生成器

**文件**: `ai-service/app/services/mineru_image_handler.py`

**功能**:
- 提取JSON中的图片URL
- 调用VLM API生成图片描述（支持GPT-4V/Qwen-VL）
- 缓存图片描述到MongoDB
- 将图片描述与上下文文本组合

### Step 3: 创建结构化分块器

**文件**: `ai-service/app/services/mineru_aware_chunking.py`

**功能**:
- 基于块类型进行智能分块
- 保留文档结构信息
- 关联图片描述与文本
- 生成增强的元数据

### Step 4: 创建CLI工具

**文件**: `scripts/parse_mineru_json.py`

**功能**:
- 独立运行的命令行工具
- 支持解析单个或批量JSON文件
- 输出结构化数据到文件
- 可选生成图片描述

```bash
# 使用示例
python scripts/parse_mineru_json.py input.json --output output.json
python scripts/parse_mineru_json.py input.json --caption-images  # 生成图片描述
```

### Step 5: 集成到文档处理流程

**修改文件**: `ai-service/app/services/document_service.py`

**改动点**:
- 上传时检测是否有对应的JSON文件（同目录同文件名.json）
- 如果有，调用MinerU解析器处理
- 使用结构化分块器生成chunks
- 扩展MongoDB schema存储额外元数据

### Step 6: 扩展数据模型

**修改文件**: `ai-service/app/models/database.py`

**新增字段**:
```python
class DocumentChunkModel:
    # 现有字段...
    page_idx: int                      # 页码
    block_types: List[str]             # 块类型 ['text', 'title']
    image_references: List[str]        # 图片URL列表
    image_captions: List[str]          # 图片描述列表
    title_path: List[str]              # 标题路径（面包屑）
```

### Step 7: 扩展RAG检索

**修改文件**: `ai-service/app/services/rag_service.py`

**改动点**:
- 检索结果返回图片引用和描述
- 支持按页码过滤
- 支持按块类型过滤

### Step 8: 添加配置项

**修改文件**: `ai-service/app/core/config.py`

**新增配置**:
```python
mineru_json_enabled: bool = True          # 启用JSON解析
mineru_image_captioning: bool = True       # 启用图片描述生成
mineru_vlm_api_key: Optional[str] = None  # VLM API密钥
mineru_vlm_base_url: Optional[str] = None # VLM API地址
mineru_vlm_model: str = "qwen-vl-max"     # VLM模型
```

---

## 五、文件结构

```
ai-service/
├── app/
│   ├── services/
│   │   ├── mineru_json_parser.py      # JSON解析器（新增）
│   │   ├── mineru_image_handler.py    # 图片描述生成（新增）
│   │   ├── mineru_aware_chunking.py   # 结构化分块（新增）
│   │   ├── document_service.py        # 修改：集成JSON解析
│   │   └── rag_service.py             # 修改：返回图片信息
│   ├── models/
│   │   └── database.py                # 修改：扩展数据模型
│   └── core/
│       └── config.py                  # 修改：新增配置项
└── scripts/
    └── parse_mineru_json.py           # CLI工具（新增）
```

---

## 六、验证方式

1. **CLI工具测试**:
   ```bash
   python scripts/parse_mineru_json.py input.json --output output.json
   # 检查输出是否包含文本、标题、图片等结构化信息
   ```

2. **图片描述测试**:
   ```bash
   python scripts/parse_mineru_json.py input.json --caption-images
   # 检查图片描述是否正确生成
   ```

3. **集成测试**:
   - 上传带JSON的文档到知识库
   - 检查MongoDB中是否存储了结构化元数据
   - 测试检索结果是否包含页码、图片信息

4. **RAG效果测试**:
   - 对比使用JSON前后的检索质量
   - 验证图片描述是否能增强检索




实现完成总结
创建的文件
ai-service/app/services/mineru_json_parser.py - MinerU JSON解析器

解析MinerU导出的JSON文件
提供结构化数据访问接口（文本、标题、图片）
支持生成增强版Markdown
ai-service/app/services/mineru_image_handler.py - 图片描述生成器

支持多种VLM后端（OpenAI GPT-4V、Qwen-VL、自定义）
图片描述缓存（MongoDB + 内存）
批量异步处理
ai-service/app/services/mineru_aware_chunking.py - 结构化分块器

按标题分块（by_title）
按页分块（by_page）
混合策略（hybrid）
多模态分块支持（图片+文本）
scripts/parse_mineru_json.py - CLI工具

独立运行的命令行工具
支持解析、Markdown生成、分块、图片描述生成
修改的文件
app/models/database.py - 扩展数据模型

DocumentChunkModel: 添加page_idx, page_indices, block_types, image_references, image_captions, title_path, structure_level
MinerUImageCaptionModel: 图片描述缓存模型
MinerUStructuredModel: MinerU结构化文档模型
app/core/config.py - 新增配置项

mineru_json_enabled: 启用JSON解析
mineru_structure_aware_chunking: 结构化分块
mineru_chunking_strategy: 分块策略
mineru_include_images: 包含图片
mineru_image_captioning: 图片描述生成
mineru_vlm_*: VLM相关配置
app/services/document_service.py - 集成MinerU JSON解析

自动查找同目录的MinerU JSON文件
使用结构化分块替代传统分块
存储MinerU元数据到ChromaDB/ElasticSearch
app/services/rag_service.py - RAG检索返回结构化信息

返回结果包含页码、图片、标题路径等信息
使用方式
CLI工具使用：


# 解析JSON
python scripts/parse_mineru_json.py input.json --output result.json

# 生成增强Markdown
python scripts/parse_mineru_json.py input.json --markdown --output enhanced.md

# 分块
python scripts/parse_mineru_json.py input.json --chunk --chunks-output chunks.json

# 生成图片描述
python scripts/parse_mineru_json.py input.json --caption-images --vlm-api-key YOUR_KEY
项目集成：

将MinerU JSON文件与PDF放在同目录
上传文档时自动检测并使用JSON解析
启用mineru_json_enabled配置