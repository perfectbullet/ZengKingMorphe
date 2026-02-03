
data/01高中数学必修第一册/
├── textbook_structured.json      # Stage: convert 输出
├── merged_structured.json        # Stage: merge 输出
├── extract_metadata.json         # Stage: extract_metadata 输出
├── generate_qa.json              # Stage: generate_qa 输出
└── teaching_script_generate.json # Stage: teaching_script 输出

---
1. textbook_structured.json (convert 阶段)

{
"book_title": "高中数学必修第一册",
"chapters": [
    {
    "chapter_title": "第一章集合与常用逻辑用语",
    "chapter_number": "第一章",
    "sections": [
        {
        "section_title": "1.1集合的概念",
        "section_number": "1.1",
        "content": "Markdown 格式的章节内容（包含公式、图片等）",
        "content_parts": [
            {
            "text": "文本内容",
            "text_level": 1,
            "type": "text|equation|table|image"
            }
        ]
        }
    ]
    }
]
}

---
2. merged_structured.json (merge 阶段)

与 textbook_structured.json 结构相同，但：
- 子小节已合并到主小节
- 标题已清理（去除空格和 \u3000）
- 移除了 content_parts 字段

{
"book_title": "高中数学必修第一册",
"chapters": [
    {
    "chapter_title": "第一章集合与常用逻辑用语",
    "chapter_number": "第一章",
    "sections": [
        {
        "section_title": "1.1集合的概念",
        "section_number": "1.1",
        "content": "合并后的 Markdown 内容"
        }
    ]
    }
]
}

---
3. extract_metadata.json (extract_metadata 阶段)

提取了7个元数据字段：

{
"book_title": "高中数学必修第一册",
"chapters": [
    {
    "chapter_title": "第一章集合与常用逻辑用语",
    "chapter_number": "第一章",
    "sections": [
        {
        "section_title": "1.1集合的概念",
        "section_number": "1.1",
        "content": {
            "basic_concepts": ["概念1", "概念2", ...],      // 基础概念
            "thinking": ["思考题1", "思考题2", ...],        // 思考
            "practice": ["练习1", "练习2", ...],            // 练习
            "review_and_reflection": ["复习内容1", ...],    // 复习巩固
            "examples": ["例题1", "例题2", ...],            // 例题
            "methods": ["方法1", "方法2", ...],             // 方法
            "key_points": ["重点1", "重点2", ...]           // 重点
        }
        }
    ]
    }
]
}

每个字段都是字符串列表。

---
4. generate_qa.json (generate_qa 阶段)

基于元数据生成口语化问答对：

{
"book_title": "高中数学必修第一册",
"chapters": [
    {
    "chapter_title": "第一章集合与常用逻辑用语",
    "chapter_number": "第一章",
    "sections": [
        {
        "section_title": "1.1集合的概念",
        "section_number": "1.1",
        "qa_pairs": [
            {
            "question": "老师，到底什么是集合呀？...",
            "answer": "其实很简单，我们把研究的..."
            },
            ...
        ]
        }
    ]
    }
]
}

---
5. teaching_script_generate.json (teaching_script 阶段)

生成互动式教学脚本：

{
"book_title": "高中数学必修第一册",
"chapters": [
    {
    "chapter_title": "第一章集合与常用逻辑用语",
    "chapter_number": "第一章",
    "sections": [
        {
        "section_title": "1.1集合的概念",
        "section_number": "1.1",
        "student_question": "老师，我在做练习题的时候...",
        "teaching_script": "你提的这个问题非常关键，很多同学..."
        }
    ]
    }
]
}

---
数据流程

原始 PDF 解析 JSON (*_content_list.json)
            ↓
    [convert] textbook_structured.json
            ↓
    [merge] merged_structured.json
            ↓
[extract_metadata] extract_metadata.json
            ↓
        ┌────┴────┐
        ↓         ↓
[generate_qa]  [teaching_script]
generate_qa.json  teaching_script_generate.json

---
通用字段说明

| 字段           | 类型          | 说明                          |
|----------------|---------------|-------------------------------|
| book_title     | string        | 教材名称                      |
| chapter_title  | string        | 章标题                        |
| chapter_number | string        | 章编号（如"第一章"）          |
| section_title  | string        | 小节标题（如"1.1集合的概念"） |
| section_number | string        | 小节编号（如"1.1"）           |
| content        | string/object | 内容字符串或元数据字典        |
