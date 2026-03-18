"""
使用 Markdown 文件测试 RAG 系统
"""
import asyncio
import re
from pathlib import Path
from src.rag_system import RAGSystem


def clean_text(text: str) -> str:
    """清理文本：移除图片链接和多余空白"""
    # 移除 markdown 图片链接
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # 移除空的 markdown 链接
    text = re.sub(r'\[([^\]]+)\]\(\)', r'\1', text)
    # 移除多余空行
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    # 去除首尾空白
    text = text.strip()
    return text


async def main():
    """主函数"""
    collection_name = "test_markdown"

    async with RAGSystem(collection_name=collection_name) as rag:
        # 清空已有数据（可选）
        await rag.clear_collection()

        # 读取 markdown 文件
        markdown_files = list(Path("data/sample_md").glob("*.md"))

        if not markdown_files:
            print("未找到 markdown 文件，请在 data/sample_md/ 目录下放入测试文件")
            return

        # 添加文档到索引
        all_texts = []
        all_metadata = []

        for md_file in markdown_files:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                # 按标题分割文本
                sections = split_by_headings(content, md_file.name)

                for i, (heading, text) in enumerate(sections):
                    cleaned = clean_text(text)
                    # 跳过空内容或太短的内容
                    if len(cleaned) < 10:
                        continue
                    all_texts.append(cleaned)
                    all_metadata.append({
                        "source": md_file.name,
                        "section": heading[:50],  # 限制 heading 长度
                        "section_index": i,
                    })

        print(f"处理后共有 {len(all_texts)} 个文本块")

        # 添加到向量库
        doc_ids = await rag.add_text_documents(all_texts, all_metadata)
        print(f"已添加 {len(doc_ids)} 个文档块")

        # 获取统计信息
        stats = await rag.get_stats()
        print(f"集合统计: {stats}")

        # 测试检索
        test_queries = [
            "珐琅是什么？",
            "珐琅釉料的成分是什么？",
            "什么是透明釉料？",
        ]

        for query in test_queries:
            results = await rag.retrieve(query, top_k=3)
            print(f"\n查询: {query}")
            for i, doc in enumerate(results, 1):
                print(f"  {i}. [{doc.score:.4f}] {doc.metadata.get('section', 'N/A')}")
                print(f"     {doc.text[:100]}...")


def split_by_headings(text: str, filename: str) -> list[tuple[str, str]]:
    """按标题分割 markdown 文本"""
    lines = text.split("\n")
    sections = []
    current_heading = filename
    current_content = []

    for line in lines:
        if line.startswith("#"):
            # 保存当前段落
            if current_content:
                sections.append((current_heading, "\n".join(current_content)))
            # 开始新段落
            current_heading = line.strip()
            current_content = []
        else:
            current_content.append(line)

    # 添加最后一个段落
    if current_content:
        sections.append((current_heading, "\n".join(current_content)))

    return sections


if __name__ == "__main__":
    asyncio.run(main())
