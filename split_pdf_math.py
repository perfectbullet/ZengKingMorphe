#!/usr/bin/env python3
import fitz
import os

pdf_path = "Digital-Human-Disciplinary-Dataset/高中数学相关资料内容/01高中数学必修第一册.pdf"
output_dir = "Digital-Human-Disciplinary-Dataset/math_file_part"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)
total_pages = len(doc)

print(f"PDF总页数: {total_pages}")

# 分割策略：连续的8页、16页、32页
chunk_sizes = [40]

for chunk_size in chunk_sizes:
    for i in range(0, total_pages, chunk_size):
        start_page = i
        end_page = min(i + chunk_size - 1, total_pages - 1)
        page_count = end_page - start_page + 1

        # 只有当剩余页数 >= chunk_size 的一半时才分割（避免最后一部分太小）
        if page_count < chunk_size / 2:
            continue

        filename = f"01高中数学必修第一册-{chunk_size}pages-part{i//chunk_size + 1}-page{start_page + 1}-{end_page + 1}.pdf"
        output_path = os.path.join(output_dir, filename)

        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
        new_doc.save(output_path)
        new_doc.close()

        print(f"已生成: {filename} (页码 {start_page + 1}-{end_page + 1}, 共{page_count}页)")

doc.close()
print("\n完成！")
