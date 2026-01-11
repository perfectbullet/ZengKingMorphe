测试文件夹说明
============

这个文件夹用于存放测试文件，可以通过以下 API 接口访问：

1. 列出所有测试文件
   GET /api/knowledge-base/test-files

2. 下载指定文件
   GET /api/knowledge-base/test-files/download/{filename}

使用方法：
1. 将测试文件（PDF、TXT、DOCX 等）放入此文件夹
2. 访问 http://localhost:8000/api/knowledge-base/test-files 查看所有可用文件
3. 复制文件的 full_download_url
4. 在 /api/knowledge-base/documents/create_with_segment 接口中使用该 URL

示例：
{
  "kb_id": "kb_xxx",
  "document_name": "test_document.pdf",
  "resource_url": "http://localhost:8000/api/knowledge-base/test-files/download/test_document.pdf",
  "resource_id": "test_001"
}
