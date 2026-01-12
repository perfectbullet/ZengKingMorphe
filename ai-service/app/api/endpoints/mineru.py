"""
MinerU PDF解析管理接口。

提供 MinerU 解析任务的查询和管理功能。
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime

from app.core.database import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 集合名称
COLLECTION_JOBS = "mineru_jobs"


class JobListItem(BaseModel):
    """任务列表项"""
    job_id: str
    file_name: str
    status: str
    total_chunks: int
    processed_chunks: int
    failed_chunks: int
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobDetail(BaseModel):
    """任务详情"""
    job_id: str
    file_name: str
    file_path: str
    status: str
    total_chunks: int
    processed_chunks: int
    failed_chunks: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    final_result: Optional[dict] = None


class JobMarkdownContent(BaseModel):
    """任务 Markdown 内容"""
    job_id: str
    file_name: str
    status: str
    markdown_content: str
    total_pages: Optional[int] = None


@router.get("/jobs", response_model=List[JobListItem])
async def list_recent_jobs(
    limit: int = Query(10, ge=1, le=100, description="返回的记录数量")
):
    """获取最近的 MinerU 解析任务列表。"""
    try:
        db = await get_database()
        collection = db[COLLECTION_JOBS]

        cursor = collection.find().sort("created_at", -1).limit(limit)

        jobs = []
        async for job in cursor:
            jobs.append(JobListItem(
                job_id=job.get("_id", ""),
                file_name=job.get("file_name", ""),
                status=job.get("status", ""),
                total_chunks=job.get("total_chunks", 0),
                processed_chunks=job.get("processed_chunks", 0),
                failed_chunks=job.get("failed_chunks", 0),
                created_at=job.get("created_at"),
                completed_at=job.get("completed_at")
            ))

        logger.info("Retrieved recent jobs", count=len(jobs), limit=limit)
        return jobs

    except Exception as e:
        logger.error("Failed to list jobs", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询任务列表失败: {str(e)}")


@router.get("/jobs/{job_id}", response_model=JobDetail)
async def get_job_detail(job_id: str):
    """获取指定任务的详细信息。"""
    try:
        db = await get_database()
        collection = db[COLLECTION_JOBS]

        job = await collection.find_one({"_id": job_id})
        if not job:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")

        return JobDetail(
            job_id=job.get("_id", ""),
            file_name=job.get("file_name", ""),
            file_path=job.get("file_path", ""),
            status=job.get("status", ""),
            total_chunks=job.get("total_chunks", 0),
            processed_chunks=job.get("processed_chunks", 0),
            failed_chunks=job.get("failed_chunks", 0),
            created_at=job.get("created_at"),
            updated_at=job.get("updated_at"),
            completed_at=job.get("completed_at"),
            failed_at=job.get("failed_at"),
            error_message=job.get("error_message"),
            final_result=job.get("final_result")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get job detail", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询任务详情失败: {str(e)}")


@router.get("/jobs/{job_id}/markdown", response_model=JobMarkdownContent)
async def get_job_markdown(job_id: str):
    """获取指定任务的 Markdown 解析内容。"""
    try:
        db = await get_database()
        collection = db[COLLECTION_JOBS]

        job = await collection.find_one({"_id": job_id})
        if not job:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")

        if job.get("status") != "completed":
            raise HTTPException(status_code=400, detail=f"任务未完成，状态: {job.get('status')}")

        final_result = job.get("final_result", {})
        content_items = final_result.get("content", [])

        markdown_content = ""
        for item in content_items:
            if isinstance(item, dict):
                markdown_content += item.get("text", "") or item.get("content", "")
            else:
                markdown_content += str(item)

        total_pages = final_result.get("metadata", {}).get("total_pages")

        return JobMarkdownContent(
            job_id=job.get("_id", ""),
            file_name=job.get("file_name", ""),
            status=job.get("status", ""),
            markdown_content=markdown_content,
            total_pages=total_pages
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get job markdown", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取 Markdown 内容失败: {str(e)}")


@router.get("/view", response_class=HTMLResponse)
async def mineru_view_page():
    """MinerU 任务管理页面。"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MinerU 解析任务管理</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: #fff;
            padding: 20px 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 24px;
            color: #1a1a1a;
        }
        .header-actions button {
            padding: 10px 20px;
            background: #007bff;
            color: #fff;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }
        .header-actions button:hover {
            background: #0056b3;
        }
        .content-wrapper {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 20px;
        }
        .job-list {
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            height: calc(100vh - 160px);
            display: flex;
            flex-direction: column;
        }
        .job-list-header {
            padding: 15px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            font-weight: 600;
            color: #495057;
        }
        .job-list-items {
            overflow-y: auto;
            flex: 1;
        }
        .job-item {
            padding: 15px 20px;
            border-bottom: 1px solid #f0f0f0;
            cursor: pointer;
            transition: background 0.2s;
        }
        .job-item:hover {
            background: #f8f9fa;
        }
        .job-item.active {
            background: #e7f3ff;
            border-left: 3px solid #007bff;
        }
        .job-item-name {
            font-weight: 500;
            color: #1a1a1a;
            margin-bottom: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .job-item-meta {
            font-size: 12px;
            color: #6c757d;
            display: flex;
            justify-content: space-between;
        }
        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        .status-completed { background: #d4edda; color: #155724; }
        .status-processing { background: #fff3cd; color: #856404; }
        .status-failed { background: #f8d7da; color: #721c24; }
        .status-pending { background: #e2e3e5; color: #383d41; }
        .markdown-view {
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: calc(100vh - 160px);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .markdown-header {
            padding: 15px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .markdown-header h2 {
            font-size: 16px;
            color: #495057;
        }
        .markdown-content {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            line-height: 1.8;
        }
        .markdown-content h1 {
            font-size: 24px;
            margin-bottom: 15px;
            color: #1a1a1a;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        .markdown-content h2 {
            font-size: 20px;
            margin-top: 25px;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .markdown-content h3 {
            font-size: 18px;
            margin-top: 20px;
            margin-bottom: 8px;
            color: #34495e;
        }
        .markdown-content p {
            margin-bottom: 15px;
            color: #333;
        }
        .markdown-content ul, .markdown-content ol {
            margin-left: 25px;
            margin-bottom: 15px;
        }
        .markdown-content li {
            margin-bottom: 5px;
        }
        .markdown-content code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
        }
        .markdown-content pre {
            background: #282c34;
            color: #abb2bf;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            margin-bottom: 15px;
        }
        .markdown-content pre code {
            background: transparent;
            color: inherit;
            padding: 0;
        }
        .markdown-content table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }
        .markdown-content table th,
        .markdown-content table td {
            border: 1px solid #dee2e6;
            padding: 10px;
            text-align: left;
        }
        .markdown-content table th {
            background: #f8f9fa;
            font-weight: 600;
        }
        .markdown-content blockquote {
            border-left: 4px solid #007bff;
            padding-left: 15px;
            margin: 15px 0;
            color: #6c757d;
        }
        .markdown-content hr {
            border: none;
            border-top: 1px solid #e9ecef;
            margin: 20px 0;
        }
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #6c757d;
        }
        .empty-state svg {
            width: 64px;
            height: 64px;
            margin-bottom: 15px;
            opacity: 0.5;
        }
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #6c757d;
        }
        .loading::after {
            content: '';
            width: 30px;
            height: 30px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px 20px;
            border-radius: 6px;
            margin: 20px;
        }
        .info-bar {
            display: flex;
            gap: 20px;
            padding: 10px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            font-size: 13px;
            color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MinerU 解析任务管理</h1>
            <div class="header-actions">
                <button onclick="loadJobs()">刷新列表</button>
            </div>
        </div>
        <div class="content-wrapper">
            <div class="job-list">
                <div class="job-list-header">最近的任务</div>
                <div class="job-list-items" id="jobListItems">
                    <div class="loading"></div>
                </div>
            </div>
            <div class="markdown-view">
                <div class="markdown-header" id="markdownHeader" style="display: none;">
                    <h2 id="markdownTitle">选择一个任务查看详情</h2>
                </div>
                <div class="info-bar" id="infoBar" style="display: none;">
                    <span id="infoStatus">状态: -</span>
                    <span id="infoPages">页数: -</span>
                    <span id="infoTime">完成时间: -</span>
                </div>
                <div class="markdown-content" id="markdownContent">
                    <div class="empty-state">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <p>请从左侧选择一个任务查看 Markdown 内容</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        let currentJobId = null;

        document.addEventListener('DOMContentLoaded', () => {
            loadJobs();
        });

        async function loadJobs() {
            const listContainer = document.getElementById('jobListItems');
            listContainer.innerHTML = '<div class="loading"></div>';

            try {
                const response = await fetch('/api/mineru/jobs?limit=50');
                if (!response.ok) throw new Error('获取任务列表失败');

                const jobs = await response.json();

                if (jobs.length === 0) {
                    listContainer.innerHTML = '<div class="empty-state"><p>暂无解析任务</p></div>';
                    return;
                }

                listContainer.innerHTML = jobs.map(job => `
                    <div class="job-item ${currentJobId === job.job_id ? 'active' : ''}" onclick="selectJob('${job.job_id}', '${escapeHtml(job.file_name)}')">
                        <div class="job-item-name" title="${escapeHtml(job.file_name)}">${escapeHtml(job.file_name)}</div>
                        <div class="job-item-meta">
                            <span class="status-badge status-${job.status}">${getStatusText(job.status)}</span>
                            <span>${job.processed_chunks}/${job.total_chunks} chunks</span>
                        </div>
                    </div>
                `).join('');

            } catch (error) {
                listContainer.innerHTML = `<div class="error-message">${escapeHtml(error.message)}</div>`;
            }
        }

        async function selectJob(jobId, fileName) {
            currentJobId = jobId;

            document.querySelectorAll('.job-item').forEach(item => {
                item.classList.remove('active');
            });
            event.currentTarget.classList.add('active');

            document.getElementById('markdownHeader').style.display = 'flex';
            document.getElementById('markdownTitle').textContent = fileName;
            document.getElementById('infoBar').style.display = 'flex';
            document.getElementById('markdownContent').innerHTML = '<div class="loading"></div>';

            try {
                const response = await fetch(`/api/mineru/jobs/${jobId}/markdown`);
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || '获取内容失败');
                }

                const data = await response.json();

                document.getElementById('infoStatus').textContent = `状态: ${getStatusText(data.status)}`;
                document.getElementById('infoPages').textContent = `页数: ${data.total_pages || '-'}`;
                document.getElementById('infoTime').textContent = `完成时间: ${new Date().toLocaleString()}`;

                document.getElementById('markdownContent').innerHTML = renderMarkdown(data.markdown_content);

            } catch (error) {
                document.getElementById('markdownContent').innerHTML = `<div class="error-message">${escapeHtml(error.message)}</div>`;
                document.getElementById('infoStatus').textContent = `状态: 错误`;
            }
        }

        function renderMarkdown(text) {
            if (!text) return '<p>无内容</p>';

            let html = text
                .replace(/^### (.*$)/gim, '<h3>$1</h3>')
                .replace(/^## (.*$)/gim, '<h2>$1</h2>')
                .replace(/^# (.*$)/gim, '<h1>$1</h1>')
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
                .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/\\[(.*?)\\]\\((.*?)\\)/g, '<a href="$2" target="_blank">$1</a>')
                .replace(/\\n/g, '<br>');

            return html;
        }

        function getStatusText(status) {
            const statusMap = {
                'pending': '等待中',
                'processing': '处理中',
                'completed': '已完成',
                'failed': '失败'
            };
            return statusMap[status] || status;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
""")
