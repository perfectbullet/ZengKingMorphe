#!/usr/bin/env python3
"""
知识库管理脚本

功能：
1. 创建知识库
2. 上传文档到知识库
3. 绑定知识库到数字员工

使用示例：
    # 创建知识库
    python scripts/kb_manager.py create-kb --name "产品手册" --description "产品使用说明"

    # 上传文档
    python scripts/kb_manager.py upload-doc --kb-id kb_abc123 --file ./document.pdf

    # 绑定到员工
    python scripts/kb_manager.py bind-employee --kb-id kb_abc123 --employee-id 29

    # 查看知识库列表
    python scripts/kb_manager.py list-kb

    # 查看知识库详情
    python scripts/kb_manager.py get-kb --kb-id kb_abc123

    # 查看文档列表
    python scripts/kb_manager.py list-docs --kb-id kb_abc123

    # 查看任务状态
    python scripts/kb_manager.py task-status --task-id task_abc123
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional

import requests
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn


class KBManager:
    """知识库管理客户端"""

    def __init__(self, base_url: str = "http://127.0.0.1:8100", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.console = Console()
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """发送HTTP请求"""
        url = f"{self.base_url}{endpoint}"
        headers = dict(self.headers)
        headers.update(kwargs.pop("headers", {}))

        # 文件上传时不使用JSON headers
        if "files" in kwargs:
            headers.pop("Content-Type", None)

        response = requests.request(method, url, headers=headers, timeout=300, **kwargs)
        response.raise_for_status()
        return response.json()

    def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        category: str = "默认",
        priority: str = "medium",
        tags: Optional[List[str]] = None
    ) -> dict:
        """创建知识库"""
        self.console.print(f"[cyan]创建知识库: {name}[/cyan]")

        payload = {
            "name": name,
            "description": description,
            "category": category,
            "priority": priority,
            "tags": tags or [],
            "config": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_model": "bge-large-zh-v1.5"
            }
        }

        result = self._request("POST", "/api/knowledge_base/create", json=payload)

        if result.get("code") == 200:
            kb_id = result.get("data", {}).get("kb_id")
            self.console.print(f"[green]知识库创建成功![/green] kb_id: {kb_id}")
            return result.get("data", {})
        else:
            self.console.print(f"[red]创建失败: {result}[/red]")
            return {}

    def list_knowledge_bases(self, category: Optional[str] = None) -> List[dict]:
        """列出知识库"""
        self.console.print("[cyan]获取知识库列表...[/cyan]")

        params = {}
        if category:
            params["category"] = category

        result = self._request("GET", "/api/knowledge_base/list", params=params)

        if result.get("code") == 200:
            data = result.get("data", {})
            items = data.get("items", [])
            total = data.get("total", 0)
            self.console.print(f"[green]共 {total} 个知识库[/green]")
            return items
        return []

    def get_knowledge_base(self, kb_id: str) -> dict:
        """获取知识库详情"""
        self.console.print(f"[cyan]获取知识库详情: {kb_id}[/cyan]")

        result = self._request("GET", f"/api/knowledge_base/{kb_id}")

        if result.get("code") == 200:
            return result.get("data", {})
        else:
            self.console.print(f"[red]获取失败: {result}[/red]")
            return {}

    def upload_document(
        self,
        kb_id: str,
        file_path: str,
        category: str = "默认"
    ) -> dict:
        """上传文档到知识库"""
        file_path = Path(file_path)
        if not file_path.exists():
            self.console.print(f"[red]文件不存在: {file_path}[/red]")
            return {}

        self.console.print(f"[cyan]上传文档: {file_path.name} -> {kb_id}[/cyan]")

        # 准备multipart表单数据
        with open(file_path, "rb") as f:
            files = {"files": (file_path.name, f, "application/octet-stream")}
            data = {"kb_id": kb_id, "category": category}

            url = f"{self.base_url}/api/knowledge_base/documents/upload"
            headers = dict(self.headers)
            headers.pop("Content-Type", None)  # 让requests自动设置
            if self.api_key:
                headers["X-API-Key"] = self.api_key

            response = requests.post(url, headers=headers, files=files, data=data, timeout=300)
            response.raise_for_status()
            result = response.json()

        if result.get("status") == "success":
            tasks = result.get("tasks", [])
            self.console.print(f"[green]文档已提交处理![/green] task_ids: {tasks}")
            return {"tasks": tasks}
        else:
            self.console.print(f"[red]上传失败: {result}[/red]")
            return {}

    def upload_documents_batch(
        self,
        kb_id: str,
        file_paths: List[str],
        category: str = "默认"
    ) -> List[str]:
        """批量上传文档"""
        self.console.print(f"[cyan]批量上传 {len(file_paths)} 个文件到 {kb_id}[/cyan]")

        all_tasks = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]上传中...", total=len(file_paths))

            for file_path in file_paths:
                result = self.upload_document(kb_id, file_path, category)
                tasks = result.get("tasks", [])
                all_tasks.extend(tasks)
                progress.update(task, advance=1)

        self.console.print(f"[green]批量上传完成! 共 {len(all_tasks)} 个任务[/green]")
        return all_tasks

    def list_documents(
        self,
        kb_id: str,
        category: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> List[dict]:
        """列出文档"""
        self.console.print(f"[cyan]获取知识库 {kb_id} 的文档列表[/cyan]")

        params = {"kb_id": kb_id}
        if category:
            params["category"] = category
        if status_filter:
            params["status"] = status_filter

        result = self._request("GET", "/api/knowledge_base/documents/list", params=params)

        if result.get("code") == 200:
            data = result.get("data", {})
            items = data.get("items", [])
            total = data.get("total", 0)
            self.console.print(f"[green]共 {total} 个文档[/green]")
            return items
        return []

    def get_task_status(self, task_id: str) -> dict:
        """获取任务状态"""
        self.console.print(f"[cyan]查询任务状态: {task_id}[/cyan]")

        result = self._request("GET", f"/api/knowledge_base/documents/tasks/{task_id}")

        if result.get("code") == 200:
            return result.get("data", {})
        else:
            self.console.print(f"[red]查询失败: {result}[/red]")
            return {}

    def bind_employee(
        self,
        employee_id: str,
        kb_ids: List[str],
        update_time: Optional[str] = None
    ) -> bool:
        """绑定知识库到数字员工"""
        from datetime import datetime

        if update_time is None:
            update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.console.print(f"[cyan]绑定知识库到员工 {employee_id}[/cyan]")
        self.console.print(f"  知识库IDs: {kb_ids}")

        payload = {
            "employee_id": employee_id,
            "update_time": update_time,
            "update_type": "knowledge",
            "knowledge": {
                "kb_ids": kb_ids,
                "faqs": [],
                "video_ids": []
            }
        }

        result = self._request("PUT", "/api/employee/setting/update", json=payload)

        if result.get("code") == 200:
            self.console.print(f"[green]绑定成功![/green]")
            return True
        else:
            self.console.print(f"[red]绑定失败: {result}")
            return False

    def get_employee_config(self, employee_id: str) -> dict:
        """获取员工配置"""
        self.console.print(f"[cyan]获取员工 {employee_id} 的配置[/cyan]")

        result = self._request("GET", f"/api/employee/detail/{employee_id}")

        if result.get("code") == 200:
            return result.get("data", {})
        else:
            self.console.print(f"[red]获取失败: {result}[/red]")
            return {}

    def display_knowledge_bases(self, kbs: List[dict]):
        """显示知识库列表表格"""
        if not kbs:
            self.console.print("[yellow]没有知识库[/yellow]")
            return

        table = Table(title="知识库列表")
        table.add_column("KB ID", style="cyan", no_wrap=False)
        table.add_column("名称", style="green")
        table.add_column("分类", style="blue")
        table.add_column("状态", style="yellow")
        table.add_column("创建时间", style="dim")

        for kb in kbs:
            table.add_row(
                kb.get("kb_id", ""),
                kb.get("name", ""),
                kb.get("category", ""),
                kb.get("status", ""),
                kb.get("created_at", "")[:19] if kb.get("created_at") else ""
            )

        self.console.print(table)

    def display_documents(self, docs: List[dict]):
        """显示文档列表表格"""
        if not docs:
            self.console.print("[yellow]没有文档[/yellow]")
            return

        table = Table(title="文档列表")
        table.add_column("Doc ID", style="cyan", no_wrap=False)
        table.add_column("文件名", style="green")
        table.add_column("分类", style="blue")
        table.add_column("状态", style="yellow")
        table.add_column("分块数", style="magenta")
        table.add_column("上传时间", style="dim")

        for doc in docs:
            table.add_row(
                doc.get("doc_id", ""),
                doc.get("filename", ""),
                doc.get("category", ""),
                doc.get("status", ""),
                str(doc.get("chunks_count", 0)),
                doc.get("uploaded_at", "")[:19] if doc.get("uploaded_at") else ""
            )

        self.console.print(table)

    def display_task_status(self, task: dict):
        """显示任务状态"""
        self.console.print(f"\n[bold]任务详情[/bold]")
        self.console.print(f"  任务ID: {task.get('task_id', '')}")
        self.console.print(f"  知识库ID: {task.get('kb_id', '')}")
        self.console.print(f"  文件名: {task.get('filename', '')}")
        self.console.print(f"  状态: [{self._get_status_color(task.get('status'))}]{task.get('status')}[/{self._get_status_color(task.get('status'))}]")
        self.console.print(f"  进度: {task.get('progress', 0):.1f}%")
        self.console.print(f"  已处理: {task.get('processed_chunks', 0)}/{task.get('total_chunks', 0)}")
        if task.get("error_message"):
            self.console.print(f"  错误: [red]{task.get('error_message')}[/red]")

    def _get_status_color(self, status: str) -> str:
        """获取状态颜色"""
        status_colors = {
            "pending": "yellow",
            "running": "blue",
            "completed": "green",
            "failed": "red",
            "cancelled": "dim"
        }
        return status_colors.get(status, "white")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="知识库管理工具")
    parser.add_argument("--url", default="http://127.0.0.1:8100", help="API服务器地址")
    parser.add_argument("--api-key", default="", help="API密钥")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # create-kb 命令
    create_kb_parser = subparsers.add_parser("create-kb", help="创建知识库")
    create_kb_parser.add_argument("--name", required=True, help="知识库名称")
    create_kb_parser.add_argument("--description", default="", help="知识库描述")
    create_kb_parser.add_argument("--category", default="默认", help="分类")
    create_kb_parser.add_argument("--priority", default="medium", choices=["low", "medium", "high"], help="优先级")
    create_kb_parser.add_argument("--tags", nargs="*", default=[], help="标签列表")

    # list-kb 命令
    list_kb_parser = subparsers.add_parser("list-kb", help="列出知识库")
    list_kb_parser.add_argument("--category", help="按分类筛选")

    # get-kb 命令
    get_kb_parser = subparsers.add_parser("get-kb", help="获取知识库详情")
    get_kb_parser.add_argument("--kb-id", required=True, help="知识库ID")

    # upload-doc 命令
    upload_parser = subparsers.add_parser("upload-doc", help="上传文档")
    upload_parser.add_argument("--kb-id", required=True, help="知识库ID")
    upload_parser.add_argument("--file", required=True, help="文件路径")
    upload_parser.add_argument("--category", default="默认", help="文档分类")

    # upload-batch 命令
    upload_batch_parser = subparsers.add_parser("upload-batch", help="批量上传文档")
    upload_batch_parser.add_argument("--kb-id", required=True, help="知识库ID")
    upload_batch_parser.add_argument("--dir", required=True, help="文件夹路径")
    upload_batch_parser.add_argument("--pattern", default="*.pdf", help="文件匹配模式")
    upload_batch_parser.add_argument("--category", default="默认", help="文档分类")

    # list-docs 命令
    list_docs_parser = subparsers.add_parser("list-docs", help="列出文档")
    list_docs_parser.add_argument("--kb-id", required=True, help="知识库ID")
    list_docs_parser.add_argument("--category", help="按分类筛选")
    list_docs_parser.add_argument("--status", help="按状态筛选")

    # task-status 命令
    task_parser = subparsers.add_parser("task-status", help="查询任务状态")
    task_parser.add_argument("--task-id", required=True, help="任务ID")

    # bind-employee 命令
    bind_parser = subparsers.add_parser("bind-employee", help="绑定知识库到员工")
    bind_parser.add_argument("--employee-id", required=True, help="员工ID")
    bind_parser.add_argument("--kb-id", action="append", required=True, help="知识库ID (可多个)")
    bind_parser.add_argument("--update-time", help="更新时间")

    # get-employee 命令
    get_emp_parser = subparsers.add_parser("get-employee", help="获取员工配置")
    get_emp_parser.add_argument("--employee-id", required=True, help="员工ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = KBManager(base_url=args.url, api_key=args.api_key)

    # 执行对应的命令
    if args.command == "create-kb":
        manager.create_knowledge_base(
            name=args.name,
            description=args.description,
            category=args.category,
            priority=args.priority,
            tags=args.tags
        )

    elif args.command == "list-kb":
        kbs = manager.list_knowledge_bases(category=args.category)
        manager.display_knowledge_bases(kbs)

    elif args.command == "get-kb":
        kb = manager.get_knowledge_base(args.kb_id)
        if kb:
            manager.console.print(json.dumps(kb, ensure_ascii=False, indent=2))

    elif args.command == "upload-doc":
        manager.upload_document(args.kb_id, args.file, args.category)

    elif args.command == "upload-batch":
        dir_path = Path(args.dir)
        files = list(dir_path.glob(args.pattern))
        file_paths = [str(f) for f in files if f.is_file()]
        if not file_paths:
            manager.console.print(f"[yellow]在 {dir_path} 中没有找到匹配的文件[/yellow]")
        else:
            manager.console.print(f"[cyan]找到 {len(file_paths)} 个文件[/cyan]")
            manager.upload_documents_batch(args.kb_id, file_paths, args.category)

    elif args.command == "list-docs":
        docs = manager.list_documents(args.kb_id, args.category, args.status)
        manager.display_documents(docs)

    elif args.command == "task-status":
        task = manager.get_task_status(args.task_id)
        if task:
            manager.display_task_status(task)

    elif args.command == "bind-employee":
        manager.bind_employee(args.employee_id, args.kb_id, args.update_time)

    elif args.command == "get-employee":
        config = manager.get_employee_config(args.employee_id)
        if config:
            manager.console.print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
