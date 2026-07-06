#!/usr/bin/env python3
"""上传单个 PDF 到 MinerU，并保存、解压解析结果 ZIP。

示例:
    python extract_mineru_sdk.py \
        --input /path/to/document.pdf \
        --output /path/to/output

输出结构:
    output/
    ├── document.pdf.zip
    └── document/
        ├── full.md
        ├── images/
        └── ...
"""

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://mineru.net"
DEFAULT_MODEL_VERSION = "vlm"
DEFAULT_UPLOAD_TIMEOUT = 30
DEFAULT_RESULT_TIMEOUT = 600
DEFAULT_POLL_INTERVAL = 5


class MinerUClient:
    """MinerU 精准解析 API 客户端。"""

    def __init__(self, token: str, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def upload_pdf(
        self,
        pdf_path: Path,
        model_version: str = DEFAULT_MODEL_VERSION,
        enable_formula: bool = True,
        enable_table: bool = True,
    ) -> str:
        """申请上传链接并上传 PDF，返回 batch_id。"""
        url = f"{self.base_url}/api/v4/file-urls/batch"
        data = {
            "files": [{"name": pdf_path.name, "data_id": pdf_path.stem}],
            "model_version": model_version,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
        }

        logger.info("申请上传链接: %s", pdf_path.name)
        response = requests.post(
            url,
            headers=self.headers,
            json=data,
            timeout=DEFAULT_UPLOAD_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(f"申请上传链接失败: {result.get('msg')}")

        batch_id = result["data"]["batch_id"]
        upload_urls = result["data"].get("file_urls", [])
        if len(upload_urls) != 1:
            raise RuntimeError(f"MinerU 返回的上传链接数量异常: {len(upload_urls)}")

        logger.info("上传 PDF: %s", pdf_path)
        with pdf_path.open("rb") as pdf_file:
            upload_response = requests.put(upload_urls[0], data=pdf_file, timeout=120)
            upload_response.raise_for_status()

        logger.info("上传完成，batch_id=%s", batch_id)
        return batch_id

    def get_batch_result(self, batch_id: str) -> dict:
        """查询任务结果。"""
        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        response = requests.get(
            url,
            headers=self.headers,
            timeout=DEFAULT_UPLOAD_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def poll_result(
        self,
        batch_id: str,
        timeout: int = DEFAULT_RESULT_TIMEOUT,
        interval: int = DEFAULT_POLL_INTERVAL,
    ) -> dict:
        """轮询单个 PDF 的解析任务，返回该文件的结果。"""
        start = time.monotonic()
        progress = tqdm(desc=f"batch {batch_id[:8]}...", unit="poll")
        try:
            while True:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"batch {batch_id} 轮询超时 ({timeout}s)")

                result = self.get_batch_result(batch_id)
                if result.get("code") != 0:
                    raise RuntimeError(f"查询结果失败: {result.get('msg')}")

                extract_results = result.get("data", {}).get("extract_result", [])
                if extract_results:
                    file_result = extract_results[0]
                    state = file_result.get("state")
                    progress.set_postfix(state=state or "unknown")
                    if state == "done":
                        return file_result
                    if state == "failed":
                        raise RuntimeError(
                            f"MinerU 解析失败: {file_result.get('err_msg') or '未知错误'}"
                        )

                progress.update(1)
                time.sleep(interval)
        finally:
            progress.close()


def _download_via_curl(zip_url: str, destination: Path) -> bool:
    """使用 curl 下载，以兼容部分 Python SSL 环境。"""
    result = subprocess.run(
        ["curl", "--noproxy", "*", "-fsSL", "-o", str(destination), zip_url],
        capture_output=True,
        timeout=120,
        check=False,
    )
    return result.returncode == 0


def download_zip(zip_url: str, zip_path: Path, max_retries: int = 3) -> None:
    """下载结果 ZIP，并在成功校验后原子地保存到目标位置。"""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception = RuntimeError("未知下载错误")

    for attempt in range(1, max_retries + 1):
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{zip_path.name}.",
            suffix=".tmp",
            dir=zip_path.parent,
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            if not _download_via_curl(zip_url, temporary_path):
                response = requests.get(zip_url, timeout=120)
                response.raise_for_status()
                temporary_path.write_bytes(response.content)

            with zipfile.ZipFile(temporary_path) as archive:
                bad_file = archive.testzip()
                if bad_file is not None:
                    raise zipfile.BadZipFile(f"ZIP 内文件校验失败: {bad_file}")

            temporary_path.replace(zip_path)
            logger.info("ZIP 已保存: %s", zip_path)
            return
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                logger.warning("下载结果失败，第 %d/%d 次重试: %s", attempt, max_retries, exc)
                time.sleep(attempt * 3)
        finally:
            temporary_path.unlink(missing_ok=True)

    raise RuntimeError(f"下载 MinerU 结果失败: {last_error}") from last_error


def _validate_archive_member(member_name: str) -> None:
    """拒绝绝对路径和父目录跳转，避免 ZIP 路径穿越。"""
    normalized = member_name.replace("\\", "/")
    member_path = PurePosixPath(normalized)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"ZIP 包含不安全路径: {member_name}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """将 ZIP 完整解压到与输入 PDF 同名的目录。"""
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{extract_dir.name}.", dir=extract_dir.parent)
    )
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                _validate_archive_member(member.filename)
            archive.extractall(temporary_dir)

        if extract_dir.exists():
            if not extract_dir.is_dir():
                raise NotADirectoryError(f"解压目标已存在且不是目录: {extract_dir}")
            shutil.rmtree(extract_dir)
        temporary_dir.replace(extract_dir)
        logger.info("ZIP 已解压: %s", extract_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def load_local_env() -> None:
    """读取脚本同级 .env，但不覆盖已有环境变量。"""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 MinerU 解析单个 PDF")
    parser.add_argument("--input", required=True, type=Path, help="输入 PDF 文件")
    parser.add_argument("--output", required=True, type=Path, help="结果输出目录")
    parser.add_argument(
        "--model_version",
        default=DEFAULT_MODEL_VERSION,
        choices=["pipeline", "vlm", "MinerU-HTML"],
        help="MinerU 模型版本（默认: vlm）",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help="轮询间隔秒数",
    )
    parser.add_argument(
        "--result_timeout",
        type=int,
        default=DEFAULT_RESULT_TIMEOUT,
        help="解析超时秒数",
    )
    return parser.parse_args()


def validate_input(pdf_path: Path) -> Path:
    """验证并返回规范化的输入 PDF 路径。"""
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"输入 PDF 不存在: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"--input 仅支持 PDF 文件: {pdf_path}")
    return pdf_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pdf_path = validate_input(args.input)
    output_dir = args.output.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"--output 必须是目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    load_local_env()
    token = os.getenv("MINERU_API_TOKEN", "").strip()
    if not token:
        raise ValueError("未设置环境变量 MINERU_API_TOKEN")

    client = MinerUClient(token)
    batch_id = client.upload_pdf(pdf_path, model_version=args.model_version)
    result = client.poll_result(
        batch_id,
        timeout=args.result_timeout,
        interval=args.poll_interval,
    )
    zip_url = result.get("full_zip_url")
    if not zip_url:
        raise RuntimeError("MinerU 解析完成，但结果中没有 full_zip_url")

    zip_path = output_dir / f"{pdf_path.name}.zip"
    extract_dir = output_dir / pdf_path.stem
    download_zip(zip_url, zip_path)
    extract_zip(zip_path, extract_dir)

    logger.info("解析完成")
    logger.info("ZIP 文件: %s", zip_path)
    logger.info("解压目录: %s", extract_dir)


if __name__ == "__main__":
    main()
