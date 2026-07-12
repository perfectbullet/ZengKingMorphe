#!/usr/bin/env python3
"""上传 PDF 到 MinerU；超过 200 页时自动拆分、串行解析并合并结果。"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://mineru.net"
DEFAULT_MODEL_VERSION = "vlm"
DEFAULT_UPLOAD_TIMEOUT = 30
DEFAULT_RESULT_TIMEOUT = 600
DEFAULT_POLL_INTERVAL = 5
MAX_MINERU_PAGES = 200
MAX_MINERU_FILE_BYTES = 200 * 1024 * 1024
DEFAULT_CHUNK_PAGES = 200


@dataclass(frozen=True)
class PdfChunk:
    """一个临时 PDF 分片及其在原文件中的 1 起始页码范围。"""

    index: int
    start_page: int
    end_page: int
    path: Path

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1


@dataclass(frozen=True)
class ProcessResult:
    """单个 MinerU 任务的本地结果，供大文件流程合并使用。"""

    input_pdf: Path
    batch_id: str
    full_zip_url: str
    zip_path: Path
    extracted_path: Path
    markdown_path: Path
    content_list_v2_path: Path
    images_dir: Path
    chunk: PdfChunk | None = None


class SinglePdfProcessError(RuntimeError):
    """保留已获得 batch_id 的单分片处理错误。"""

    def __init__(self, message: str, batch_id: str | None = None):
        super().__init__(message)
        self.batch_id = batch_id


class MinerUClient:
    """MinerU 精准解析 API 客户端。"""

    def __init__(self, token: str, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def upload_pdf(self, pdf_path: Path, model_version: str = DEFAULT_MODEL_VERSION) -> str:
        """申请上传链接并上传 PDF，返回 batch_id。"""
        url = f"{self.base_url}/api/v4/file-urls/batch"
        data = {
            "files": [{"name": pdf_path.name}],
            "model_version": model_version,
            "enable_formula": True,
            "enable_table": True,
        }
        logger.info("申请上传链接: %s", pdf_path.name)
        response = requests.post(url, headers=self.headers, json=data, timeout=DEFAULT_UPLOAD_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(f"申请上传链接失败: {result.get('msg')}")
        batch_id = result["data"]["batch_id"]
        upload_urls = result["data"].get("file_urls", [])
        if len(upload_urls) != 1:
            raise RuntimeError(f"MinerU 返回的上传链接数量异常: {len(upload_urls)}")
        logger.info("上传 PDF: %s", pdf_path.name)
        with pdf_path.open("rb") as pdf_file:
            upload_response = requests.put(upload_urls[0], data=pdf_file, timeout=120)
            upload_response.raise_for_status()
        logger.info("上传完成，batch_id=%s", batch_id)
        return batch_id

    def get_batch_result(self, batch_id: str) -> dict[str, Any]:
        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        response = requests.get(url, headers=self.headers, timeout=DEFAULT_UPLOAD_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def poll_result(self, batch_id: str, timeout: int, interval: int) -> dict[str, Any]:
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
                        raise RuntimeError(f"MinerU 解析失败: {file_result.get('err_msg') or '未知错误'}")
                progress.update(1)
                time.sleep(interval)
        finally:
            progress.close()


def _download_via_curl(zip_url: str, destination: Path) -> bool:
    """使用 curl 下载，以兼容部分 Python SSL 环境。"""
    result = subprocess.run(
        ["curl", "--noproxy", "*", "-fsSL", "-o", str(destination), zip_url],
        capture_output=True, timeout=120, check=False,
    )
    return result.returncode == 0


def download_zip(zip_url: str, zip_path: Path, max_retries: int = 3) -> None:
    """下载结果 ZIP，并在成功校验后原子地保存到目标位置。"""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception = RuntimeError("未知下载错误")
    for attempt in range(1, max_retries + 1):
        fd, temporary_name = tempfile.mkstemp(prefix=f".{zip_path.name}.", suffix=".tmp", dir=zip_path.parent)
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
    """安全解压 ZIP；仅在解压完整成功后替换目标目录。"""
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{extract_dir.name}.", dir=extract_dir.parent))
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


def rename_extracted_results(extract_dir: Path, document_name: str) -> tuple[Path, Path]:
    """重命名核心结果，并返回 Markdown 与 content_list_v2 的路径。"""
    markdown_path = extract_dir / "full.md"
    renamed_markdown_path = extract_dir / f"{document_name}_full.md"
    if not markdown_path.is_file():
        raise FileNotFoundError(f"MinerU 结果中未找到 full.md: {extract_dir}")
    if renamed_markdown_path.exists():
        raise FileExistsError(f"目标文件已存在: {renamed_markdown_path}")
    markdown_path.rename(renamed_markdown_path)

    target_list_path = extract_dir / f"{document_name}_content_list_v2.json"
    content_list_files = sorted(extract_dir.glob("*_content_list_v2.json"))
    non_target_files = [
        path for path in content_list_files if path.resolve() != target_list_path.resolve()
    ]
    if target_list_path.exists() and not non_target_files:
        return renamed_markdown_path, target_list_path
    if len(non_target_files) != 1:
        names = ", ".join(path.name for path in content_list_files) or "无"
        raise RuntimeError(f"预期一个 *_content_list_v2.json，实际为: {names}")
    source_list_path = non_target_files[0]
    if source_list_path.resolve() != target_list_path.resolve():
        if target_list_path.exists():
            raise FileExistsError(f"目标文件已存在: {target_list_path}")
        source_list_path.rename(target_list_path)
    return renamed_markdown_path, target_list_path


def get_pdf_page_count(pdf_path: Path) -> int:
    """读取 PDF 页数，并为损坏或无法解密的文件提供明确错误。"""
    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError("PDF 已加密且无法使用空密码读取")
        page_count = len(reader.pages)
    except PdfReadError as exc:
        raise ValueError(f"无法读取 PDF（文件可能损坏或加密）: {pdf_path}: {exc}") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"无法读取 PDF 页数: {pdf_path}: {exc}") from exc
    if page_count < 1:
        raise ValueError(f"PDF 没有可解析页面: {pdf_path}")
    return page_count


def get_pdf_file_size(pdf_path: Path) -> int:
    """返回 PDF 文件大小，单独封装以便测试分片大小决策。"""
    return pdf_path.stat().st_size


def is_within_mineru_limits(
    page_count: int,
    file_size_bytes: int,
    max_pages: int = MAX_MINERU_PAGES,
    max_file_bytes: int = MAX_MINERU_FILE_BYTES,
) -> bool:
    """判断一个待上传 PDF 是否同时满足 MinerU 页数和文件大小限制。"""
    return page_count <= max_pages and file_size_bytes <= max_file_bytes


def split_pdf(
    pdf_path: Path,
    temporary_dir: Path,
    chunk_pages: int,
    max_file_bytes: int = MAX_MINERU_FILE_BYTES,
) -> list[PdfChunk]:
    """物理拆分 PDF，并对超大分片递归二分直到符合 MinerU 限制。"""
    total_pages = get_pdf_page_count(pdf_path)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError("PDF 已加密且无法使用空密码读取")

        def write_range(start_index: int, end_index: int) -> Path:
            range_path = temporary_dir / f".range_p{start_index + 1:04d}-p{end_index:04d}.pdf"
            writer = PdfWriter()
            for page_index in range(start_index, end_index):
                writer.add_page(reader.pages[page_index])
            with range_path.open("wb") as file_handle:
                writer.write(file_handle)
            return range_path

        def split_range(start_index: int, end_index: int) -> list[tuple[int, int, Path]]:
            range_path = write_range(start_index, end_index)
            page_count = end_index - start_index
            file_size_bytes = get_pdf_file_size(range_path)
            if is_within_mineru_limits(page_count, file_size_bytes, chunk_pages, max_file_bytes):
                return [(start_index, end_index, range_path)]
            if page_count == 1:
                range_path.unlink(missing_ok=True)
                raise ValueError(
                    f"原始第 {start_index + 1} 页单页 PDF 仍超过 MinerU 文件大小限制："
                    f"{file_size_bytes} bytes（限制 {max_file_bytes} bytes）"
                )
            range_path.unlink(missing_ok=True)
            middle_index = start_index + page_count // 2
            return split_range(start_index, middle_index) + split_range(middle_index, end_index)

        raw_chunks: list[tuple[int, int, Path]] = []
        for start_index in range(0, total_pages, chunk_pages):
            raw_chunks.extend(split_range(start_index, min(start_index + chunk_pages, total_pages)))

        chunks: list[PdfChunk] = []
        for index, (start_index, end_index, range_path) in enumerate(sorted(raw_chunks), start=1):
            filename = f"{pdf_path.stem}_part_{index:04d}_p{start_index + 1:04d}-p{end_index:04d}.pdf"
            chunk_path = temporary_dir / filename
            range_path.replace(chunk_path)
            chunks.append(PdfChunk(index, start_index + 1, end_index, chunk_path))
        return chunks
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"拆分 PDF 失败: {pdf_path}: {exc}") from exc


def process_single_pdf(
    client: MinerUClient,
    pdf_path: Path,
    output_dir: Path,
    model_version: str,
    result_timeout: int,
    poll_interval: int,
    chunk: PdfChunk | None = None,
) -> ProcessResult:
    """处理一个实际上传任务，供普通和分片模式复用。"""
    batch_id: str | None = None
    try:
        batch_id = client.upload_pdf(pdf_path, model_version=model_version)
        result = client.poll_result(batch_id, timeout=result_timeout, interval=poll_interval)
        zip_url = result.get("full_zip_url")
        if not zip_url:
            raise RuntimeError("MinerU 解析完成，但结果中没有 full_zip_url")
        zip_path = output_dir / f"{pdf_path.name}.zip"
        extract_dir = output_dir / pdf_path.stem
        download_zip(zip_url, zip_path)
        extract_zip(zip_path, extract_dir)
        markdown_path, content_list_path = rename_extracted_results(extract_dir, pdf_path.stem)
        return ProcessResult(pdf_path, batch_id, zip_url, zip_path, extract_dir, markdown_path,
                             content_list_path, extract_dir / "images", chunk)
    except Exception as exc:
        if isinstance(exc, SinglePdfProcessError):
            raise
        raise SinglePdfProcessError(str(exc), batch_id=batch_id) from exc


_IMAGE_REFERENCE = re.compile(r"(?<![A-Za-z0-9_/-])(?:\./)?images/")
_IMAGE_PATH = re.compile(r"(?<![A-Za-z0-9_/-])(?:\./)?(images/[^\s\]\)\}\"'<>]+)")


def _rewrite_image_references(value: Any, replacement_prefix: str) -> Any:
    if isinstance(value, str):
        return _IMAGE_REFERENCE.sub(replacement_prefix, value)
    if isinstance(value, list):
        return [_rewrite_image_references(item, replacement_prefix) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_image_references(item, replacement_prefix) for key, item in value.items()}
    return value


def _image_references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [match.group(1) for match in _IMAGE_PATH.finditer(value)]
    if isinstance(value, list):
        return [reference for item in value for reference in _image_references(item)]
    if isinstance(value, dict):
        return [reference for item in value.values() for reference in _image_references(item)]
    return []


def _validate_image_references(result_dir: Path, markdown: str, content_list: list[Any]) -> None:
    references = _image_references(markdown) + _image_references(content_list)
    # MinerU 的 image_source.path 可为 "images/" 这种目录元数据；它不是图片文件引用。
    file_references = [reference for reference in references if not reference.endswith("/")]
    missing = [
        reference
        for reference in file_references
        if not (result_dir / PurePosixPath(reference)).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"合并结果引用了不存在的图片: {', '.join(sorted(set(missing))[:5])}")


def _atomic_replace_directory(staging_dir: Path, final_dir: Path) -> None:
    """用已验证的暂存目录替换最终目录，失败时尽可能保留旧目录。"""
    backup_dir: Path | None = None
    if final_dir.exists():
        if not final_dir.is_dir():
            raise NotADirectoryError(f"最终结果路径已存在且不是目录: {final_dir}")
        backup_dir = final_dir.with_name(f".{final_dir.name}.previous-{int(time.time() * 1000)}")
        final_dir.replace(backup_dir)
    try:
        staging_dir.replace(final_dir)
    except Exception:
        if backup_dir and backup_dir.exists() and not final_dir.exists():
            backup_dir.replace(final_dir)
        raise
    if backup_dir:
        shutil.rmtree(backup_dir)


def merge_chunk_results(
    results: list[ProcessResult],
    output_dir: Path,
    source_pdf: Path,
    total_pages: int,
    chunk_pages: int,
    model_version: str,
) -> Path:
    """合并大 PDF 各分片的正文、content_list_v2 与图片，成功后原子发布。"""
    final_dir = output_dir / source_pdf.stem
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{source_pdf.stem}.merge-", dir=output_dir))
    try:
        images_dir = staging_dir / "images"
        merged_markdown: list[str] = []
        merged_content: list[Any] = []
        manifest_chunks: list[dict[str, Any]] = []
        for result in results:
            if result.chunk is None:
                raise ValueError("合并分片结果缺少页码范围")
            chunk = result.chunk
            part_dir_name = f"part_{chunk.index:04d}"
            replacement_prefix = f"images/{part_dir_name}/"
            if not result.markdown_path.is_file() or not result.content_list_v2_path.is_file():
                raise FileNotFoundError(f"分片 {chunk.index} 缺少可合并的核心结果")
            markdown = _rewrite_image_references(result.markdown_path.read_text(encoding="utf-8"), replacement_prefix)
            try:
                content = json.loads(result.content_list_v2_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"分片 {chunk.index} content_list_v2.json 不是合法 JSON: {exc}") from exc
            if not isinstance(content, list):
                raise ValueError(f"分片 {chunk.index} content_list_v2.json 顶层必须为 list")
            if len(content) != chunk.page_count:
                raise ValueError(
                    f"分片 {chunk.index} content_list_v2 页数异常: 期望 {chunk.page_count}，实际 {len(content)}"
                )
            merged_markdown.append(markdown)
            merged_content.extend(_rewrite_image_references(content, replacement_prefix))
            if result.images_dir.exists():
                if not result.images_dir.is_dir():
                    raise NotADirectoryError(f"分片 {chunk.index} images 不是目录: {result.images_dir}")
                shutil.copytree(result.images_dir, images_dir / part_dir_name)
            manifest_chunks.append({
                "index": chunk.index, "start_page": chunk.start_page, "end_page": chunk.end_page,
                "page_count": chunk.page_count, "filename": result.input_pdf.name,
                "batch_id": result.batch_id, "state": "done", "zip_path": str(result.zip_path),
                "extracted_path": str(result.extracted_path),
            })
        if len(merged_content) != total_pages:
            raise ValueError(f"合并 content_list_v2 页数异常: 期望 {total_pages}，实际 {len(merged_content)}")
        final_markdown = "\n\n".join(merged_markdown)
        _validate_image_references(staging_dir, final_markdown, merged_content)
        logger.info("开始合并 Markdown")
        (staging_dir / f"{source_pdf.stem}_full.md").write_text(final_markdown, encoding="utf-8")
        logger.info("开始合并 content_list_v2.json")
        (staging_dir / f"{source_pdf.stem}_content_list_v2.json").write_text(
            json.dumps(merged_content, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "source_file": str(source_pdf), "total_pages": total_pages, "chunk_pages": chunk_pages,
            "chunk_count": len(results), "model_version": model_version,
            "created_at": datetime.now(timezone.utc).isoformat(), "chunks": manifest_chunks,
        }
        (staging_dir / "merge_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _atomic_replace_directory(staging_dir, final_dir)
        logger.info("合并完成：%d页", total_pages)
        return final_dir
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def process_pdf(
    client: MinerUClient,
    pdf_path: Path,
    output_dir: Path,
    model_version: str,
    result_timeout: int,
    poll_interval: int,
    chunk_pages: int = DEFAULT_CHUNK_PAGES,
) -> ProcessResult | Path:
    """按 MinerU 页数和文件大小限制选择直接或分片串行解析。"""
    total_pages = get_pdf_page_count(pdf_path)
    file_size_bytes = get_pdf_file_size(pdf_path)
    if is_within_mineru_limits(total_pages, file_size_bytes):
        return process_single_pdf(client, pdf_path, output_dir, model_version, result_timeout, poll_interval)
    logger.info(
        "检测到PDF共%d页、大小%d bytes，超过MinerU单文件限制（%d页、%d bytes），将拆分处理",
        total_pages, file_size_bytes, MAX_MINERU_PAGES, MAX_MINERU_FILE_BYTES,
    )
    parts_dir = output_dir / f"{pdf_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{pdf_path.stem}-mineru-") as temporary_name:
        chunks = split_pdf(pdf_path, Path(temporary_name), chunk_pages)
        logger.info("分片生成完成，共%d个分片", len(chunks))
        results: list[ProcessResult] = []
        for chunk in chunks:
            logger.info("开始处理分片 %d/%d：原始页码%d-%d", chunk.index, len(chunks), chunk.start_page, chunk.end_page)
            try:
                result = process_single_pdf(client, chunk.path, parts_dir, model_version, result_timeout, poll_interval, chunk)
            except Exception as exc:
                batch_id = getattr(exc, "batch_id", None) or "未获得"
                raise RuntimeError(
                    f"分片 {chunk.index}/{len(chunks)}（原始页码{chunk.start_page}-{chunk.end_page}，batch_id={batch_id}）解析失败: {exc}"
                ) from exc
            results.append(result)
            logger.info("分片 %d/%d 解析完成", chunk.index, len(chunks))
    return merge_chunk_results(results, output_dir, pdf_path, total_pages, chunk_pages, model_version)


def load_local_env() -> None:
    """读取脚本同级 .env，但不覆盖已有环境变量。"""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 MinerU 解析单个 PDF；超过 200 页会自动分片")
    parser.add_argument("--input", required=True, type=Path, help="输入 PDF 文件")
    parser.add_argument("--output", required=True, type=Path, help="结果输出目录")
    parser.add_argument("--model_version", default=DEFAULT_MODEL_VERSION, choices=["pipeline", "vlm", "MinerU-HTML"])
    parser.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL, help="轮询间隔秒数")
    parser.add_argument("--result_timeout", type=int, default=DEFAULT_RESULT_TIMEOUT, help="解析超时秒数")
    parser.add_argument("--chunk_pages", type=int, default=DEFAULT_CHUNK_PAGES, help="自动分片页数（1-200，默认 200）")
    args = parser.parse_args()
    if not 1 <= args.chunk_pages <= MAX_MINERU_PAGES:
        parser.error(f"--chunk_pages 必须在 1 到 {MAX_MINERU_PAGES} 之间")
    if args.poll_interval <= 0:
        parser.error("--poll_interval 必须大于 0")
    if args.result_timeout <= 0:
        parser.error("--result_timeout 必须大于 0")
    return args


def validate_input(pdf_path: Path) -> Path:
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
    process_pdf(MinerUClient(token), pdf_path, output_dir, args.model_version, args.result_timeout,
                args.poll_interval, args.chunk_pages)
    logger.info("解析完成")


if __name__ == "__main__":
    main()
