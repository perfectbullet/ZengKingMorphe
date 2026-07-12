import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pypdf import PdfWriter


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "extract_mineru_sdk.py"
SPEC = importlib.util.spec_from_file_location("extract_mineru_sdk", SCRIPT_PATH)
mineru = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mineru
SPEC.loader.exec_module(mineru)


def make_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


class MinerUSplitTests(unittest.TestCase):
    def make_extracted_result_dir(self, root, document_name, content_filename):
        extract_dir = root / "extracted"
        extract_dir.mkdir()
        (extract_dir / "full.md").write_text("markdown", encoding="utf-8")
        content_path = extract_dir / content_filename
        content_path.write_text("[]", encoding="utf-8")
        return extract_dir, content_path

    def fake_processor(self, calls, fail_chunk=None):
        def processor(client, pdf_path, output_dir, model_version, result_timeout, poll_interval, chunk=None):
            calls.append(chunk.index if chunk else 0)
            if chunk and chunk.index == fail_chunk:
                raise mineru.SinglePdfProcessError("模拟分片失败", batch_id="batch-2")
            extract_dir = output_dir / pdf_path.stem
            images_dir = extract_dir / "images"
            images_dir.mkdir(parents=True)
            label = f"chunk-{chunk.index}" if chunk else "single"
            (images_dir / "same.jpg").write_text(label, encoding="utf-8")
            markdown_path = extract_dir / f"{pdf_path.stem}_full.md"
            markdown_path.write_text(f"{label}\n![](images/same.jpg)", encoding="utf-8")
            content_path = extract_dir / f"{pdf_path.stem}_content_list_v2.json"
            page_count = chunk.page_count if chunk else mineru.get_pdf_page_count(pdf_path)
            content_path.write_text(
                json.dumps([
                    {
                        "image": "images/same.jpg",
                        "image_source": {"path": "images/"},
                    }
                ] * page_count),
                encoding="utf-8",
            )
            return mineru.ProcessResult(
                pdf_path, f"batch-{chunk.index if chunk else 0}", "https://example.invalid/redacted",
                output_dir / f"{pdf_path.name}.zip", extract_dir, markdown_path, content_path, images_dir, chunk,
            )
        return processor

    def test_small_and_200_page_pdf_process_once_without_split(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            for pages in (2, 200):
                pdf_path = root / f"{pages}.pdf"
                make_pdf(pdf_path, pages)
                calls = []
                with patch.object(mineru, "process_single_pdf", self.fake_processor(calls)):
                    result = mineru.process_pdf(None, pdf_path, root / f"out-{pages}", "vlm", 1, 1)
                self.assertIsInstance(result, mineru.ProcessResult)
                self.assertEqual(calls, [0])

    def test_201_pages_are_serial_and_merged(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            pdf_path = root / "book.pdf"
            make_pdf(pdf_path, 201)
            calls = []
            with patch.object(mineru, "process_single_pdf", self.fake_processor(calls)):
                final_dir = mineru.process_pdf(None, pdf_path, root / "output", "vlm", 1, 1)
            self.assertEqual(calls, [1, 2])
            markdown = (final_dir / "book_full.md").read_text(encoding="utf-8")
            self.assertLess(markdown.index("chunk-1"), markdown.index("chunk-2"))
            content = json.loads((final_dir / "book_content_list_v2.json").read_text(encoding="utf-8"))
            self.assertEqual(len(content), 201)

    def test_401_pages_split_200_200_1(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            pdf_path = root / "book.pdf"
            make_pdf(pdf_path, 401)
            chunks = mineru.split_pdf(pdf_path, root / "chunks", 200)
            self.assertEqual([(chunk.start_page, chunk.end_page) for chunk in chunks], [(1, 200), (201, 400), (401, 401)])
            self.assertEqual([mineru.get_pdf_page_count(chunk.path) for chunk in chunks], [200, 200, 1])

    def test_same_named_images_are_isolated_and_references_rewritten(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            pdf_path = root / "book.pdf"
            make_pdf(pdf_path, 201)
            calls = []
            with patch.object(mineru, "process_single_pdf", self.fake_processor(calls)):
                final_dir = mineru.process_pdf(None, pdf_path, root / "output", "vlm", 1, 1)
            self.assertEqual((final_dir / "images/part_0001/same.jpg").read_text(), "chunk-1")
            self.assertEqual((final_dir / "images/part_0002/same.jpg").read_text(), "chunk-2")
            markdown = (final_dir / "book_full.md").read_text(encoding="utf-8")
            self.assertIn("images/part_0001/same.jpg", markdown)
            self.assertIn("images/part_0002/same.jpg", markdown)
            content = json.loads((final_dir / "book_content_list_v2.json").read_text(encoding="utf-8"))
            self.assertEqual(content[0]["image"], "images/part_0001/same.jpg")
            self.assertEqual(content[-1]["image"], "images/part_0002/same.jpg")
            self.assertEqual(content[0]["image_source"]["path"], "images/part_0001/")
            self.assertEqual(content[-1]["image_source"]["path"], "images/part_0002/")

    def test_second_chunk_failure_stops_before_third_and_no_final_directory(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            pdf_path = root / "book.pdf"
            make_pdf(pdf_path, 401)
            calls = []
            with patch.object(mineru, "process_single_pdf", self.fake_processor(calls, fail_chunk=2)):
                with self.assertRaisesRegex(RuntimeError, r"分片 2/3.*原始页码201-400.*batch-2.*模拟分片失败"):
                    mineru.process_pdf(None, pdf_path, root / "output", "vlm", 1, 1)
            self.assertEqual(calls, [1, 2])
            self.assertFalse((root / "output/book").exists())

    def test_invalid_content_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            chunk = mineru.PdfChunk(1, 1, 2, root / "part.pdf")
            extract_dir = root / "part"
            extract_dir.mkdir()
            markdown = extract_dir / "part_full.md"
            markdown.write_text("text", encoding="utf-8")
            content = extract_dir / "part_content_list_v2.json"
            content.write_text("{}", encoding="utf-8")
            result = mineru.ProcessResult(chunk.path, "batch", "", root / "part.pdf.zip", extract_dir, markdown, content, extract_dir / "images", chunk)
            with self.assertRaisesRegex(ValueError, "顶层必须为 list"):
                mineru.merge_chunk_results([result], root, root / "book.pdf", 2, 200, "vlm")

    def test_content_list_page_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            chunk = mineru.PdfChunk(1, 1, 2, root / "part.pdf")
            extract_dir = root / "part"
            extract_dir.mkdir()
            markdown = extract_dir / "part_full.md"
            markdown.write_text("text", encoding="utf-8")
            content = extract_dir / "part_content_list_v2.json"
            content.write_text("[{}]", encoding="utf-8")
            result = mineru.ProcessResult(chunk.path, "batch", "", root / "part.pdf.zip", extract_dir, markdown, content, extract_dir / "images", chunk)
            with self.assertRaisesRegex(ValueError, "页数异常"):
                mineru.merge_chunk_results([result], root, root / "book.pdf", 2, 200, "vlm")

    def test_invalid_chunk_pages_are_rejected(self):
        for invalid in ("0", "201"):
            with patch.object(sys, "argv", ["extract_mineru_sdk.py", "--input", "a.pdf", "--output", "out", "--chunk_pages", invalid]):
                with self.assertRaises(SystemExit) as error:
                    mineru.parse_args()
            self.assertEqual(error.exception.code, 2)

    def test_content_list_already_has_target_name_is_not_renamed(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            extract_dir, content_path = self.make_extracted_result_dir(
                root, "document", "document_content_list_v2.json"
            )
            markdown_path, returned_content_path = mineru.rename_extracted_results(extract_dir, "document")
            self.assertEqual(markdown_path, extract_dir / "document_full.md")
            self.assertEqual(returned_content_path, content_path)
            self.assertTrue(content_path.is_file())

    def test_content_list_with_different_source_name_is_renamed(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            extract_dir, source_path = self.make_extracted_result_dir(
                root, "document", "mineru_content_list_v2.json"
            )
            _, returned_content_path = mineru.rename_extracted_results(extract_dir, "document")
            self.assertEqual(returned_content_path, extract_dir / "document_content_list_v2.json")
            self.assertFalse(source_path.exists())
            self.assertTrue(returned_content_path.is_file())

    def test_content_list_target_conflict_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            extract_dir, source_path = self.make_extracted_result_dir(
                root, "document", "mineru_content_list_v2.json"
            )
            target_path = extract_dir / "document_content_list_v2.json"
            target_path.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                mineru.rename_extracted_results(extract_dir, "document")
            self.assertTrue(source_path.is_file())
            self.assertEqual(target_path.read_text(encoding="utf-8"), "existing")

    def test_chinese_filename_is_sent_without_data_id(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            for filename in ("珠宝手绘效果图表现.pdf", "珠宝手绘效果图表现_part_0001_p0001-p0200.pdf"):
                with self.subTest(filename=filename):
                    pdf_path = Path(temporary_name) / filename
                    pdf_path.write_bytes(b"not parsed by upload")
                    post_response = MagicMock()
                    post_response.json.return_value = {
                        "code": 0,
                        "data": {"batch_id": "batch-1", "file_urls": ["https://upload.example.invalid/file"]},
                    }
                    put_response = MagicMock()
                    with patch.object(mineru.requests, "post", return_value=post_response) as post, patch.object(
                        mineru.requests, "put", return_value=put_response
                    ) as put:
                        batch_id = mineru.MinerUClient("test-token").upload_pdf(pdf_path)
                    self.assertEqual(batch_id, "batch-1")
                    request_json = post.call_args.kwargs["json"]
                    self.assertEqual(request_json["files"], [{"name": filename}])
                    self.assertNotIn("data_id", request_json["files"][0])
                    put.assert_called_once()

    def test_small_page_count_but_oversized_file_enters_split_flow(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            pdf_path = root / "book.pdf"
            make_pdf(pdf_path, 2)
            calls = []
            def fake_size(path):
                return mineru.MAX_MINERU_FILE_BYTES + 1 if path == pdf_path else 1
            with patch.object(mineru, "get_pdf_file_size", side_effect=fake_size), patch.object(
                mineru, "process_single_pdf", self.fake_processor(calls)
            ):
                final_dir = mineru.process_pdf(None, pdf_path, root / "output", "vlm", 1, 1)
            self.assertEqual(calls, [1])
            self.assertTrue((final_dir / "book_full.md").is_file())

    def test_oversized_chunk_is_recursively_split_with_continuous_pages(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            pdf_path = root / "book.pdf"
            make_pdf(pdf_path, 4)
            def fake_size(path):
                return 101 if mineru.get_pdf_page_count(path) > 1 else 100
            with patch.object(mineru, "get_pdf_file_size", side_effect=fake_size):
                chunks = mineru.split_pdf(pdf_path, root / "chunks", 4, max_file_bytes=100)
            self.assertEqual([(chunk.index, chunk.start_page, chunk.end_page) for chunk in chunks], [
                (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4),
            ])
            self.assertEqual([chunk.path.name for chunk in chunks], [
                "book_part_0001_p0001-p0001.pdf", "book_part_0002_p0002-p0002.pdf",
                "book_part_0003_p0003-p0003.pdf", "book_part_0004_p0004-p0004.pdf",
            ])

    def test_oversized_single_page_fails_before_processing(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            pdf_path = root / "book.pdf"
            make_pdf(pdf_path, 1)
            with patch.object(mineru, "get_pdf_file_size", return_value=mineru.MAX_MINERU_FILE_BYTES + 1), patch.object(
                mineru, "process_single_pdf"
            ) as processor:
                with self.assertRaisesRegex(
                    ValueError,
                    rf"第 1 页.*{mineru.MAX_MINERU_FILE_BYTES + 1} bytes.*{mineru.MAX_MINERU_FILE_BYTES} bytes",
                ):
                    mineru.process_pdf(None, pdf_path, root / "output", "vlm", 1, 1)
            processor.assert_not_called()

    def test_nonpositive_poll_interval_and_result_timeout_are_rejected(self):
        for argument, invalid in (("--poll_interval", "0"), ("--poll_interval", "-1"),
                                  ("--result_timeout", "0"), ("--result_timeout", "-1")):
            with patch.object(sys, "argv", ["extract_mineru_sdk.py", "--input", "a.pdf", "--output", "out", argument, invalid]):
                with self.assertRaises(SystemExit) as error:
                    mineru.parse_args()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
