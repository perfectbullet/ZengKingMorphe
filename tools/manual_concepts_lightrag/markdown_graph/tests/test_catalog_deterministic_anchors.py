from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from importlib.machinery import SourceFileLoader

step3 = SourceFileLoader("catalog_structure_plan", str(ROOT / "03_catalog_structure_plan.py")).load_module()


def catalog(title: str, level: int = 1) -> dict:
    return {"title": title, "level": level, "aliases": []}


def candidate(line: int, text: str, kind: str = "markdown_heading") -> dict:
    return {
        "candidate_id": f"L{line}", "line_no": line, "line_span": [line, line],
        "text": text, "normalized_text": step3.normalized_title(text),
        "normalized_full_text": step3.normalized_full_text(text),
        "number_key": step3.extract_number_key(text), "candidate_type": kind,
        "structure_marker": step3.extract_structure_marker(text), "score": 0,
        "reasons": [], "context_before": [], "context_after": [],
    }


class DeterministicAnchorTests(unittest.TestCase):
    def test_chinese_structure_markers(self):
        self.assertEqual(step3.extract_structure_marker("第二章 工具与蜡材")["kind"], "chapter")
        self.assertEqual(step3.extract_structure_marker("第二章 工具与蜡材")["ordinal"], 2)
        self.assertEqual(step3.extract_structure_marker("第五节 翡翠吊坠")["kind"], "section")
        self.assertEqual(step3.extract_structure_marker("第二十一章 综合实训")["ordinal"], 21)
        self.assertEqual(step3.extract_structure_marker("1.5.2")["number_key"], "1.5.2")

    def test_exact_markdown_heading_locks_without_llm(self):
        item = catalog("第二章 工具与蜡材")
        locked, _ = step3.resolve_deterministic_anchors([item], [candidate(1099, "## 第二章 工具与蜡材")])
        self.assertTrue(locked[0]["matched"])
        self.assertEqual(locked[0]["matched_start_line"], 1099)
        self.assertEqual(locked[0]["confidence"], 1.0)
        self.assertEqual(locked[0]["match_method"], "deterministic_exact")
        with tempfile.TemporaryDirectory() as directory, patch.object(step3, "call_llm_json", AsyncMock()) as mocked:
            result = asyncio.run(step3.generate_anchors([item], [[candidate(1099, "## 第二章 工具与蜡材")]], locked, raw_dir=Path(directory), book_stem="book", body_start_line=1, line_count=1200, min_confidence=.65))
            self.assertTrue(result[0]["matched"])
            mocked.assert_not_awaited()

    def test_markdown_heading_beats_plain_mini_toc(self):
        item = catalog("2 手镯腕寸表", 2)
        locked, _ = step3.resolve_deterministic_anchors([item], [candidate(8744, "2 手镯腕寸表", "numbered_line"), candidate(8753, "## 2. 手镯腕寸表")])
        self.assertEqual(locked[0]["matched_start_line"], 8753)

    def test_plain_short_title_is_not_deterministically_locked(self):
        locked, _ = step3.resolve_deterministic_anchors([catalog("第五节 翡翠吊坠", 2)], [candidate(4506, "第五节 翡翠吊坠", "numbered_line")])
        self.assertIsNone(locked[0])

    def test_duplicate_same_strength_remains_ambiguous(self):
        locked, _ = step3.resolve_deterministic_anchors([catalog("质地鉴别", 2)], [candidate(20, "## 质地鉴别"), candidate(80, "## 质地鉴别")])
        self.assertIsNone(locked[0])

    def test_structure_conflict_is_not_locked(self):
        locked, _ = step3.resolve_deterministic_anchors([catalog("第二章 工具与蜡材")], [candidate(20, "## 第三章 工具与蜡材")])
        self.assertIsNone(locked[0])

    def test_candidate_window_uses_locked_boundaries(self):
        anchors = [
            step3.deterministic_anchor(1, catalog("第一章 A"), candidate(10, "## 第一章 A"), []),
            None,
            step3.deterministic_anchor(3, catalog("第三章 C"), candidate(30, "## 第三章 C"), []),
        ]
        filtered, window = step3.candidates_in_deterministic_window(2, [candidate(5, "## X"), candidate(20, "## B"), candidate(35, "## Y")], anchors)
        self.assertEqual(window, {"lower_bound": 10, "upper_bound": 30})
        self.assertEqual([item["line_no"] for item in filtered], [20])

    def test_order_keeps_deterministic_anchor_and_rejects_llm_conflict(self):
        anchors = [
            step3.deterministic_anchor(1, catalog("第一章 A"), candidate(100, "## 第一章 A"), []),
            step3.unmatched_anchor(2, catalog("第二章 B"), [], ""),
        ]
        anchors[1].update({"matched": True, "matched_candidate_id": "L90", "matched_start_line": 90, "matched_title_text": "## 第二章 B", "confidence": .9, "match_method": "llm"})
        step3.validate_anchor_order(anchors)
        self.assertTrue(anchors[0]["matched"])
        self.assertFalse(anchors[1]["matched"])
        self.assertIn("确定性锚点顺序冲突", anchors[1]["reason"])

    def test_catalog_remains_the_only_anchor_source(self):
        locked, _ = step3.resolve_deterministic_anchors([catalog("目录标题")], [candidate(10, "## 目录标题"), candidate(20, "## 正文额外标题")])
        self.assertEqual(len(locked), 1)

    def test_video_candidate_is_rejected_by_anchor_validation(self):
        item = catalog("第六节 豹头吊坠", 2)
        video = candidate(20, "视频5-6-1豹头吊坠制作", "short_title")
        result = step3.validate_anchor(
            {"catalog_index": 1, "matched": True, "matched_candidate_id": "L20", "matched_start_line": 20, "matched_title_text": video["text"], "confidence": .9, "reason": "wrong"},
            1, item, [video], body_start_line=1, line_count=100, min_confidence=.65,
        )
        self.assertFalse(result["matched"])
        self.assertIn("视频", result["reason"])


if __name__ == "__main__":
    unittest.main()
