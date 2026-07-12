from __future__ import annotations

import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
outline = SourceFileLoader("outline_split_titles", str(ROOT / "02_detect_front_matter_and_toc.py")).load_module()
plan = SourceFileLoader("plan_split_titles", str(ROOT / "03_catalog_structure_plan.py")).load_module()


def item(title: str, *, level: int = 1, page_hint: str = "", normalized_title: str | None = None):
    return {"title": title, "level": level, "page_hint": page_hint, "normalized_title": normalized_title or title}


class SplitChapterTitleTests(unittest.TestCase):
    def merge(self, values, lines=None):
        return outline.merge_split_chapter_titles(values, lines or [])

    def test_chinese_and_english_chapter_pairs_merge(self):
        for chapter, title in (("第1章", "设计创意与构思"), ("第二章", "绘图工具"), ("CHAPTER 01", "Drawing Tools")):
            with self.subTest(chapter=chapter):
                merged, stats, warnings = self.merge([item(chapter), item(title)])
                self.assertEqual(stats["merged_chapter_title_count"], 1)
                self.assertFalse(warnings)
                self.assertEqual(len(merged), 1)
                self.assertEqual(merged[0]["title"], f"{chapter} {title}")
                self.assertEqual(merged[0]["aliases"], [chapter, title, f"{chapter} {title}"])

    def test_source_empty_line_still_merges(self):
        lines = [{"line_no": 10, "text": "第1章"}, {"line_no": 11, "text": ""}, {"line_no": 12, "text": "设计创意与构思"}]
        merged, stats, _ = self.merge([item("第1章"), item("设计创意与构思")], lines)
        self.assertEqual(stats["merged_chapter_title_count"], 1)
        self.assertEqual(merged[0]["title"], "第1章 设计创意与构思")

    def test_conservative_non_merge_cases(self):
        cases = [
            [item("第1章"), item("第2章")],
            [item("绘图工具"), item("铅笔", page_hint="025")],
            [item("第1章"), item("◆灵感捕捉与选题", page_hint="012")],
            [item("第1章"), item("珠宝首饰概述", page_hint="012")],
        ]
        for values in cases:
            with self.subTest(values=values):
                merged, stats, warnings = self.merge(values)
                self.assertEqual(stats["merged_chapter_title_count"], 0)
                self.assertEqual(len(merged), 2)
                self.assertTrue(warnings if outline.is_chapter_number_only(values[0]["title"]) else True)

    def test_validation_keeps_merged_metadata_and_only_warns_unresolved(self):
        merged, _, warnings = self.merge([item("第1章"), item("设计创意与构思")])
        normalized = outline.validate_and_normalize_catalog_items(merged, warnings=warnings)
        self.assertEqual(normalized[0]["chapter_no"], "第1章")
        self.assertEqual(normalized[0]["level"], 1)
        self.assertFalse(warnings)
        _, _, unresolved = self.merge([item("第1章"), item("珠宝首饰概述", page_hint="012")])
        self.assertEqual(len(unresolved), 1)

    def test_step3_builds_split_heading_candidate_at_chapter_line(self):
        prepared = {"lines": [*({"line_no": i, "text": ""} for i in range(1, 100)),
            {"line_no": 100, "text": "第1章"},
            {"line_no": 101, "text": ""},
            {"line_no": 102, "text": "## 设计创意与构思"},
            {"line_no": 103, "text": "CHAPTER 01"},
        ]}
        candidates = plan.build_global_candidates(prepared, 100)
        group = next(item for item in candidates if item["line_no"] == 100 and item["candidate_type"] == "chapter_group")
        self.assertEqual(group["line_span"], [100, 102])
        self.assertIn("设计创意与构思", group["text"])
        catalog = {"title": "第1章 设计创意与构思", "normalized_title": "设计创意与构思", "aliases": ["第1章", "设计创意与构思"]}
        scored = plan.score_candidates(catalog, 1, 1, candidates, 100, 103)
        self.assertEqual(scored[0]["line_no"], 100)
        self.assertEqual(scored[0]["candidate_type"], "chapter_group")

    def test_only_chapter_number_is_not_title_equivalent(self):
        catalog = {"title": "第1章 设计创意与构思", "normalized_title": "设计创意与构思"}
        candidates = [
            {"candidate_id": "L10", "line_no": 10, "text": "第1章", "normalized_text": "", "number_key": "1", "candidate_type": "numbered_line", "reasons": []},
            {"candidate_id": "L20", "line_no": 20, "text": "第1章 | 设计创意与构思", "normalized_text": "设计创意与构思", "number_key": "1", "candidate_type": "chapter_group", "reasons": []},
        ]
        scored = plan.score_candidates(catalog, 1, 2, candidates, 1, 100)
        self.assertEqual(scored[0]["candidate_id"], "L20")


if __name__ == "__main__":
    unittest.main()
