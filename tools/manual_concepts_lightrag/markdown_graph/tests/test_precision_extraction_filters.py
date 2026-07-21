from __future__ import annotations

import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
mod = SourceFileLoader(
    "precision_import_chunks", str(ROOT / "06_import_custom_chunks_to_lightrag.py")
).load_module()
PROFILE_PATH = ROOT / "prompts/entity_type/entity_type_prompt.new.yml"


class PrecisionExtractionFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = mod.load_entity_type_profile(PROFILE_PATH)

    def test_profile_loads_semantic_filter_lists(self):
        self.assertEqual(len(self.profile["allowed_entity_types"]), 12)
        self.assertIn("标准工具栏", self.profile["excluded_entity_names"])
        self.assertIn("工具按钮", self.profile["excluded_entity_suffixes"])
        self.assertIn("曲线", self.profile["entity_name_whitelist"])

    def test_semantic_entity_filter_keeps_core_terms(self):
        rejected_names = [
            "标准工具栏",
            "群组下三角按钮",
            "锁定拓展工具面板",
            "Enter键",
            "Page Up/Page Down键",
            "预览",
            "复制选项",
            "硬性选项",
            "物件",
            "操作步骤",
            "文件 | 导出选取的物件",
        ]
        accepted_names = [
            "Rhino",
            "NURBS技术",
            "NURBS曲线",
            "NURBS曲面",
            "控制点曲线",
            "单轨扫掠",
            "双轨扫掠",
            "沿着曲线流动",
            "曲线连续性",
            "位置连续（G0）",
            "正切连续（G1）",
            "曲率连续（G2）",
            "烧制温度",
            "焊接缺陷",
        ]
        nodes = {
            name: [{"entity_type": "Tool", "description": "教材中的专业实体。"}]
            for name in rejected_names
        }
        nodes.update(
            {
                name: [{"entity_type": "Concept", "description": "教材中的专业实体。"}]
                for name in accepted_names
            }
        )
        result, stats, rejected = mod.filter_extraction_results([(nodes, {})], self.profile)
        self.assertEqual(stats["accepted_entity_mentions"], len(accepted_names))
        self.assertEqual(stats["rejected_entity_mentions"], len(rejected_names))
        self.assertEqual(set(result[0][0]), set(accepted_names))
        self.assertIn("suspicious_entity_name", {item["reason"] for item in rejected})

    def test_relations_are_normalized_or_rejected_with_audit_reason(self):
        def entity(entity_type: str) -> list[dict]:
            return [{"entity_type": entity_type, "description": "教材明确提及。"}]

        nodes = {
            "掐丝": entity("Process"),
            "金属丝": entity("Material"),
            "掐丝珐琅工艺": entity("Technique"),
            "标准工具栏": entity("Tool"),
            "组合": entity("Technique"),
            "Enter键": entity("Tool"),
            "修剪": entity("Technique"),
            "曲线": entity("Concept"),
            "甲概念": entity("Concept"),
            "乙技法": entity("Technique"),
        }
        edges = {
            ("金属丝", "掐丝"): [{"src_id": "金属丝", "tgt_id": "掐丝", "keywords": "使用", "description": "掐丝使用金属丝。"}],
            ("掐丝", "掐丝珐琅工艺"): [{"src_id": "掐丝", "tgt_id": "掐丝珐琅工艺", "keywords": "包括", "description": "工艺包括掐丝。"}],
            ("标准工具栏", "组合"): [{"src_id": "标准工具栏", "tgt_id": "组合", "keywords": "用于", "description": "菜单关系。"}],
            ("Enter键", "修剪"): [{"src_id": "Enter键", "tgt_id": "修剪", "keywords": "使用", "description": "按键关系。"}],
            ("曲线", "甲概念"): [{"src_id": "曲线", "tgt_id": "甲概念", "keywords": "", "description": "没有关键词。"}],
            ("甲概念", "乙技法"): [{"src_id": "甲概念", "tgt_id": "乙技法", "keywords": "包括", "description": "类型不足以确定方向。"}],
        }
        result, stats, rejected = mod.filter_extraction_results([(nodes, edges)], self.profile)
        accepted_edges = result[0][1]
        self.assertIn(("掐丝", "金属丝"), accepted_edges)
        self.assertIn(("掐丝珐琅工艺", "掐丝"), accepted_edges)
        self.assertEqual(stats["normalized_relation_direction_count"], 2)
        reasons = [item["reason"] for item in rejected]
        self.assertEqual(reasons.count("invalid_relation_endpoint"), 2)
        self.assertIn("invalid_relation_keyword", reasons)
        self.assertIn("ambiguous_relation_direction", reasons)


if __name__ == "__main__":
    unittest.main()
