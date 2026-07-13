from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from importlib.machinery import SourceFileLoader

export = SourceFileLoader(
    "export_graph_records", str(ROOT / "09_export_graph_records.py")
).load_module()


class FakeGraph:
    def __init__(self, nodes):
        self.nodes = nodes

    async def get_all_nodes(self):
        return self.nodes


class ExportGraphRecordsTests(unittest.TestCase):
    def test_all_node_names_mode_allows_missing_meta(self):
        export.validate_args(SimpleNamespace(all_node_names=True, meta=None))

    def test_book_mode_requires_meta(self):
        with self.assertRaisesRegex(ValueError, "--meta"):
            export.validate_args(SimpleNamespace(all_node_names=False, meta=None))

    def test_book_mode_parameters_remain_compatible(self):
        export.validate_args(
            SimpleNamespace(all_node_names=False, meta=Path("book_meta.json"), chunks=Path("chunks.jsonl"))
        )

    def test_normalizes_list_nodes_and_prefers_top_level_id(self):
        raw_nodes = [
            {"id": "珐琅", "entity_name": "不应采用", "data": {"entity_name": "也不采用"}},
            {"data": {"entity_name": "釉料"}},
            {"name": "金属底板"},
            {"id": "  "},
        ]
        self.assertEqual(
            export.collect_sorted_node_names(raw_nodes),
            (["珐琅", "釉料", "金属底板"], 4),
        )

    def test_normalizes_mapping_nodes_and_uses_mapping_key_as_id(self):
        raw_nodes = {
            "ENTITY_A": {"data": {"entity_name": "不应优先于映射 key"}},
            "ENTITY_B": {"id": "显式 ID"},
        }
        self.assertEqual(
            export.collect_sorted_node_names(raw_nodes),
            (["ENTITY_A", "显式 ID"], 2),
        )

    def test_empty_names_are_ignored_and_duplicates_are_deduplicated(self):
        raw_nodes = [
            {"entity_name": "  "},
            {"name": None},
            {"id": "重复"},
            {"entity_name": "重复"},
            {"data": {"entity_id": "唯一"}},
        ]
        self.assertEqual(export.collect_sorted_node_names(raw_nodes), (["唯一", "重复"], 5))

    def test_all_node_names_export_is_atomic_text_with_final_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "all_entity_node_names.txt"
            graph = FakeGraph([{"id": "beta"}, {"id": "Alpha"}, {"id": "beta"}])
            asyncio.run(export._export_all_node_names(graph, output, Path(directory) / "working"))
            self.assertEqual(output.read_text(encoding="utf-8"), "Alpha\nbeta\n")


if __name__ == "__main__":
    unittest.main()
