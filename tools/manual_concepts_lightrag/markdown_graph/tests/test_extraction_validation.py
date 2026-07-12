from __future__ import annotations
import sys, unittest
from pathlib import Path
from importlib.machinery import SourceFileLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
mod=SourceFileLoader('import_chunks',str(ROOT/'06_import_custom_chunks_to_lightrag.py')).load_module()
PROFILE={'allowed_entity_types':['Technique'],'allowed_relation_keywords':['使用']}
class ExtractionValidationTests(unittest.TestCase):
 def test_rejects_invalid_types_other_and_relations(self):
  nodes={'拓稿':[{'entity_type':'Technique','description':'正文出现的技法'}],'图1-2':[{'entity_type':'Technique','description':'图号'}],'x':[{'entity_type':'Other','description':'bad'}]}
  edges={('拓稿','图1-2'):[{'src_id':'拓稿','tgt_id':'图1-2','keywords':'使用','description':'bad endpoint'}],('拓稿','拓稿'):[{'src_id':'拓稿','tgt_id':'拓稿','keywords':'包括','description':'bad keyword'}]}
  result,stats,rejected=mod.filter_extraction_results([(nodes,edges)],PROFILE)
  self.assertEqual(stats['accepted_entity_mentions'],1); self.assertGreaterEqual(stats['rejected_entity_mentions'],2); self.assertEqual(stats['accepted_relation_mentions'],0); self.assertTrue(rejected)
if __name__=='__main__': unittest.main()
