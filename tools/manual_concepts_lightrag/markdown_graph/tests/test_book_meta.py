from __future__ import annotations
import hashlib, json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from book_meta import load_book_meta, resolve_book_paths, resolve_config_value, validate_book_meta
from common import read_jsonl, write_jsonl
from importlib.machinery import SourceFileLoader
init=SourceFileLoader('init_book_meta',str(ROOT/'00_init_book_meta.py')).load_module()

class BookMetaTests(unittest.TestCase):
 def setUp(self):
  self.old_dir=init.ENTITY_TYPE_DIR
  self.old_template=init.TEMPLATE
  self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
  init.ENTITY_TYPE_DIR=self.root/'prompts'/'entity_type'; init.TEMPLATE=ROOT/'prompts'/'entity_type_templates'/'industrial_training_entity_types.template.yml'
  self.md=self.root/'02珠宝手绘效果图表现_full.md'; self.md.write_text('# test',encoding='utf-8')
  self.artifact,self.book_id,self.book_title=init.derive_book_values(self.md)
  self.yaml=init.ENTITY_TYPE_DIR/f'{self.artifact}_entity_types.yml'; self.meta=self.root/f'{self.book_id}_meta.json'
 def tearDown(self):
  init.ENTITY_TYPE_DIR=self.old_dir; init.TEMPLATE=self.old_template; self.temp.cleanup()
 def args(self,**overrides):
  values={'markdown':self.md,'domain':'industrial_training','subject':'jewelry_hand_rendering','force_meta':False,'force_entity_types':False};values.update(overrides);return SimpleNamespace(**values)
 def write_valid_yaml(self,**overrides):
  self.yaml.parent.mkdir(parents=True,exist_ok=True)
  data={'book_id':self.book_id,'book_title':self.book_title,'domain':'industrial_training','subject':'jewelry_hand_rendering','allowed_entity_types':['Technique'],'allowed_relation_keywords':['使用'],'entity_types_guidance':'仅正文','entity_extraction_examples':['entity example'],'entity_extraction_json_examples':['{"entities":[]}']};data.update(overrides);import yaml;self.yaml.write_text(yaml.safe_dump(data,allow_unicode=True),encoding='utf-8')
 def test_full_name_and_content_list_order(self):
  self.assertEqual(init.derive_book_values(Path('02珠宝手绘效果图表现_full.md')),('02珠宝手绘效果图表现_full','02珠宝手绘效果图表现','珠宝手绘效果图表现'))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); (p/'02珠宝手绘效果图表现_list_v2.json').write_text('[]'); (p/'anything_content_list_v2.json').write_text('[]')
   self.assertEqual(init.find_content_list_v2(p,'02珠宝手绘效果图表现','02珠宝手绘效果图表现_full'),'02珠宝手绘效果图表现_list_v2.json')
 def test_relative_path_and_precedence(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); (p/'a.md').write_text('x'); data={'schema_version':'markdown_graph_book_meta.v1','book_id':'a','book_title':'a','artifact_stem':'a','inputs':{'markdown':'a.md','content_list_v2':''},'business':{'domain':'d','subject':'s'},'structure':{'front_lines':10,'toc_start_line':0,'toc_end_line':0,'body_start_line':0,'window_lines':1,'overlap_lines':1},'entity_extraction':{'prompt_file':''}}
   meta=p/'a_meta.json';meta.write_text(json.dumps(data)); loaded=load_book_meta(meta); self.assertEqual(resolve_book_paths(loaded)['markdown'],(p/'a.md').resolve()); self.assertEqual(resolve_config_value('cli','meta',None,'default'),'cli')
   with self.assertRaises(ValueError): validate_book_meta(loaded,stage='step2')
 def test_atomic_jsonl(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.jsonl'; write_jsonl(p,[{'x':1}]); self.assertEqual(read_jsonl(p),[{'x':1}])
 def test_existing_yaml_creates_meta_without_rewriting_profile(self):
  self.write_valid_yaml(); before=hashlib.sha256(self.yaml.read_bytes()).hexdigest()
  init.run(self.args())
  self.assertTrue(self.meta.is_file()); self.assertEqual(before,hashlib.sha256(self.yaml.read_bytes()).hexdigest())
  self.assertEqual(json.loads(self.meta.read_text())['entity_extraction']['prompt_file'],self.yaml.name)
 def test_repeat_is_idempotent(self):
  self.write_valid_yaml(); init.run(self.args()); before_meta=self.meta.read_bytes();before_yaml=self.yaml.read_bytes()
  init.run(self.args()); self.assertEqual(before_meta,self.meta.read_bytes());self.assertEqual(before_yaml,self.yaml.read_bytes())
 def test_yaml_book_id_or_subject_mismatch_fails_without_writes(self):
  for field,value in [('book_id','wrong'),('subject','wrong')]:
   with self.subTest(field=field):
    self.write_valid_yaml(**{field:value}); before=self.yaml.read_bytes()
    with self.assertRaisesRegex(ValueError,field): init.run(self.args())
    self.assertFalse(self.meta.exists());self.assertEqual(before,self.yaml.read_bytes())
    self.yaml.unlink()
 def test_force_meta_keeps_yaml_unchanged(self):
  self.write_valid_yaml();init.run(self.args()); before=self.yaml.read_bytes();init.run(self.args(force_meta=True));self.assertEqual(before,self.yaml.read_bytes())
 def test_force_yaml_keeps_other_meta_fields(self):
  self.write_valid_yaml();init.run(self.args()); data=json.loads(self.meta.read_text());data['custom']='keep';self.meta.write_text(json.dumps(data),encoding='utf-8');before={k:v for k,v in json.loads(self.meta.read_text()).items() if k!='entity_extraction'}
  init.run(self.args(force_entity_types=True));after=json.loads(self.meta.read_text());self.assertEqual(before,{k:v for k,v in after.items() if k!='entity_extraction'});self.assertEqual(after['entity_extraction']['prompt_file'],self.yaml.name)
if __name__=='__main__': unittest.main()
