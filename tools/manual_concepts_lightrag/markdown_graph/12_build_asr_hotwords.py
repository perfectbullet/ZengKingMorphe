#!/usr/bin/env python3
"""Build editable ASR hotwords JSONL from completed entity review JSONL."""
from __future__ import annotations
import argparse
from pathlib import Path
from book_meta import get_business_config, load_book_meta
from common import PROJECT_DIR, read_jsonl, write_jsonl
def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--meta',required=True,type=Path); p.add_argument('--review',type=Path); p.add_argument('--output',type=Path); return p.parse_args()
def run(a):
 m=load_book_meta(a.meta); b=get_business_config(m); s=m['artifact_stem']; rows=read_jsonl(a.review or PROJECT_DIR/'outputs/11_review'/f'{s}.entity_review.jsonl'); seen=set(); result=[]
 for row in rows:
  decision=row.get('human_decision'); term=(row.get('human_entity_name') if decision=='alias' else row.get('entity_name')) or ''
  if decision not in {'valid','alias'} or not str(term).strip() or term in seen: continue
  seen.add(term); risk=(row.get('qwen_review') or {}).get('asr_risk') or (row.get('deepseek_review') or {}).get('asr_risk') or 'low'
  result.append({'term':term,'weight':5,'enabled':True,'domain':b['domain'],'subject':b['subject'],'source_book':m['book_title'],'asr_risk':risk,'review_status':'pending','note':''})
 write_jsonl(a.output or PROJECT_DIR/'outputs/12_asr'/f'{s}.asr_hotwords.jsonl',result)
if __name__=='__main__': run(parse_args())
