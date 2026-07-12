#!/usr/bin/env python3
"""Combine exported entities and independent model reviews for human review."""
from __future__ import annotations
import argparse
from pathlib import Path
from book_meta import load_book_meta
from common import PROJECT_DIR, read_jsonl, write_jsonl
def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--meta',required=True,type=Path); p.add_argument('--entities',type=Path); p.add_argument('--qwen-review',type=Path); p.add_argument('--deepseek-review',type=Path); p.add_argument('--conflicts',type=Path); p.add_argument('--output',type=Path); return p.parse_args()
def run(a):
 m=load_book_meta(a.meta); s=m['artifact_stem']; root=PROJECT_DIR/'outputs'; entities=read_jsonl(a.entities or root/'09_export'/f'{s}.entities.jsonl'); q={x.get('entity_name'):x for x in read_jsonl(a.qwen_review or root/'10_model_review'/f'{s}.qwen_review.raw.jsonl')}; d={x.get('entity_name'):x for x in read_jsonl(a.deepseek_review or root/'10_model_review'/f'{s}.deepseek_review.raw.jsonl')}; c={x.get('entity_name'):x.get('conflict_flags',[]) for x in read_jsonl(a.conflicts or root/'10_model_review'/f'{s}.review_conflicts.jsonl')}
 rows=[{**e,'qwen_review':q.get(e.get('entity_name'),{}),'deepseek_review':d.get(e.get('entity_name'),{}),'conflict_flags':c.get(e.get('entity_name'),[]),'human_decision':'pending','human_entity_name':'','human_entity_type':'','human_note':'','reviewed_at':None} for e in entities]
 write_jsonl(a.output or root/'11_review'/f'{s}.entity_review.jsonl',rows)
if __name__=='__main__': run(parse_args())
