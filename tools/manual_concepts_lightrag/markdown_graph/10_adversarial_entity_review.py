#!/usr/bin/env python3
"""Run independent Qwen/DeepSeek evidence-grounded entity reviews (or dry-run)."""
from __future__ import annotations
import argparse, asyncio, json, os
from pathlib import Path
from typing import Any
from book_meta import load_book_meta
from common import PROJECT_DIR, read_jsonl, write_json, write_jsonl

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--meta',required=True,type=Path); p.add_argument('--entities',type=Path); p.add_argument('--knowledge-points',type=Path); p.add_argument('--dry-run',action='store_true'); p.add_argument('--output-dir',type=Path); p.add_argument('--batch-size',type=int,default=20); p.add_argument('--retries',type=int,default=2); p.add_argument('--max-concurrency',type=int,default=4); p.add_argument('--resume',action='store_true'); return p.parse_args()
def focused_payload(entity:dict)->dict:
 return {k:entity.get(k) for k in ('entity_name','entity_type','description','evidence')}
async def review(model:str, base_url:str, api_key:str, model_name:str, entities:list[dict], dry:bool, retries:int, max_concurrency:int):
 out=[]
 sem=asyncio.Semaphore(max(1,max_concurrency))
 async def one(entity):
  async with sem:
   payload=focused_payload(entity)
   if dry: result={"is_valid_entity":None,"suggested_type":"","is_name_grounded_in_text":None,"possible_ocr_error":None,"suggested_canonical_name":"","asr_risk":"unknown","keep_as_hotword_candidate":None,"reason":"dry-run","confidence":0.0}
   else:
    from openai import AsyncOpenAI
    client=AsyncOpenAI(base_url=base_url,api_key=api_key); prompt="只根据给定教材证据审核实体。返回 JSON：is_valid_entity,suggested_type,is_name_grounded_in_text,possible_ocr_error,suggested_canonical_name,asr_risk,keep_as_hotword_candidate,reason,confidence。不得改写原实体。\n"+json.dumps(payload,ensure_ascii=False)
    last=None
    for _ in range(max(0,retries)+1):
     try:
      response=await client.chat.completions.create(model=model_name,messages=[{"role":"user","content":prompt}],temperature=0.0,response_format={"type":"json_object"}); result=json.loads(response.choices[0].message.content or '{}'); break
     except Exception as exc: last=exc
    else: result={"is_valid_entity":None,"reason":f"review_error: {last}","confidence":0.0}
   return {"entity_name":entity.get("entity_name",""),"model":model,"evidence_chunk_ids":[x.get("chunk_id") for x in entity.get("evidence") or []],**result}
 for entity in entities:
  out.append(await one(entity))
 return out
async def run(a):
 meta=load_book_meta(a.meta); stem=meta['artifact_stem']; out=(a.output_dir or PROJECT_DIR/'outputs/10_model_review').resolve(); entities=read_jsonl(a.entities or PROJECT_DIR/'outputs/09_export'/f'{stem}.entities.jsonl')
 q=await review('qwen',os.getenv('QWEN_REVIEW_BASE_URL',''),os.getenv('QWEN_REVIEW_API_KEY','EMPTY'),os.getenv('QWEN_REVIEW_MODEL','Qwen3-32B-AWQ'),entities,a.dry_run,a.retries,a.max_concurrency)
 d=await review('deepseek',os.getenv('DEEPSEEK_BASE_URL',''),os.getenv('DEEPSEEK_API_KEY',''),os.getenv('DEEPSEEK_REVIEW_MODEL','deepseek-v4-flash'),entities,a.dry_run,a.retries,a.max_concurrency)
 by_q={x['entity_name']:x for x in q}; conflicts=[]
 for item in d:
  other=by_q[item['entity_name']]; flags=[key for key in ('is_valid_entity','suggested_type','possible_ocr_error','suggested_canonical_name') if other.get(key)!=item.get(key)]
  if flags: conflicts.append({'entity_name':item['entity_name'],'conflict_flags':flags,'qwen_review':other,'deepseek_review':item})
 write_jsonl(out/f'{stem}.qwen_review.raw.jsonl',q); write_jsonl(out/f'{stem}.deepseek_review.raw.jsonl',d); write_jsonl(out/f'{stem}.review_conflicts.jsonl',conflicts); write_json(out/f'{stem}.review_report.json',{'entities':len(entities),'conflicts':len(conflicts),'dry_run':a.dry_run})
if __name__=='__main__': asyncio.run(run(parse_args()))
