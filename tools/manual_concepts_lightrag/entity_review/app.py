#!/usr/bin/env python3
"""Standalone JSONL entity review application; no database is used."""
from __future__ import annotations
import argparse, shutil
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'markdown_graph'))
from common import read_jsonl, write_jsonl

DECISIONS={'pending','valid','invalid','alias','needs_check'}
class Update(BaseModel): human_decision:str; human_entity_name:str=''; human_entity_type:str=''; human_note:str=''
def backup_once(p:Path):
 b=Path(str(p)+'.bak')
 if not b.exists(): shutil.copy2(p,b)
def create_app(data:Path):
 app=FastAPI(title='实体人工审核')
 def load(): return read_jsonl(data)
 @app.get('/',response_class=HTMLResponse)
 def page(): return '''<!doctype html><meta charset="utf-8"><title>实体人工审核</title><style>body{font:14px system-ui;margin:1rem;display:grid;grid-template-columns:36% 64%;gap:1rem}input,select,textarea,button{margin:.2rem;padding:.35rem}#list{max-height:90vh;overflow:auto}li{cursor:pointer;padding:.4rem;border-bottom:1px solid #ddd}.active{background:#def}textarea{width:95%;height:8rem}pre{white-space:pre-wrap}</style><section><h2>实体列表</h2><input id=q placeholder="搜索"><input id=t placeholder="类型"><select id=d><option value="">全部状态</option><option>pending</option><option>valid</option><option>invalid</option><option>alias</option><option>needs_check</option></select><label><input id=c type=checkbox>仅冲突</label><button onclick="load()">筛选</button><ol id=list></ol></section><section id=detail><h2>详情</h2><p>选择一条记录；快捷键 j/k 上下切换，Ctrl+S 保存。</p></section><script>let rows=[],pos=0;async function load(){let u='/api/records?q='+encodeURIComponent(q.value)+'&entity_type='+encodeURIComponent(t.value)+'&decision='+encodeURIComponent(d.value)+'&conflict='+c.checked;rows=await (await fetch(u)).json();pos=0;renderList();show()}function renderList(){list.innerHTML=rows.map((r,i)=>`<li class="${i==pos?'active':''}" onclick="pos=${i};renderList();show()"><b>${r.entity_name}</b> [${r.entity_type}] ${r.conflict_flags?.join(',')||''}</li>`).join('')}function show(){let r=rows[pos];if(!r){detail.innerHTML='<h2>无记录</h2>';return}detail.innerHTML=`<h2>${r.entity_name}</h2><p>类型：${r.entity_type}</p><pre>${r.description||''}</pre><h3>正文证据</h3><pre>${JSON.stringify(r.evidence||[],null,2)}</pre><h3>模型审核</h3><pre>Qwen: ${JSON.stringify(r.qwen_review||{},null,2)}\nDeepSeek: ${JSON.stringify(r.deepseek_review||{},null,2)}</pre><label>人工结论<select id=hd>${['pending','valid','invalid','alias','needs_check'].map(x=>`<option ${x==r.human_decision?'selected':''}>${x}</option>`).join('')}</select></label><br><input id=hn placeholder="规范实体名" value="${r.human_entity_name||''}"><input id=ht placeholder="规范类型" value="${r.human_entity_type||''}"><br><textarea id=note placeholder="备注">${r.human_note||''}</textarea><br><button onclick="save()">保存</button>`}async function save(){let r=rows[pos];await fetch('/api/records/'+r.index,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({human_decision:hd.value,human_entity_name:hn.value,human_entity_type:ht.value,human_note:note.value})});await load()}document.onkeydown=e=>{if(e.key==='j'){pos=Math.min(rows.length-1,pos+1);renderList();show()}if(e.key==='k'){pos=Math.max(0,pos-1);renderList();show()}if(e.ctrlKey&&e.key==='s'){e.preventDefault();save()}};load()</script>'''
 @app.get('/api/records')
 def records(q:str='', entity_type:str='', conflict:bool=False, decision:str=''):
  rows=load(); result=[]
  for i,r in enumerate(rows):
   hay=' '.join(map(str,[r.get('entity_name',''),r.get('description','')]))
   if q and q.casefold() not in hay.casefold(): continue
   if entity_type and r.get('entity_type')!=entity_type: continue
   if conflict and not r.get('conflict_flags'): continue
   if decision and r.get('human_decision')!=decision: continue
   result.append({'index':i,**r})
  return result
 @app.put('/api/records/{index}')
 def update(index:int, value:Update):
  if value.human_decision not in DECISIONS: raise HTTPException(400,'无效人工结论')
  rows=load()
  if not 0<=index<len(rows): raise HTTPException(404,'记录不存在')
  backup_once(data); rows[index].update(value.model_dump()); rows[index]['reviewed_at']=datetime.now(timezone.utc).isoformat(); write_jsonl(data,rows); return {'ok':True,'record':rows[index]}
 return app
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=Path);p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8011);a=p.parse_args(); import uvicorn; uvicorn.run(create_app(a.data.resolve()),host=a.host,port=a.port)
if __name__=='__main__': main()
