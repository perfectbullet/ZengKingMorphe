#!/usr/bin/env python3
"""Standalone JSONL ASR hotword manager; no ASR adapter or database."""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'markdown_graph'))
from common import read_jsonl, write_jsonl
class Hotword(BaseModel): term:str; weight:int=5; enabled:bool=True; asr_risk:str='low'; review_status:str='pending'; note:str=''
def backup(p):
 b=Path(str(p)+'.bak')
 if not b.exists(): shutil.copy2(p,b)
def create_app(data):
 app=FastAPI(title='ASR 热词管理')
 def save(rows): backup(data); write_jsonl(data,rows)
 @app.get('/',response_class=HTMLResponse)
 def page(): return '''<!doctype html><meta charset="utf-8"><title>ASR 热词管理</title><style>body{font:14px system-ui;margin:1rem}input,select,button{margin:.25rem;padding:.35rem}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.35rem}</style><h1>ASR 热词管理</h1><input id=q placeholder="搜索"><select id=r><option value="">全部风险</option><option>low</option><option>medium</option><option>high</option></select><button onclick="load()">筛选</button><button onclick="add()">新增</button><a href="/api/export.txt">导出启用词</a><table><thead><tr><th>词</th><th>权重</th><th>启用</th><th>风险</th><th>备注</th><th>操作</th></tr></thead><tbody id=rows></tbody></table><script>async function load(){let x=await (await fetch('/api/hotwords?q='+encodeURIComponent(q.value)+'&risk='+r.value)).json();rows.innerHTML=x.map(v=>`<tr><td><input id="t${v.index}" value="${v.term}"></td><td><input id="w${v.index}" type=number min=1 max=10 value="${v.weight}"></td><td><input id="e${v.index}" type=checkbox ${v.enabled?'checked':''}></td><td><select id="r${v.index}">${['low','medium','high'].map(a=>`<option ${a==v.asr_risk?'selected':''}>${a}</option>`).join('')}</select></td><td><input id="n${v.index}" value="${v.note||''}"></td><td><button onclick="save(${v.index})">保存</button><button onclick="del(${v.index})">删除</button></td></tr>`).join('')}async function save(i){await fetch('/api/hotwords/'+i,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({term:t[i].value,weight:+w[i].value,enabled:e[i].checked,asr_risk:r[i].value,review_status:'pending',note:n[i].value})});load()}async function del(i){if(confirm('删除该热词？')){await fetch('/api/hotwords/'+i,{method:'DELETE'});load()}}async function add(){let term=prompt('热词');if(term){await fetch('/api/hotwords',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({term,weight:5,enabled:true,asr_risk:'low',review_status:'pending',note:''})});load()}}load()</script>'''
 @app.get('/api/hotwords')
 def all(q:str='',enabled:str='',risk:str=''):
  return [{'index':i,**r} for i,r in enumerate(read_jsonl(data)) if (not q or q.casefold() in str(r.get('term','')).casefold()) and (not enabled or str(r.get('enabled')).lower()==enabled.lower()) and (not risk or r.get('asr_risk')==risk)]
 @app.post('/api/hotwords')
 def add(v:Hotword):
  if not 1<=v.weight<=10: raise HTTPException(400,'weight 必须为 1～10')
  rows=read_jsonl(data); rows.append(v.model_dump());save(rows);return {'ok':True}
 @app.put('/api/hotwords/{index}')
 def edit(index:int,v:Hotword):
  if not 1<=v.weight<=10: raise HTTPException(400,'weight 必须为 1～10')
  rows=read_jsonl(data)
  if not 0<=index<len(rows): raise HTTPException(404,'记录不存在')
  rows[index].update(v.model_dump());save(rows);return {'ok':True}
 @app.delete('/api/hotwords/{index}')
 def delete(index:int):
  rows=read_jsonl(data)
  if not 0<=index<len(rows): raise HTTPException(404,'记录不存在')
  rows.pop(index);save(rows);return {'ok':True}
 @app.get('/api/export.txt',response_class=PlainTextResponse)
 def export(): return '\n'.join(r.get('term','') for r in read_jsonl(data) if r.get('enabled'))+'\n'
 return app
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=Path);p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8012);a=p.parse_args();import uvicorn;uvicorn.run(create_app(a.data.resolve()),host=a.host,port=a.port)
if __name__=='__main__':main()
