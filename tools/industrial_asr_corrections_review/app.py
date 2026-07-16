#!/usr/bin/env python3
"""独立的工业实训 ASR 纠错规则维护工具。"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response


DEFAULT_JSON_PATH = Path("ai-service/app/services/conversation/industrial_asr_corrections.json")
CONTEXT_FIELDS = ("any", "all", "none")


def _issue(index: int | None, rule: Any, field: str, message: str) -> dict[str, Any]:
    return {
        "index": index,
        "rule_id": rule.get("id") if isinstance(rule, dict) else None,
        "field": field,
        "message": message,
    }


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def validate_config(config: dict) -> dict:
    """校验完整配置；未知字段只保留，不参与报错。"""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": [_issue(None, None, "root", "顶层必须是 JSON object")], "warnings": [], "rule_count": 0}
    version = config.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        errors.append(_issue(None, None, "version", "version 必须为整数"))
    elif version != 1:
        errors.append(_issue(None, None, "version", "当前仅支持 version=1"))
    rules = config.get("rules")
    if not isinstance(rules, list):
        errors.append(_issue(None, None, "rules", "rules 必须为 list"))
        return {"ok": False, "errors": errors, "warnings": warnings, "rule_count": 0}

    ids: dict[str, int] = {}
    variants: dict[str, list[tuple[int, dict]]] = {}
    canonicals: dict[str, list[int]] = {}
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(_issue(index, rule, "rule", "规则必须是 object"))
            continue
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            errors.append(_issue(index, rule, "id", "规则 ID 必须是非空字符串"))
        elif rule_id in ids:
            errors.append(_issue(index, rule, "id", f"规则 ID 重复，首次出现在第 {ids[rule_id] + 1} 条"))
        else:
            ids[rule_id] = index
        enabled = rule.get("enabled")
        if not isinstance(enabled, bool):
            errors.append(_issue(index, rule, "enabled", "enabled 必须为 boolean"))
        elif not enabled:
            warnings.append(_issue(index, rule, "enabled", "规则当前已禁用"))
        canonical = rule.get("canonical")
        if not isinstance(canonical, str) or not canonical.strip():
            errors.append(_issue(index, rule, "canonical", "canonical 必须为非空字符串"))
        else:
            canonicals.setdefault(canonical, []).append(index)
        rule_variants = rule.get("variants")
        if not isinstance(rule_variants, list) or not rule_variants:
            errors.append(_issue(index, rule, "variants", "variants 必须为非空字符串列表"))
        elif not _is_string_list(rule_variants) or any(not item.strip() for item in rule_variants):
            errors.append(_issue(index, rule, "variants", "variants 不能包含空值或非字符串"))
        else:
            seen_variants: set[str] = set()
            for variant in rule_variants:
                if variant in seen_variants:
                    warnings.append(_issue(index, rule, "variants", f"同一规则中的 variant 重复：{variant}"))
                seen_variants.add(variant)
                if isinstance(canonical, str) and variant == canonical:
                    errors.append(_issue(index, rule, "variants", "variant 不能与 canonical 相同"))
                variants.setdefault(variant, []).append((index, rule))
        mode = rule.get("match_mode")
        if mode not in {"direct", "contextual"}:
            errors.append(_issue(index, rule, "match_mode", "match_mode 必须为 direct 或 contextual"))
        priority = rule.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool):
            errors.append(_issue(index, rule, "priority", "priority 必须为整数"))
        context = rule.get("context")
        if mode == "direct" and context is not None:
            warnings.append(_issue(index, rule, "context", "direct 规则携带 context，不会参与匹配"))
        if mode == "contextual":
            if not isinstance(context, dict):
                errors.append(_issue(index, rule, "context", "contextual 规则必须包含 context object"))
            else:
                values: list[list[str]] = []
                for field in CONTEXT_FIELDS:
                    value = context.get(field)
                    if not _is_string_list(value):
                        errors.append(_issue(index, rule, f"context.{field}", f"context.{field} 必须为字符串列表"))
                    else:
                        values.append(value)
                if len(values) == 3 and not any(values):
                    warnings.append(_issue(index, rule, "context", "contextual 规则的 any/all/none 均为空"))

    for canonical, indexes in canonicals.items():
        if len(indexes) > 1:
            for index in indexes:
                errors.append(_issue(index, rules[index], "canonical", f"canonical 重复，首次出现在第 {indexes[0] + 1} 条：{canonical}"))
    for variant, entries in variants.items():
        if len(entries) < 2:
            continue
        modes = {rule.get("match_mode") for _, rule in entries}
        targets = {rule.get("canonical") for _, rule in entries}
        for index, rule in entries:
            warnings.append(_issue(index, rule, "variants", f"variant 出现在多条规则：{variant}"))
            if len(modes) > 1:
                warnings.append(_issue(index, rule, "variants", f"variant 同时存在 direct 与 contextual：{variant}"))
            if len(targets) > 1:
                warnings.append(_issue(index, rule, "variants", f"variant 对应不同 canonical：{variant}"))
    return {"ok": not errors, "errors": errors, "warnings": warnings, "rule_count": len(rules)}


def load_config(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        raise ValueError(f"JSON 文件不存在: {json_path}")
    if not json_path.is_file():
        raise ValueError(f"JSON 路径不是普通文件: {json_path}")
    try:
        config = json.loads(json_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON 文件无法按 UTF-8 读取: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {exc.msg}") from exc
    if not isinstance(config, dict):
        raise ValueError("JSON 顶层必须是 object")
    if "version" not in config or "rules" not in config:
        raise ValueError("JSON 顶层必须包含 version 和 rules")
    return config


def build_stats(config: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    rules = config.get("rules") if isinstance(config.get("rules"), list) else []
    return {
        "version": config.get("version"),
        "rule_count": len(rules),
        "enabled_count": sum(isinstance(rule, dict) and rule.get("enabled") is True for rule in rules),
        "direct_count": sum(isinstance(rule, dict) and rule.get("match_mode") == "direct" for rule in rules),
        "contextual_count": sum(isinstance(rule, dict) and rule.get("match_mode") == "contextual" for rule in rules),
        "error_count": len(validation["errors"]),
        "warning_count": len(validation["warnings"]),
    }


def _backup_path(json_path: Path) -> Path:
    backup_dir = json_path.parent / f"bak_{json_path.stem}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir / f"{json_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"


def save_config(json_path: Path, config: dict[str, Any]) -> None:
    validation = validate_config(config)
    if not validation["ok"]:
        raise ValueError("配置校验失败，不能保存")
    backup = _backup_path(json_path)
    try:
        shutil.copy2(json_path, backup)
    except Exception as exc:
        raise OSError("创建保存前备份失败，原文件未覆盖") from exc
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=json_path.parent, prefix=f".{json_path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
        temporary_path.replace(json_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def _rule_summary(index: int, rule: Any, validation: dict[str, Any]) -> dict[str, Any]:
    rule = rule if isinstance(rule, dict) else {}
    problems = [item for item in validation["errors"] + validation["warnings"] if item["index"] == index]
    return {
        "index": index, "id": rule.get("id"), "canonical": rule.get("canonical"),
        "variants": rule.get("variants", []), "enabled": rule.get("enabled"),
        "match_mode": rule.get("match_mode"), "priority": rule.get("priority"),
        "error_count": sum(item in validation["errors"] for item in problems),
        "warning_count": sum(item in validation["warnings"] for item in problems),
    }


def _next_copy_value(config: dict[str, Any], field: str, source_value: Any) -> str:
    existing = {
        rule.get(field) for rule in config.get("rules", []) if isinstance(rule, dict)
    }
    base = f"{source_value or field}_copy"
    candidate, number = base, 2
    while candidate in existing:
        candidate = f"{base}{number}"
        number += 1
    return candidate


PAGE_HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>工业实训 ASR 纠错规则维护</title><style>
:root{font-family:system-ui,sans-serif;color:#1f2933;background:#f4f7f8}*{box-sizing:border-box}body{height:100vh;margin:0;display:flex;flex-direction:column;overflow:hidden}header{flex:0 0 auto;padding:14px 20px;color:#fff;background:#163a4a}h1{margin:0;font-size:20px}header p{margin:6px 0 0;font-size:12px;overflow-wrap:anywhere}.stats,.actions{display:flex;flex-wrap:wrap;gap:9px;align-items:center;padding:10px 18px;background:#fff;border-bottom:1px solid #ccd6db}.stats{flex:0 0 auto}.stats span{font-size:13px}.layout{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:minmax(300px,360px) minmax(0,1fr);overflow:hidden}.left-panel{min-height:0;display:flex;flex-direction:column;overflow:hidden;padding:12px;border-right:1px solid #ccd6db;background:#fff}.left-toolbar{flex:0 0 auto}.filters{display:grid;grid-template-columns:1fr 1fr;gap:8px}.filters input{grid-column:1/-1}.left-toolbar .actions{padding:10px 0 0;background:transparent;border:0}.rule-list{flex:1 1 auto;min-height:0;margin-top:10px;padding-right:6px;overflow-y:auto;overflow-x:hidden;overscroll-behavior:contain}button,input,textarea,select{font:inherit}button{padding:7px 10px;border:1px solid #9aabb3;border-radius:5px;background:#fff;cursor:pointer}button.primary{background:#176b4d;color:#fff;border-color:#176b4d}button.danger{color:#9b2c2c}.rule{display:block;width:100%;margin:7px 0;padding:9px;text-align:left}.rule.active{border-color:#277b9b;background:#eaf6fb}.badge{display:inline-block;margin-right:5px;padding:2px 6px;border-radius:10px;background:#e8eef0;font-size:11px}.bad{color:#a12622;background:#feecec}.warn{color:#815600;background:#fff3d9}.detail{min-width:0;min-height:0;overflow:auto;padding:18px;max-width:none}.form{display:grid;grid-template-columns:1fr 1fr;gap:13px}.field{display:flex;flex-direction:column;gap:5px}.full{grid-column:1/-1}input,textarea,select{width:100%;padding:8px;border:1px solid #9aabb3;border-radius:5px}textarea{min-height:100px;resize:vertical}.context{border:1px solid #ccd6db;border-radius:6px;padding:12px}.hidden{display:none}#issues{white-space:pre-wrap;color:#8b251f}.dirty{color:#9a6200}@media(max-width:760px){body{height:auto;min-height:100vh;overflow:auto}.layout{display:block;overflow:visible}.left-panel{height:45vh;min-height:260px;border-right:0;border-bottom:1px solid #ccd6db}.form{grid-template-columns:1fr}}</style></head><body>
<header><h1>工业实训 ASR 纠错规则维护</h1><p id="path"></p></header><div class="stats"><span>version: <b id="version"></b></span><span>规则: <b id="total"></b></span><span>启用: <b id="enabled"></b></span><span>direct: <b id="direct"></b></span><span>contextual: <b id="contextual"></b></span><span>错误: <b id="errors"></b></span><span>警告: <b id="warnings"></b></span><span id="status">就绪</span></div>
<div class="layout"><aside class="left-panel"><div class="left-toolbar"><div class="filters"><input id="search" placeholder="搜索 id、术语、variants、context"><select id="enabled-filter"><option value="all">全部启用状态</option><option value="enabled">仅启用</option><option value="disabled">仅禁用</option></select><select id="mode-filter"><option value="all">全部模式</option><option value="direct">direct</option><option value="contextual">contextual</option></select></div><div class="actions"><button id="new">新建规则</button><button id="copy">复制规则</button><button id="delete" class="danger">删除规则</button><button id="reload">从磁盘重新加载</button></div></div><div id="rules" class="rule-list"></div></aside>
<main class="detail"><div id="empty">请选择或新建规则。</div><div id="editor" class="hidden"><div class="form"><label class="field">id<input id="id"></label><label class="field">enabled<input id="enabled-input" type="checkbox"></label><label class="field">canonical<input id="canonical"></label><label class="field">match_mode<select id="mode"><option value="direct">direct</option><option value="contextual">contextual</option></select></label><label class="field">priority<input id="priority" type="number" step="1"></label><label class="field full">variants（一行一个）<textarea id="variants"></textarea></label></div><section id="context" class="context"><h2>context</h2><div class="form"><label class="field">any（一行一个）<textarea id="context-any"></textarea></label><label class="field">all（一行一个）<textarea id="context-all"></textarea></label><label class="field full">none（一行一个）<textarea id="context-none"></textarea></label></div></section><div class="actions"><button id="save" class="primary">保存当前规则</button><button id="save-all" class="primary">保存全部</button><button id="validate">校验全部</button><button id="prev">上一条</button><button id="next">下一条</button><button id="download">下载 JSON</button></div><pre id="issues"></pre></div></main></div>
<script>
const state={config:null,validation:null,index:null,loaded:null,dirty:false,isNew:false};const $=id=>document.getElementById(id);const split=v=>[...new Set(v.split(/\\r?\\n/).map(x=>x.trim()).filter(Boolean))];
function msg(t,error=false){$('status').textContent=t;$('status').className=error?'bad':state.dirty?'dirty':''}function setDirty(v=true){state.dirty=v;msg(v?'有未保存修改':'已保存')}
function stats(){const s=state.stats;$('version').textContent=s.version;$('total').textContent=s.rule_count;$('enabled').textContent=s.enabled_count;$('direct').textContent=s.direct_count;$('contextual').textContent=s.contextual_count;$('errors').textContent=s.error_count;$('warnings').textContent=s.warning_count}
function matches(r){const q=$('search').value.trim().toLowerCase();const ef=$('enabled-filter').value,mf=$('mode-filter').value;if(ef==='enabled'&&!r.enabled||ef==='disabled'&&r.enabled||mf!=='all'&&r.match_mode!==mf)return false;return !q||[r.id,r.canonical,...(r.variants||[]),...Object.values((r.context)||{}).flat()].join(' ').toLowerCase().includes(q)}
function renderList(){const box=$('rules');box.replaceChildren();(state.rules||[]).filter(matches).forEach(r=>{const b=document.createElement('button');b.className='rule'+(r.index===state.index?' active':'');b.textContent='#'+(r.index+1)+' '+(r.canonical||'(未填写)')+' ['+(r.variants||[]).join('、')+'] '+(r.id||'')+' '+(r.match_mode||'')+' P'+r.priority;b.onclick=()=>open(r.index);if(r.error_count){const x=document.createElement('span');x.className='badge bad';x.textContent='错误 '+r.error_count;b.appendChild(x)}if(r.warning_count){const x=document.createElement('span');x.className='badge warn';x.textContent='警告 '+r.warning_count;b.appendChild(x)}box.appendChild(b)})}
function formRule(){const old=state.index===null?{}:state.config.rules[state.index];const rule=Object.assign({},old,{id:$('id').value,enabled:$('enabled-input').checked,canonical:$('canonical').value,variants:split($('variants').value),match_mode:$('mode').value,priority:Number($('priority').value)});if(rule.match_mode==='contextual')rule.context=Object.assign({},old.context||{},{any:split($('context-any').value),all:split($('context-all').value),none:split($('context-none').value)});else if(old.context===undefined)delete rule.context;return rule}
function fill(rule){$('empty').classList.add('hidden');$('editor').classList.remove('hidden');$('id').value=rule.id||'';$('enabled-input').checked=rule.enabled===true;$('canonical').value=rule.canonical||'';$('mode').value=rule.match_mode||'direct';$('priority').value=Number.isInteger(rule.priority)?rule.priority:100;$('variants').value=(rule.variants||[]).join('\\n');const c=rule.context||{};$('context-any').value=(c.any||[]).join('\\n');$('context-all').value=(c.all||[]).join('\\n');$('context-none').value=(c.none||[]).join('\\n');toggleContext();state.loaded=JSON.stringify(rule);state.dirty=false;msg('已加载')}
function toggleContext(){$('context').classList.toggle('hidden',$('mode').value==='direct')}
async function load(){const r=await fetch('/api/config');if(!r.ok)throw Error(await r.text());const d=await r.json();state.config=d.config;state.validation=d.validation;state.stats=d.stats;state.rules=d.rules;$('path').textContent='当前文件：'+d.json_path;stats();renderList();showIssues()}
function showIssues(){const all=[...(state.validation?.errors||[]),...(state.validation?.warnings||[])];$('issues').textContent=all.map(x=>(state.validation.errors.includes(x)?'错误':'警告')+' #'+(x.index===null?'-':x.index+1)+' ['+x.field+'] '+x.message).join('\\n')}
async function open(i){if(state.dirty&&!confirm('当前规则有未保存修改，确定切换吗？'))return;state.isNew=false;state.index=i;fill(state.config.rules[i]);renderList()}
async function saveCurrent(){if(state.index===null&&!state.isNew){msg('请先新建或选择规则',true);return}const isNew=state.isNew,oldIndex=state.index,rule=formRule(),url=isNew?'/api/rules':'/api/rules/'+oldIndex,method=isNew?'POST':'PUT';const r=await fetch(url,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(rule)});if(!r.ok){msg('保存失败：'+await r.text(),true);return}const data=await r.json();const savedIndex=isNew?data.stats.rule_count-1:oldIndex;state.isNew=false;state.dirty=false;await load();await open(savedIndex);msg(isNew?'已新建并保存':'已保存')}
async function saveAll(){if(state.isNew){await saveCurrent();return}if(state.index!==null)state.config.rules[state.index]=formRule();const r=await fetch('/api/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(state.config)});if(!r.ok){msg('保存失败：'+await r.text(),true);return}await load();if(state.index!==null)fill(state.config.rules[state.index]);msg('已保存全部')}
function newRule(){if(state.dirty&&!confirm('放弃当前未保存修改吗？'))return;state.isNew=true;state.index=null;fill({id:'',enabled:true,canonical:'',variants:[],match_mode:'direct',priority:100});setDirty(true);renderList()}
function copyRule(){if(state.index===null||state.isNew)return;const r=structuredClone(formRule()),copyValue=(field,fallback)=>{const base=(r[field]||fallback)+'_copy',values=new Set(state.config.rules.map(x=>x[field]));let value=base,n=2;while(values.has(value)){value=base+n;n++}return value};r.id=copyValue('id','rule');r.canonical=copyValue('canonical','术语');state.isNew=true;state.index=null;fill(r);setDirty(true);renderList()}
async function del(){if(state.isNew){if(confirm('确定放弃新建规则吗？')){state.isNew=false;state.index=null;$('editor').classList.add('hidden');$('empty').classList.remove('hidden');setDirty(false);renderList()}return}if(state.index===null)return;const r=formRule();if(!confirm('确定删除规则「'+(r.canonical||'未命名')+'」吗？'))return;const res=await fetch('/api/rules/'+state.index,{method:'DELETE'});if(!res.ok){msg('删除失败：'+await res.text(),true);return}state.index=null;await load();msg('已删除')}
['id','enabled-input','canonical','mode','priority','variants','context-any','context-all','context-none'].forEach(id=>$(id).addEventListener('input',()=>{toggleContext();setDirty()}));$('search').oninput=renderList;$('enabled-filter').onchange=renderList;$('mode-filter').onchange=renderList;$('save').onclick=saveCurrent;$('save-all').onclick=saveAll;$('validate').onclick=async()=>{const r=await fetch('/api/validate',{method:'POST'});state.validation=await r.json();state.stats.error_count=state.validation.errors.length;state.stats.warning_count=state.validation.warnings.length;stats();showIssues();renderList()};$('reload').onclick=async()=>{if(state.dirty&&!confirm('当前规则有未保存修改，确定重新加载吗？'))return;const r=await fetch('/api/reload',{method:'POST'});if(!r.ok){msg(await r.text(),true);return}state.index=null;await load();msg('已从磁盘重新加载')};$('new').onclick=newRule;$('copy').onclick=copyRule;$('delete').onclick=del;$('prev').onclick=()=>open(Math.max(0,(state.index??0)-1));$('next').onclick=()=>open(Math.min(state.config.rules.length-1,(state.index??-1)+1));$('download').onclick=()=>location.href='/api/export';window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue=''}});window.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();saveCurrent()}});load().catch(e=>msg('加载失败：'+e.message,true));
</script></body></html>"""


def create_app(json_path: Path) -> FastAPI:
    json_path = json_path.expanduser().resolve()
    config = load_config(json_path)
    app = FastAPI(title="Industrial ASR Corrections Review", docs_url=None, redoc_url=None)
    app.state.json_path = json_path
    app.state.config = config
    app.state.save_lock = threading.Lock()

    def response_payload() -> dict[str, Any]:
        validation = validate_config(app.state.config)
        return {
            "config": copy.deepcopy(app.state.config), "validation": validation,
            "stats": build_stats(app.state.config, validation),
            "rules": [_rule_summary(index, rule, validation) for index, rule in enumerate(app.state.config.get("rules", []))],
            "json_path": str(app.state.json_path),
        }

    def persist(candidate: dict[str, Any]) -> dict[str, Any]:
        validation = validate_config(candidate)
        if not validation["ok"]:
            raise HTTPException(status_code=400, detail={"message": "配置校验失败", "validation": validation})
        try:
            save_config(app.state.json_path, candidate)
        except (OSError, ValueError):
            raise HTTPException(status_code=500, detail="保存失败，原文件未覆盖")
        app.state.config = candidate
        return response_payload()

    @app.get("/", response_class=HTMLResponse)
    async def page() -> str:
        return PAGE_HTML

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        with app.state.save_lock:
            return response_payload()

    @app.get("/api/rules")
    async def get_rules() -> dict[str, Any]:
        payload = response_payload()
        return {"rules": payload["rules"], "stats": payload["stats"], "validation": payload["validation"]}

    @app.get("/api/rules/{index}")
    async def get_rule(index: int) -> dict[str, Any]:
        rules = app.state.config.get("rules", [])
        if index < 0 or index >= len(rules):
            raise HTTPException(status_code=404, detail="规则不存在")
        return {"index": index, "rule": copy.deepcopy(rules[index])}

    @app.put("/api/rules/{index}")
    async def update_rule(index: int, rule: dict[str, Any] = Body(...)) -> dict[str, Any]:
        with app.state.save_lock:
            candidate = copy.deepcopy(app.state.config)
            rules = candidate.get("rules", [])
            if index < 0 or index >= len(rules):
                raise HTTPException(status_code=404, detail="规则不存在")
            rules[index] = rule
            return persist(candidate)

    @app.post("/api/rules")
    async def create_rule(rule: dict[str, Any] = Body(...)) -> dict[str, Any]:
        with app.state.save_lock:
            candidate = copy.deepcopy(app.state.config)
            candidate["rules"].append(rule)
            return persist(candidate)

    @app.post("/api/rules/{index}/duplicate")
    async def duplicate_rule(index: int) -> dict[str, Any]:
        with app.state.save_lock:
            candidate = copy.deepcopy(app.state.config)
            rules = candidate.get("rules", [])
            if index < 0 or index >= len(rules):
                raise HTTPException(status_code=404, detail="规则不存在")
            duplicated = copy.deepcopy(rules[index])
            duplicated["id"] = _next_copy_value(candidate, "id", duplicated.get("id"))
            duplicated["canonical"] = _next_copy_value(
                candidate, "canonical", duplicated.get("canonical")
            )
            rules.insert(index + 1, duplicated)
            payload = persist(candidate)
            payload["index"] = index + 1
            return payload

    @app.delete("/api/rules/{index}")
    async def delete_rule(index: int) -> dict[str, Any]:
        with app.state.save_lock:
            candidate = copy.deepcopy(app.state.config)
            rules = candidate.get("rules", [])
            if index < 0 or index >= len(rules):
                raise HTTPException(status_code=404, detail="规则不存在")
            rules.pop(index)
            return persist(candidate)

    @app.post("/api/validate")
    async def validate() -> dict[str, Any]:
        return validate_config(app.state.config)

    @app.post("/api/reload")
    async def reload_config() -> dict[str, Any]:
        with app.state.save_lock:
            try:
                app.state.config = load_config(app.state.json_path)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return response_payload()

    @app.get("/api/export")
    async def export() -> Response:
        content = json.dumps(app.state.config, ensure_ascii=False, indent=2) + "\n"
        return Response(content, media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{app.state.json_path.name}"'})

    @app.put("/api/config")
    async def update_config(config: dict[str, Any] = Body(...)) -> dict[str, Any]:
        with app.state.save_lock:
            return persist(copy.deepcopy(config))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="工业实训 ASR 纠错规则维护工具")
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        app = create_app(args.json_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
