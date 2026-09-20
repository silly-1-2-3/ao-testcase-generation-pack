"""Portable short-JSON review. Direction editing is intentionally absent."""
import hashlib
import json
from pathlib import Path

SCHEMA='pdf-table-directions-v1'  # Read old CLI overrides, not exposed in UI.
DIRECTIONS={'auto','vertical','horizontal','horizontal-pairs'}


def identity(cell_html,headers,tolerance):
    return dict(schema=SCHEMA,source_sha256=hashlib.sha256(cell_html.read_bytes()).hexdigest(),
                headers=headers,tolerance=tolerance)


def load_overrides(path,expected):
    if path is None:return {}
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if any(data.get(k)!=v for k,v in expected.items()):raise ValueError('Override source/settings do not match')
    values=data.get('overrides')
    if not isinstance(values,dict) or any(not isinstance(k,str) or v not in DIRECTIONS for k,v in values.items()):
        raise ValueError('Invalid direction overrides')
    return values


def write_review(cell_html,headers,tolerance,tables,overrides):
    folder=cell_html.parent
    payload=json.loads((folder/'document.structured.short.json').read_text(encoding='utf8'))
    manifest=dict(format='pdf-json-review-v1',headers=headers,payload=payload)
    (folder/'document.review.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    data=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    (folder/'document.review.html').write_text(TEMPLATE.replace('__DATA__',data),encoding='utf8')


TEMPLATE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PDF JSON 核验</title>
<style>body{font:15px system-ui;margin:0;color:#243447}header{padding:18px;border-bottom:1px solid #ddd}main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}#tables,iframe{height:78vh;overflow:auto}iframe{width:100%;border:1px solid #ddd}article{padding:12px;margin-bottom:14px;border:1px solid #ccd6df;border-radius:8px}table{border-collapse:collapse;width:100%;margin:8px 0}td,th{border:1px solid #bbc;padding:6px;white-space:pre-wrap}th{background:#eef3f7}button{padding:7px 12px;cursor:pointer}textarea{width:75vw;height:65vh}dialog{max-width:90vw}h2{font-size:17px}a{color:#1260a0}@media(max-width:850px){main{grid-template-columns:1fr}}</style>
<header><h1>PDF → 结构化内容核验</h1><p>表头决定方向。逐块核对，可替换该块的 short JSON。下载修订版不会覆盖磁盘原文件；long JSON 保留原始提取坐标。</p>
<button id="download">核验正确 · 下载 short JSON</button> <a href="document.cells.html" target="_blank">中间版式 HTML</a> · <a href="document.structured.long.json" download>原始 long JSON</a></header>
<main><div id="tables"></div><iframe id="source" src="source.pdf" title="PDF 原文"></iframe></main>
<dialog id="editor"><h2>修改原始 JSON（当前表格块）</h2><textarea id="json"></textarea><p id="error"></p><button id="save">替换此块</button> <button id="cancel">取消</button></dialog>
<script type="application/json" id="data">__DATA__</script><script>
const payload=JSON.parse(document.getElementById('data').textContent);let editing=-1;
function el(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n}
function render(t){const table=el('table'),head=el('tr');for(const h of t.headers||[])head.append(el('th',h));table.append(head);for(const row of t.rows||[]){const tr=el('tr');for(const c of row){const td=el('td');for(const v of c.content||[]){if(v.type==='image'){if(v.src.startsWith('images/')&&!v.src.includes('..')){const a=el('a','这里有一个图片');a.href=v.src;a.target='_blank';td.append(a)}}else td.append(el('div',v.text||''))}for(const child of c.nested_tables||[])td.append(render(child));tr.append(td)}table.append(tr)}const wrap=el('div');wrap.append(table);for(const child of t.nested_tables||[])wrap.append(render(child));return wrap}
function draw(){const root=document.getElementById('tables');root.replaceChildren();payload.tables.forEach((t,i)=>{const card=el('article');card.append(el('h2',`块 ${i+1} · 第 ${t.page||1} 页`));const b=el('button','修改原始 JSON');b.onclick=()=>{editing=i;document.getElementById('json').value=JSON.stringify(t,null,2);document.getElementById('error').textContent='';document.getElementById('editor').showModal()};card.append(b);const p=el('button','定位 PDF');p.onclick=()=>document.getElementById('source').src='source.pdf#page='+(t.page||1);card.append(p,render(t));root.append(card)})}
document.getElementById('save').onclick=()=>{try{const t=JSON.parse(document.getElementById('json').value);if(!Array.isArray(t.headers)||!Array.isArray(t.rows))throw Error('需要 headers 和 rows 数组');payload.tables[editing]=t;document.getElementById('editor').close();draw()}catch(e){document.getElementById('error').textContent=e.message}};
document.getElementById('cancel').onclick=()=>document.getElementById('editor').close();
document.getElementById('download').onclick=()=>{const a=el('a'),url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.href=url;a.download='document.reviewed.short.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};draw();
</script></html>'''
