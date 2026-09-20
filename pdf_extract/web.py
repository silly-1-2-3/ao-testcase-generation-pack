"""Optional FastAPI adapter, mounted by pre/server.py before its static route.

No model inference here. Raw PDF uploads avoid an extra multipart dependency.
All uploaded documents and edits remain under a dedicated local data directory.
"""
from __future__ import annotations
import asyncio
import copy
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import tempfile
import uuid
import zipfile

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response

SOURCE=Path(__file__).resolve().parent/'src'
if str(SOURCE) not in sys.path: sys.path.insert(0,str(SOURCE))
LIMIT=40*1024*1024


def register_pdf_routes(app, project_root):
    # Keep runtime uploads outside the source checkout by default.  Some
    # Windows deployments copy the repository read-only; writing under
    # PROJECT_ROOT/output then raises WinError 5 before PDF parsing starts.
    default_root=Path(tempfile.gettempdir())/'ao_pdf_workflow'
    root=Path(os.environ.get('AO_PDF_WORK_DIR',str(default_root))).resolve()
    root.mkdir(parents=True,exist_ok=True)
    template_file=Path(os.environ.get('AO_PDF_TEMPLATES',str(SOURCE.parent/'file_templates.txt')))
    card_template_file=Path(os.environ.get('AO_PDF_CARD_TEMPLATES',str(SOURCE.parent/'card_templates.txt')))
    gate=asyncio.Semaphore(1)
    lock=threading.RLock()

    def modules():
        try:
            import pdf_to_html, structured, templates, cards, review
            return pdf_to_html,structured,templates,cards,review
        except ImportError as exc:
            raise HTTPException(503,'PDF dependencies missing; install pdf_extract/requirements.txt') from exc

    def folder(document_id):
        if not re.fullmatch('[0-9a-f]{32}',document_id):raise HTTPException(404,'Unknown document')
        path=root/document_id
        if not (path/'meta.json').is_file():raise HTTPException(404,'Unknown document')
        return path

    def read(path):return json.loads(path.read_text(encoding='utf8'))
    def save(path,data):
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        temp.replace(path)

    def snapshot(document_id):
        path=folder(document_id);meta=read(path/'meta.json');payload=read(path/'document.structured.short.json')
        _,_,_,cards,_=modules()
        import fitz
        with fitz.open(path/'source.pdf') as source:
            page_count=source.page_count
        return dict(**meta,document_id=document_id,payload=payload,
                    page_count=page_count,
                    cards=cards.make_cards(payload,document_id,meta['filename'],meta['card_headers']) if meta['approved'] and meta.get('cards_enabled',True) else [],
                    files={p.name:f'/api/pdf/{document_id}/files/{p.name}' for p in path.iterdir()
                           if p.is_file() and p.suffix in {'.html','.json','.pdf'} and p.name!='meta.json'})

    def convert(data,filename,file_template,headers,card_headers,cards_enabled=True):
        pdf,structured,templates,cards,review=modules()
        kind=file_template.strip() or Path(filename).stem
        selected=headers.strip() or templates.resolve(kind,template_file)
        selected_cards=card_headers.strip() or templates.resolve_card(kind,card_template_file) or cards.DEFAULT_HEADERS
        ident=uuid.uuid4().hex;path=root/ident;path.mkdir(parents=True)
        (path/'source.pdf').write_bytes(data)
        pdf.extract(path/'source.pdf',path/'document.cells.html')
        found=[];payload=structured.extract(path/'document.cells.html',selected,2.5,review=found)
        structured.write_outputs(payload,path/'document.structured.html',path/'document.structured.long.json',path/'document.structured.short.json')
        shutil.copyfile(path/'document.structured.short.json',path/'document.original.short.json')
        review.write_review(path/'document.cells.html',selected,2.5,found,{})
        save(path/'meta.json',dict(filename=filename,file_template=kind,headers=selected,
                                  card_headers=selected_cards,cards_enabled=bool(cards_enabled),approved=False,revision=0,
                                  warning=payload['selection'].get('fallback','')))
        return snapshot(ident)

    @app.get('/api/pdf/templates')
    async def get_templates():
        _,_,templates,cards,_=modules()
        return dict(templates=templates.load(template_file),card_templates=templates.load(card_template_file),card_headers=cards.DEFAULT_HEADERS,max_bytes=LIMIT)

    @app.post('/api/pdf/convert')
    async def upload(request:Request,filename:str='document.pdf',file_template:str='',headers:str='',card_headers:str='',cards:bool=True):
        # Not a file-system path import. A typed local path belongs in the file
        # chooser; remote clients must upload bytes, never access server paths.
        filename=filename.replace('\\','/').split('/')[-1][:180]
        if not filename.lower().endswith('.pdf'):raise HTTPException(400,'Only PDF files are supported')
        data=bytearray()
        async for chunk in request.stream():
            if len(data)+len(chunk)>LIMIT:raise HTTPException(413,'PDF exceeds 40 MiB')
            data.extend(chunk)
        if b'%PDF-' not in data[:1024]:raise HTTPException(400,'Not a PDF file')
        async with gate:
            try:return await asyncio.to_thread(convert,bytes(data),filename,file_template,headers,card_headers,cards)
            except HTTPException:raise
            except Exception as exc:raise HTTPException(422,f'PDF conversion failed: {exc}') from exc

    @app.get('/api/pdf/{document_id}')
    async def get_document(document_id:str):
        with lock:return snapshot(document_id)

    @app.delete('/api/pdf/{document_id}')
    async def delete_document(document_id:str):
        """Remove one uploaded document and its derived artifacts."""
        with lock:
            path=folder(document_id)
            shutil.rmtree(path)
        return {'deleted':document_id}

    @app.put('/api/pdf/{document_id}/tables/{index}')
    async def edit_table(document_id:str,index:int,request:Request):
        raw=await request.body()
        if len(raw)>4*1024*1024:raise HTTPException(413,'Table JSON exceeds 4 MiB')
        try:
            body=json.loads(raw);value=body['table'];expected=body['revision']
            _,structured,_,cards,_=modules();cards.validate_table(value)
        except (ValueError,KeyError,TypeError,RecursionError) as exc:raise HTTPException(422,str(exc)) from exc
        with lock:
            path=folder(document_id);meta=read(path/'meta.json')
            if meta['revision']!=expected:raise HTTPException(409,'Document changed; reload before editing')
            payload=read(path/'document.structured.short.json')
            if not 0<=index<len(payload['tables']):raise HTTPException(404,'Unknown table block')
            payload['tables'][index]=copy.deepcopy(value)
            save(path/'document.structured.short.json',payload)
            (path/'document.structured.html').write_text(structured.render_html(payload),encoding='utf8')
            import review
            review.write_review(path/'document.cells.html',meta['headers'],2.5,[],{})
            meta.update(approved=False,revision=meta['revision']+1);save(path/'meta.json',meta)
            return snapshot(document_id)

    @app.post('/api/pdf/{document_id}/approve')
    async def approve(document_id:str,request:Request):
        body=await request.json()
        with lock:
            path=folder(document_id);meta=read(path/'meta.json')
            if body.get('revision')!=meta['revision']:raise HTTPException(409,'Document changed; review again')
            _,_,_,cards,_=modules()
            for table in read(path/'document.structured.short.json')['tables']:cards.validate_table(table)
            meta['approved']=True
            if isinstance(body.get('card_headers'),str):meta['card_headers']=body['card_headers']
            save(path/'meta.json',meta)
            return snapshot(document_id)

    @app.get('/api/pdf/{document_id}/files/{relative:path}')
    async def artifact(document_id:str,relative:str):
        path=folder(document_id);target=(path/relative).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file() or target.name=='meta.json':
            raise HTTPException(404,'Unknown artifact')
        if target.suffix not in {'.pdf','.html','.json','.png','.jpg','.jpeg','.webp'}:raise HTTPException(404,'Unsupported artifact')
        return FileResponse(target,headers={'X-Content-Type-Options':'nosniff'})

    @app.get('/api/pdf/{document_id}/preview/{page}.png')
    async def preview(document_id:str,page:int):
        """Render the original PDF, not reconstructed cells; no browser PDF plugin required."""
        path=folder(document_id)
        def render():
            import fitz
            with fitz.open(path/'source.pdf') as source:
                if not 1<=page<=source.page_count:raise HTTPException(404,'Unknown PDF page')
                p=source[page-1]
                scale=min(1.6,1800/max(p.rect.width,p.rect.height))
                return p.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).tobytes('png')
        raw=await asyncio.to_thread(render)
        return Response(raw,media_type='image/png',headers={'Cache-Control':'private, max-age=3600'})

    @app.post('/api/pdf/export-results')
    async def export_results(request:Request):
        raw=await request.body()
        if len(raw)>32*1024*1024:raise HTTPException(413,'Export exceeds 32 MiB')
        try:
            items=json.loads(raw)['items']
            if not isinstance(items,list) or not 1<=len(items)<=1000:raise ValueError('Expected 1..1000 results')
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
                for index,item in enumerate(items,1):
                    if not isinstance(item,dict) or not isinstance(item.get('result'),dict):raise ValueError('Invalid result record')
                    label=re.sub(r'[^\w\u4e00-\u9fff.-]+','_',str(item.get('label','case')))[:80]
                    archive.writestr(f'ao_results/{index:04d}_{label}.json',json.dumps(item,ensure_ascii=False,indent=2,allow_nan=False))
            return Response(stream.getvalue(),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="ao_results.zip"'})
        except (ValueError,KeyError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
