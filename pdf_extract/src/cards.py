"""Split reviewed short tables into complete, independently selectable AO inputs."""
import copy
import json
import structured

DEFAULT_HEADERS = '工种,序号,工序内容|序号,项目内容'


def validate_table(table, depth=0):
    if depth>16 or not isinstance(table,dict):
        raise ValueError('Table must be an object; nesting must not exceed 16 levels')
    if not isinstance(table.get('headers'),list) or not table['headers'] or not all(isinstance(h,str) for h in table['headers']):
        raise ValueError('headers must be a nonempty string array')
    if not isinstance(table.get('rows'),list):
        raise ValueError('rows must be an array')
    for row in table['rows']:
        if not isinstance(row,list): raise ValueError('Every row must be an array')
        for cell in row:
            if not isinstance(cell,dict): raise ValueError('Every cell must be an object')
            column=cell.get('column')
            if type(column) is not int or not 0<=column<len(table['headers']):
                raise ValueError('Cell column is outside headers')
            if not isinstance(cell.get('content',[]),list): raise ValueError('content must be an array')
            for item in cell.get('content',[]):
                if not isinstance(item,dict): raise ValueError('content item must be an object')
                if item.get('type')=='image':
                    src=item.get('src','')
                    if not isinstance(src,str) or not src.startswith('images/') or '..' in src or '\\' in src:
                        raise ValueError('Image src must refer to a local images/ resource')
                elif not isinstance(item.get('text'),str): raise ValueError('Text content needs a text string')
            children=cell.get('nested_tables',[])
            if not isinstance(children,list): raise ValueError('nested_tables must be an array')
            for child in children: validate_table(child,depth+1)
    children=table.get('nested_tables',[])
    if not isinstance(children,list): raise ValueError('nested_tables must be an array')
    for child in children: validate_table(child,depth+1)
    json.dumps(table,allow_nan=False)


def make_cards(payload, doc_id, filename, header_spec=DEFAULT_HEADERS):
    specs=structured.parse_header_specs(header_spec)
    result=[]
    def matches(t): return structured._matches_header(t.get('headers',[]),specs)
    def children(t):
        return t.get('nested_tables',[])+[child for row in t.get('rows',[]) for c in row for child in c.get('nested_tables',[])]
    def has_match(t): return matches(t) or any(has_match(c) for c in children(t))
    def add(t,path,label,context):
        content=copy.deepcopy(t)
        if context: content={'context':context,'table':content}
        # Card image URLs point to the actual server resource; stored short JSON
        # stays portable with local relative paths.
        def urls(obj):
            if isinstance(obj,dict):
                if doc_id!='local' and obj.get('type')=='image' and obj.get('src','').startswith('images/'):
                    obj['src']=f'/api/pdf/{doc_id}/files/'+obj['src']
                for v in obj.values(): urls(v)
            elif isinstance(obj,list):
                for v in obj: urls(v)
        urls(content)
        result.append(dict(id=f'{doc_id}:{path}',document_id=doc_id,filename=filename,
                           label=label,text=json.dumps(content,ensure_ascii=False,indent=2)))
    def visit(t,path,context):
        if matches(t) and t.get('orientation','vertical')=='vertical':
            seq=next((i for i,h in enumerate(t['headers']) if structured._norm(h) in {'序号','工步','编号'}),None)
            groups={}
            ordered=[]
            for ri,row in enumerate(t.get('rows',[])):
                kids=[c for cell in row for c in cell.get('nested_tables',[]) if has_match(c)]
                if kids:
                    parent=copy.deepcopy(row)
                    for cell in parent:
                        # Keep reference/record tables that are not split into cards.
                        remaining=[c for c in cell.get('nested_tables',[]) if not has_match(c)]
                        if remaining: cell['nested_tables']=remaining
                        else: cell.pop('nested_tables',None)
                    # The outer step is a real input block too.  Its text is
                    # kept as the parent card; matched child tables are then
                    # emitted as additional, independent cards below.
                    parent_index=''
                    if seq is not None:
                        parent_cell=next((c for c in parent if c.get('column')==seq),{})
                        parent_index=''.join(x.get('text','') for x in parent_cell.get('content',[])).strip()
                    parent_key=parent_index or f'row-{ri+1}'
                    parent_value={k:copy.deepcopy(v) for k,v in t.items() if k not in {'rows','nested_tables'}}
                    parent_value['rows']=[parent]
                    ordered.append(('parent',parent_value,f'{path}.{parent_key}',f'工步 {parent_key}' if seq is not None else parent_key))
                    for ci,kid in enumerate(kids):
                        ordered.append(('child',kid,f'{path}.r{ri}.n{ci}',context+[{'headers':t['headers'],'rows':[parent]}]))
                    continue
                index=''
                if seq is not None:
                    cell=next((c for c in row if c.get('column')==seq),{})
                    index=''.join(x.get('text','') for x in cell.get('content',[])).strip()
                key=index or f'row-{ri+1}'
                if key not in groups:
                    groups[key]=[]
                    ordered.append(('group',key))
                groups[key].append(row)
            for event in ordered:
                if event[0]=='parent':
                    add(event[1],event[2],event[3],context)
                    continue
                if event[0]=='child':
                    visit(*event[1:])
                    continue
                key=event[1]
                rows=groups[key]
                value={k:copy.deepcopy(v) for k,v in t.items() if k not in {'rows','nested_tables'}}
                value['rows']=rows
                add(value,f'{path}.{key}',f'工步 {key}' if seq is not None else key,context)
            for ci,kid in enumerate(t.get('nested_tables',[])): visit(kid,f'{path}.n{ci}',context)
        elif any(has_match(k) for k in children(t)):
            for ci,kid in enumerate(children(t)):
                if has_match(kid):visit(kid,f'{path}.n{ci}',context)
        else:
            add(t,path,t.get('title') or ' / '.join(t.get('headers',[])),context)
    for index,t in enumerate(payload.get('tables',[])): visit(t,str(index),[])
    return result
