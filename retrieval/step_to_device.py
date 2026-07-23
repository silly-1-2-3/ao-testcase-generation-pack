#!/usr/bin/env python3
"""
step_to_device.py - 从测试用例步骤提取设备信息并检索设备指令号

流程:
  1. 从测试用例表格中提取所有包含设备的执行步骤
  2. 可选: 提取父结构(判据描述、步骤说明)增强查询
  3. [可选] AI 扩写查询 (HyDE风格)
  4. BM25 + BGE 双路检索 -> RRF 融合
  5. 返回每个步骤的最佳设备指令号
"""
import json, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_pipeline import load_commands, search_bm25, search_bge, rrf_fusion

# 步骤里有设备的标志字段
DEVICE_FIELDS = ['设备类型', '设备单元号', '设备指令号', '设备参数']
CONTEXT_FIELDS = ['说明', '操作内容', '操作对象', '操作目的', '判据描述', '操作类型']

def extract_device_steps(table_rows):
    """
    从测试用例表格中提取包含设备信息的步骤行
    返回: [(row_idx, step_data, context_text), ...]
    """
    device_steps = []
    for i, row in enumerate(table_rows):
        if not isinstance(row, dict):
            continue
        # 检查是否有设备相关字段
        has_device = False
        for f in DEVICE_FIELDS:
            v = row.get(f, '')
            if v and str(v).strip() not in ('', '[]', 'null', '无', '否'):
                has_device = True
                break
        # 也检查"是否使用设备"字段
        if str(row.get('是否使用设备', '')).strip() == '是':
            has_device = True
        
        if has_device:
            # 构建查询文本: 所有非空字段拼接
            query_parts = []
            for f in CONTEXT_FIELDS:
                v = row.get(f, '')
                if v and str(v).strip() not in ('', '[]'):
                    query_parts.append(str(v).strip())
            query_text = '。'.join(query_parts) if query_parts else ''
            
            device_steps.append({
                'row_idx': i,
                'row': row,
                'query_text': query_text,
                'original_device': {
                    '设备类型': str(row.get('设备类型', '')).strip(),
                    '设备指令号': str(row.get('设备指令号', '')).strip(),
                    '设备单元号': str(row.get('设备单元号', '')).strip(),
                }
            })
    return device_steps

def rewrite_query_for_retrieval(query_text, api_url=None, api_key=None, model='qwen3-max'):
    """HyDE: 将步骤描述改写为设备功能需求"""
    if not query_text or not api_url:
        return query_text
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_url)
        prompt = (
            '你是航空维修设备专家。以下文本来自AMM测试用例的一个步骤，描述了使用某设备完成某项操作。'
            '请提取并改写为一段设备功能需求描述（30-80字），聚焦于"这个设备需要具备什么功能"。'
            '规则: 1)保留设备名/部件名/参数 2)以设备功能而非操作步骤的视角描述 3)不添加原文没有的信息。'
            '只输出改写文本。'
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': query_text}],
            temperature=0.1, max_tokens=256
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten or query_text
    except Exception as e:
        return query_text

def process_table(table_json_or_rows, commands, use_rewrite=False, api_url=None, api_key=None, top_k=5):
    """
    处理一个完整的测试用例表格
    """
    if isinstance(table_json_or_rows, str):
        rows = json.loads(table_json_or_rows)
    else:
        rows = table_json_or_rows
    
    device_steps = extract_device_steps(rows)
    results = []
    
    for ds in device_steps:
        query = ds['query_text']
        
        # AI扩写
        if use_rewrite and query:
            rewritten = rewrite_query_for_retrieval(query, api_url, api_key)
            if rewritten != query:
                ds['rewritten_query'] = rewritten
                query = rewritten
        
        # 检索
        r_bm25 = search_bm25(query, commands, top_k * 3) if query else []
        r_bge = []  # BGE is optional; enable if model available
        try:
            r_bge = search_bge(query, commands, top_k * 3) if query else []
        except:
            pass
        
        if r_bm25 and r_bge:
            ranked = rrf_fusion(r_bm25, r_bge, top_k=top_k)
        elif r_bm25:
            ranked = [(r['command_id'], r['bm25']) for r in r_bm25[:top_k]]
        else:
            ranked = []
        
        ds['retrieved'] = [
            {
                'command_id': cid,
                'device_type': next((c.get('设备类型','') for c in commands if c.get('设备指令号','')==cid), ''),
                'score': round(score, 4)
            }
            for cid, score in ranked
        ]
        ds['best_match'] = ds['retrieved'][0] if ds['retrieved'] else None
        results.append(ds)
    
    return results

def main():
    p = argparse.ArgumentParser(description='从测试用例步骤检索设备指令号')
    p.add_argument('--input', required=True, help='测试用例JSON文件 (包含rows数组)')
    p.add_argument('--data', default=None, help='设备指令字典路径')
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--rewrite', action='store_true', help='AI扩写查询')
    p.add_argument('--api_url', default=None)
    p.add_argument('--api_key', default=None)
    p.add_argument('--output', default=None, help='输出JSON路径')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    
    # Load commands
    if args.data is None:
        base = os.path.dirname(__file__)
        args.data = os.path.join(base, 'devices_index.jsonl')
    commands = load_commands(args.data)
    print(f'Corpus: {len(commands)} commands')
    
    # Load test case
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different input formats
    if 'rows' in data:
        rows = data['rows']
    elif 'pred_rows' in data:
        rows = data['pred_rows']
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    
    print(f'Table rows: {len(rows)}')
    
    # Process
    results = process_table(rows, commands, args.rewrite, args.api_url, args.api_key, args.top_k)
    
    print(f'\nDevice-related steps found: {len(results)}')
    for i, r in enumerate(results):
        print(f'\n--- Step {i+1} (row {r["row_idx"]}) ---')
        print(f'  Original device: {r["original_device"]["设备类型"]} / {r["original_device"]["设备指令号"]}')
        print(f'  Query: {r["query_text"][:120]}')
        if r.get('rewritten_query'):
            print(f'  Rewritten: {r["rewritten_query"][:120]}')
        if r.get('best_match'):
            print(f'  Best match: {r["best_match"]["command_id"]} ({r["best_match"]["device_type"]}) score={r["best_match"]["score"]}')
            if r['original_device']['设备指令号'] and r['original_device']['设备指令号'] != '[]':
                hit = any(m['command_id'] == r['original_device']['设备指令号'] for m in r['retrieved'])
                print(f'  MATCH: {"YES" if hit else "NO (original not in top-K)"}')
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'\nSaved: {args.output}')

if __name__ == '__main__':
    main()
