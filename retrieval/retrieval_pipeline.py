#!/usr/bin/env python3
"""
retrieval_pipeline.py - 设备指令检索 (v6 final)

输入: plain text (测试用例中某步骤的所有字段拼接)
输出: Top-K 设备指令号

流程:
  1. [可选] LLM 改写查询为设备功能需求描述 (HyDE)
  2. BM25 + BGE 双路检索, RRF 融合
  3. 返回 Top-K 指令号
"""
import json, os, sys, argparse, re
sys.path.insert(0, os.path.dirname(__file__))

def load_commands(data_path=None):
    if data_path is None:
        base = os.path.dirname(__file__)
        enriched = os.path.join(base, 'devices_instr_with_desc.jsonl')
        original = os.path.join(base, 'devices_full.jsonl')
        data_path = enriched if os.path.exists(enriched) else original
    commands = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                dtype = obj.get('设备类型','')
                cid = obj.get('设备指令号','')
                desc = obj.get('设备指令功能说明','')
                obj['_text'] = f'{dtype} {desc} {cid}'.strip()
                commands.append(obj)
    return commands

def rewrite_query(query, api_url=None, api_key=None, model='qwen3-max'):
    """HyDE: 改写查询为设备功能需求描述, 保持术语不变"""
    if not (api_url and api_key):
        return query
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_url)
        prompt = (
            '航空维修设备专家。以下文本来自测试用例表格的一个步骤行。'
            '改写为一小段设备功能需求描述（30-80字），风格类似设备指令功能说明。\n'
            '规则: 1)聚焦设备需要完成什么功能 2)保留原文的设备名/部件名/参数 3)不添加原文没有的信息。'
            '只输出改写文本。'
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role':'system','content':prompt},{'role':'user','content':query}],
            temperature=0.1, max_tokens=256
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten or query
    except Exception as e:
        print(f'[WARN] rewrite failed: {e}')
        return query

_bm25_cache = {}  # {(data_path_hash,): (BM25, docs)}

def search_bm25(query, commands, top_k=10):
    from bm25_index import BM25
    cmd_ids = tuple(sorted(c['设备指令号'] for c in commands))
    if cmd_ids in _bm25_cache:
        bm25, docs = _bm25_cache[cmd_ids]
    else:
        docs = [{'设备类型':c['设备类型'],'设备指令号':c['设备指令号'],'_text':c['_text']} for c in commands]
        bm25 = BM25()
        bm25.build_doc_text = lambda doc: doc['_text']
        bm25.fit(docs)
        _bm25_cache[cmd_ids] = (bm25, docs)
    return [{'command_id':d['设备指令号'],'device_type':d['设备类型'],'bm25':round(s,4)} for d,s in bm25.search(query,top_k)]

_bge_reranker = None

def search_bge(query, commands, top_k=10):
    """BGE semantic reranking. Falls back to empty on any error."""
    global _bge_reranker
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        if _bge_reranker is None:
            from bge_reranker import BGEReranker
            _bge_reranker = BGEReranker(model_name='BAAI/bge-small-zh-v1.5')
        reranker = _bge_reranker
        docs = [{'设备类型':c['设备类型'],'设备指令号':c['设备指令号'],'说明':c['_text']} for c in commands]
        results = reranker.rerank(query, docs, top_k=top_k)
        return [{'command_id':d['设备指令号'],'device_type':d['设备类型'],'bge':round(s,4)} for d,s in results]
    except Exception as e:
        print(f'[WARN] BGE rerank failed: {e}')
        return []

def rrf_fusion(bm25_results, bge_results, k=60, top_k=10):
    scores = {}
    for rank, r in enumerate(bm25_results):
        scores[r['command_id']] = scores.get(r['command_id'],0) + 1.0/(k+rank+1)
    for rank, r in enumerate(bge_results):
        scores[r['command_id']] = scores.get(r['command_id'],0) + 1.0/(k+rank+1)
    return sorted(scores.items(), key=lambda x:-x[1])[:top_k]

def main():
    p = argparse.ArgumentParser(description='设备指令检索: plain text -> Top-K 设备指令号')
    p.add_argument('--query', required=True, help='plain text, 步骤全部字段拼接')
    p.add_argument('--method', default='both', choices=['bm25','bge','both'])
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--data', default=None, help='设备指令字典路径')
    p.add_argument('--rewrite', action='store_true', help='LLM HyDE 改写')
    p.add_argument('--api_url', default=None)
    p.add_argument('--api_key', default=None)
    p.add_argument('--api_model', default='qwen3-max')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    commands = load_commands(args.data)
    has_desc = any(c.get('设备指令功能说明','') for c in commands)
    print(f'Commands: {len(commands)} | Desc: {"REAL" if has_desc else "EMPTY"} | Method: {args.method}')

    query = args.query
    if args.rewrite:
        rewritten = rewrite_query(query, args.api_url, args.api_key, args.api_model)
        if rewritten != query:
            print(f'[HyDE] {query[:100]}... -> {rewritten[:150]}...')
            query = rewritten
    print(f'Query: {query[:200]}{"..." if len(query)>200 else ""}\n')

    r_bm25 = search_bm25(query, commands, args.top_k*3) if args.method in ('bm25','both') else []
    r_bge = search_bge(query, commands, args.top_k*3) if args.method in ('bge','both') else []

    if r_bm25 and r_bge:
        ranked = rrf_fusion(r_bm25, r_bge, top_k=args.top_k)
    elif r_bm25:
        ranked = [(r['command_id'], r['bm25']) for r in r_bm25[:args.top_k]]
    else:
        ranked = [(r['command_id'], r['bge']) for r in r_bge[:args.top_k]]

    print(f"{'Rank':<5} {'Command ID':<28} {'Device Type':<22} {'Score':>8}")
    print('-'*65)
    for i, (cid, score) in enumerate(ranked):
        dtype = next((c['设备类型'] for c in commands if c['设备指令号']==cid), '?')
        print(f'{i+1:<5} {cid:<28} {dtype:<22} {score:>8.4f}')

    if ranked:
        out = [{'rank':i+1,'command_id':cid,'device_type':next((c['设备类型'] for c in commands if c['设备指令号']==cid),'?'),'score':round(score,4)} for i,(cid,score) in enumerate(ranked)]
        print('\n' + json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
