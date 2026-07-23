#!/usr/bin/env python3
"""
bge_reranker.py - BGE 向量精排: 对 BM25 粗筛结果做语义重排序

原理:
  BGE (BAAI General Embedding) 是智源研究院的中英双语向量模型。
  将查询和文档分别编码为向量，用余弦相似度排序。
  优先使用 bge-large-zh-v1.5 (1024维), 兼顾精度和速度。

用法:
  python bge_reranker.py --query "主飞控地面维护设备" --candidates candidates.jsonl
"""
import json, argparse, os, pickle
import numpy as np

class BGEReranker:
    """BGE 向量重排序器"""
    
    def __init__(self, model_name='BAAI/bge-large-zh-v1.5', cache_path='bge_cache.pkl'):
        self.model_name = model_name
        self.cache_path = cache_path
        self.model = None
        self.cache = {}  # text -> vector
        self._load_cache()
    
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
    
    def _save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f'BGE model loaded: {self.model_name}')
    
    def encode(self, texts, batch_size=32):
        """Encode texts to vectors"""
        self._load_model()
        
        # Check cache
        uncached = []
        uncached_idx = []
        for i, t in enumerate(texts):
            if t not in self.cache:
                uncached.append(t)
                uncached_idx.append(i)
        
        if uncached:
            vectors = self.model.encode(uncached, batch_size=batch_size, normalize_embeddings=True)
            for idx, vec in zip(uncached_idx, vectors):
                self.cache[texts[idx]] = vec
            self._save_cache()
        
        return np.array([self.cache[t] for t in texts])
    
    def build_doc_text(self, doc):
        """Build semantic text from device fields for embedding"""
        parts = []
        for f in ['设备类型', '设备单元号', '操作对象', '操作内容', '说明']:
            v = doc.get(f, '')
            if v and v not in ('[]', ''):
                parts.append(v)
        return ' | '.join(parts) if parts else ' '
    
    def rerank(self, query, candidates, top_k=20):
        """Re-rank BM25 candidates with BGE cosine similarity"""
        if not candidates:
            return []
        
        doc_texts = [self.build_doc_text(doc) for doc in candidates]
        all_texts = [query] + doc_texts
        vectors = self.encode(all_texts)
        
        query_vec = vectors[0]
        doc_vecs = vectors[1:]
        
        # Cosine similarity (vectors already normalized)
        scores = np.dot(doc_vecs, query_vec)
        
        ranked_idx = np.argsort(-scores)[:top_k]
        return [(candidates[i], float(scores[i])) for i in ranked_idx]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', required=True)
    p.add_argument('--candidates', help='JSONL file of BM25 candidates')
    p.add_argument('--candidates_json', help='JSON file of BM25 candidates')
    p.add_argument('--top_k', type=int, default=20)
    p.add_argument('--model', default='BAAI/bge-large-zh-v1.5')
    args = p.parse_args()
    
    # Load candidates
    candidates = []
    if args.candidates:
        with open(args.candidates, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: candidates.append(json.loads(line))
                    except: pass
    elif args.candidates_json:
        with open(args.candidates_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                candidates = [item['doc'] if isinstance(item, dict) else item for item in data]
    
    if not candidates:
        print('No candidates provided')
        return
    
    reranker = BGEReranker(model_name=args.model)
    results = reranker.rerank(args.query, candidates, args.top_k)
    
    print(f'\nQuery: {args.query}')
    print(f'BGE re-ranked: {len(results)} results')
    for i, (doc, score) in enumerate(results[:10]):
        print(f'  [{i+1}] sim={score:.4f} | {doc.get("设备类型","")} | {doc.get("操作对象","")[:50]}')
    
    # Return top results as JSON for pipeline
    output = [{'rank': i+1, 'score': s, '设备类型': d.get('设备类型',''),
               '设备单元号': d.get('设备单元号',''), '设备指令号': d.get('设备指令号',''),
               '设备参数': d.get('设备参数',''), '说明': d.get('说明','')[:100]}
              for i, (d, s) in enumerate(results)]
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
