#!/usr/bin/env python3
"""
bm25_index.py - BM25 粗筛: 快速关键词匹配，从设备语料中召回候选

原理:
  BM25 是 TF-IDF 的改进版，考虑词频饱和度和文档长度归一化。
  对中文文本，先用 jieba 分词，再建倒排索引。

用法:
  python bm25_index.py --corpus device_corpus.jsonl --build
  python bm25_index.py --corpus device_corpus.jsonl --query "主飞控地面维护" --top_k 50
"""
import json, argparse, os, pickle, re
from collections import defaultdict
import math

# ============================================================
# BM25 实现 (from scratch, no external deps beyond jieba)
# ============================================================

class BM25:
    """BM25 检索器，支持中文分词"""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # term frequency saturation
        self.b = b    # length normalization
        self.corpus = []
        self.doc_len = []
        self.avgdl = 0
        self.idf = {}
        self.doc_freqs = defaultdict(int)
        self.inverted_index = defaultdict(list)  # term -> [(doc_id, tf), ...]
        self.N = 0
    
    def tokenize(self, text):
        """中文分词 + 英文小写 + 去停用词"""
        try:
            import jieba
            tokens = list(jieba.cut(text))
        except ImportError:
            # Fallback: character-level for Chinese + whitespace for English
            tokens = []
            for ch in text:
                if ch.isalnum() or '\u4e00' <= ch <= '\u9fff':
                    tokens.append(ch)
        
        # Filter: keep meaningful tokens
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
                     '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
                     '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
                     'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during'}
        return [t.lower().strip() for t in tokens if t.strip() and t.lower().strip() not in stopwords]
    
    def build_doc_text(self, doc):
        """Build searchable text from device doc fields"""
        parts = []
        for f in ['设备类型', '设备单元号', '设备指令号', '设备参数',
                   '操作对象', '操作内容', '说明', '判据描述']:
            v = doc.get(f, '')
            if v and v not in ('[]', ''):
                parts.append(v)
        return ' '.join(parts)
    
    def fit(self, corpus):
        """Build BM25 index from corpus"""
        self.corpus = corpus
        self.N = len(corpus)
        
        # Tokenize all docs
        tokenized = []
        for doc in corpus:
            text = self.build_doc_text(doc)
            tokens = self.tokenize(text)
            tokenized.append(tokens)
            self.doc_len.append(len(tokens))
        
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        
        # Build inverted index and doc frequencies
        for doc_id, tokens in enumerate(tokenized):
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for t, freq in tf.items():
                self.inverted_index[t].append((doc_id, freq))
                self.doc_freqs[t] += 1
        
        # Compute IDF
        for t, df in self.doc_freqs.items():
            self.idf[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        
        print(f'BM25 index built: {self.N} docs, {len(self.idf)} unique terms, avgdl={self.avgdl:.1f}')
    
    def search(self, query, top_k=50):
        """Search and return top_k results with scores"""
        query_tokens = self.tokenize(query)
        scores = defaultdict(float)
        
        for t in query_tokens:
            if t not in self.inverted_index:
                continue
            idf = self.idf.get(t, 0)
            for doc_id, tf in self.inverted_index[t]:
                dl = self.doc_len[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * numerator / denominator
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(self.corpus[doc_id], score) for doc_id, score in ranked]
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'corpus': self.corpus, 'doc_len': self.doc_len, 'avgdl': self.avgdl,
                'idf': self.idf, 'doc_freqs': dict(self.doc_freqs),
                'inverted_index': dict(self.inverted_index), 'N': self.N,
                'k1': self.k1, 'b': self.b
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.corpus = data['corpus']; self.doc_len = data['doc_len']
        self.avgdl = data['avgdl']; self.idf = data['idf']
        self.doc_freqs = defaultdict(int, data['doc_freqs'])
        self.inverted_index = defaultdict(list, data['inverted_index'])
        self.N = data['N']; self.k1 = data['k1']; self.b = data['b']
        print(f'BM25 index loaded: {self.N} docs')

# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default='device_corpus.jsonl')
    p.add_argument('--index', default='bm25_index.pkl')
    p.add_argument('--build', action='store_true')
    p.add_argument('--query', default=None)
    p.add_argument('--top_k', type=int, default=50)
    args = p.parse_args()
    
    bm25 = BM25()
    
    if args.build:
        corpus = []
        with open(args.corpus, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
        bm25.fit(corpus)
        bm25.save(args.index)
        print(f'Saved: {args.index}')
    
    if args.query:
        if os.path.exists(args.index):
            bm25.load(args.index)
        else:
            print('Index not found, building first...')
            corpus = []
            with open(args.corpus, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        corpus.append(json.loads(line))
            bm25.fit(corpus)
        
        results = bm25.search(args.query, args.top_k)
        print(f'\nQuery: {args.query}')
        print(f'Results: {len(results)}')
        for i, (doc, score) in enumerate(results[:10]):
            dev_type = doc.get('设备类型', 'N/A')
            dev_unit = doc.get('设备单元号', 'N/A')
            dev_cmd = doc.get('设备指令号', 'N/A')
            print(f'  [{i+1}] score={score:.3f} | {dev_type} | {dev_unit} | {dev_cmd[:60]}')

if __name__ == '__main__':
    main()
