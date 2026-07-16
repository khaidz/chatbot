"""BM25 Okapi tự viết (cho numpy backend) — không phụ thuộc lib ngoài."""
import math
from collections import Counter


class BM25Okapi:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_freqs = [Counter(toks) for toks in corpus_tokens]
        self.doc_lens = [len(toks) for toks in corpus_tokens]
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0
        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        # idf theo công thức Okapi có +1 để không âm
        self.idf = {
            term: math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0)
            for term, n in df.items()
        }

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        freqs = self.doc_freqs[doc_index]
        dl = self.doc_lens[doc_index] or 1
        s = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            f = freqs[term]
            idf = self.idf.get(term, 0.0)
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def top_k(self, query_tokens: list[str], k: int, mask: list[bool] | None = None):
        """Trả về list (doc_index, score) giảm dần, chỉ những doc có mask=True."""
        scored = []
        for i in range(self.n_docs):
            if mask is not None and not mask[i]:
                continue
            s = self.score(query_tokens, i)
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
