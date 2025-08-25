# hybrid_search.py
import sqlite3, numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from sklearn.preprocessing._data import minmax_scale
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rank_bm25 import BM25Okapi
from hybrid_index import connect, DEFAULT_DB, DEFAULT_FAISS, DEFAULT_FALLBACK_NN, VectorStore, TFIDFEmbedder
from utils_text import make_snippet

class Hit:
    def __init__(self, chunk_id, doc_id, score, rank, source, title, snippet):
        self.chunk_id = int(chunk_id)
        self.doc_id = int(doc_id)
        self.score = float(score)
        self.rank = int(rank)
        self.source = source
        self.title = title
        self.snippet = snippet

# --- Keyword search (FTS5 first; then BM25; finally TF-IDF) ---
def keyword_search(query: str, db_path: str = DEFAULT_DB, top_k: int = 10) -> List[Hit]:
    conn = connect(db_path)
    cur = conn.cursor()

    # Try FTS5
    try:
        cur.execute("""
          SELECT c.chunk_id, c.doc_id, d.title, c.content
          FROM doc_chunks_fts f
          JOIN chunks c ON c.chunk_id=f.rowid
          JOIN documents d ON d.doc_id=c.doc_id
          WHERE f.content MATCH ?
          LIMIT ?;
        """, (query, top_k))
        rows = cur.fetchall()
        conn.close()
        return [Hit(r[0], r[1], top_k - i, i+1, "keyword", r[2], make_snippet(r[3])) for i, r in enumerate(rows)]
    except Exception:
        pass

    # Fallback corpus (all chunks)
    cur.execute("""
        SELECT c.chunk_id, c.doc_id, d.title, c.content
        FROM chunks c JOIN documents d ON d.doc_id=c.doc_id
    """)
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return []

    texts = [r[3] for r in rows]

    # Try BM25
    try:
        tok = [t.split() for t in texts]
        bm25 = BM25Okapi(tok)
        scores = bm25.get_scores(query.split())
        order = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(rows[i][0], rows[i][1], float(scores[i]), rank+1, "keyword",
                rows[i][2], make_snippet(rows[i][3]))
            for rank, i in enumerate(order)
        ]
    except Exception:
        # Final fallback: TF-IDF cosine
        vect = TfidfVectorizer(max_features=5000)
        X = vect.fit_transform(texts)
        sims = cosine_similarity(vect.transform([query]), X)[0]
        order = sims.argsort()[::-1][:top_k]
        return [
            Hit(rows[i][0], rows[i][1], float(sims[i]), rank+1, "keyword",
                rows[i][2], make_snippet(rows[i][3]))
            for rank, i in enumerate(order)
        ]

def vector_search(query: str, top_k: int = 10, model: str = "tfidf",
                  db_path: str = DEFAULT_DB, vector_backend_path: Optional[str] = None) -> List[Hit]:
    # Always use TF-IDF embedder (loaded from storage/tfidf.pkl persisted at build time)
    emb = TFIDFEmbedder()
    qv = emb.transform([query])

    # pick backend (FAISS if available -> *.index file; else sklearn)
    path = vector_backend_path or DEFAULT_FAISS
    use_faiss = str(path).endswith(".index")
    vs = VectorStore(dim=qv.shape[1], use_faiss=use_faiss)
    if vector_backend_path:
        if use_faiss: vs.faiss_path = vector_backend_path
        else: vs.fallback_path = vector_backend_path
    vs.load()

    ids, sims = vs.search(qv, top_k=top_k)

    conn = connect(db_path); cur = conn.cursor()
    hits: List[Hit] = []
    for rank, (cid, sc) in enumerate(zip(ids.tolist(), sims.tolist()), start=1):
        cur.execute("""
          SELECT c.doc_id, d.title, c.content
          FROM chunks c JOIN documents d ON d.doc_id=c.doc_id
          WHERE c.chunk_id=?;
        """, (int(cid),))
        row = cur.fetchone()
        if row:
            hits.append(Hit(cid, row[0], float(sc), rank, "vector", row[1], make_snippet(row[2])))
    conn.close()
    return hits

def normalize_scores(hits: List[Hit]) -> None:
    if not hits: return
    scores = np.array([h.score for h in hits], dtype=float)
    if np.all(scores == scores[0]):
        for h in hits: h.score = 1.0
        return
    scaled = minmax_scale(scores)
    for h, s in zip(hits, scaled): h.score = float(s)

def weighted_sum_fusion(vec_hits: List[Hit], key_hits: List[Hit], alpha: float = 0.6, top_k: int = 10) -> List[Hit]:
    normalize_scores(vec_hits); normalize_scores(key_hits)
    by_id: Dict[int, Hit] = {}
    for h in vec_hits:
        by_id[h.chunk_id] = Hit(**h.__dict__)
    for h in key_hits:
        if h.chunk_id in by_id:
            by_id[h.chunk_id].score = alpha * by_id[h.chunk_id].score + (1 - alpha) * h.score
            by_id[h.chunk_id].source = "hybrid"
        else:
            copy = Hit(**h.__dict__)
            copy.score *= (1 - alpha)
            copy.source = "hybrid"
            by_id[h.chunk_id] = copy
    out = list(by_id.values())
    out.sort(key=lambda x: x.score, reverse=True)
    for i,h in enumerate(out, start=1): h.rank = i
    return out[:top_k]

def rrf_fusion(vec_hits: List[Hit], key_hits: List[Hit], k: int = 60, top_k: int = 10) -> List[Hit]:
    scores, meta = defaultdict(float), {}
    for hits in (vec_hits, key_hits):
        for h in hits:
            scores[h.chunk_id] += 1.0 / (k + h.rank)
            if h.chunk_id not in meta: meta[h.chunk_id] = h
    out = []
    for cid, sc in scores.items():
        m = meta[cid]
        out.append(Hit(cid, m.doc_id, float(sc), 0, "hybrid", m.title, m.snippet))
    out.sort(key=lambda x: x.score, reverse=True)
    for i,h in enumerate(out, start=1): h.rank=i
    return out[:top_k]

def hybrid_search(query: str, k: int = 3, fusion: str = "rrf", alpha: float = 0.6,
                  db_path: str = DEFAULT_DB, vector_backend_path: Optional[str] = None):
    vec = vector_search(query, top_k=max(10,k), db_path=db_path, vector_backend_path=vector_backend_path)
    key = keyword_search(query, db_path=db_path, top_k=max(10,k))
    fused = rrf_fusion(vec, key, top_k=k) if fusion == "rrf" else weighted_sum_fusion(vec, key, alpha=alpha, top_k=k)

    def pack(h: Hit): 
        return {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title,
                "score": float(round(h.score, 6)), "rank": h.rank, "source": h.source, "snippet": h.snippet}

    return {
        "vector_hits": [pack(h) for h in vec[:k]],
        "keyword_hits": [pack(h) for h in key[:k]],
        "hybrid_hits": [pack(h) for h in fused]
    }
