# hybrid_index.py (TF-IDF embedding backend; FAISS still supported)

from __future__ import annotations
import os, sqlite3, pathlib
from typing import List, Dict, Tuple
import numpy as np

try:
    import faiss  # optional; we’ll still use FAISS if available
except Exception:
    faiss = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from joblib import dump, load

STORAGE_DIR = pathlib.Path("storage")
STORAGE_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_DB = str(STORAGE_DIR / "index.sqlite")
DEFAULT_FAISS = str(STORAGE_DIR / "faiss.index")
DEFAULT_FALLBACK_NN = str(STORAGE_DIR / "fallback_nn.pkl")
TFIDF_PATH = str(STORAGE_DIR / "tfidf.pkl")

def connect(db_path=DEFAULT_DB):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_sqlite(db_path=DEFAULT_DB):
    conn = connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents(
            doc_id   INTEGER PRIMARY KEY,
            title    TEXT,
            author   TEXT,
            year     INTEGER,
            keywords TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            chunk_id INTEGER PRIMARY KEY,
            doc_id   INTEGER,
            ordinal  INTEGER,
            content  TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        );
    """)
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
            content, tokenize='porter'
        );
    """)
    conn.commit(); conn.close()

def insert_document(conn, title, author, year, keywords) -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO documents(title,author,year,keywords) VALUES(?,?,?,?)",
                (title, author, int(year) if year else 0, keywords))
    return cur.lastrowid

def insert_chunk(conn, doc_id: int, ordinal: int, content: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO chunks(doc_id,ordinal,content) VALUES(?,?,?)",
                (doc_id, ordinal, content))
    cid = cur.lastrowid
    cur.execute("INSERT INTO doc_chunks_fts(rowid,content) VALUES(?,?)", (cid, content))
    return cid

class TFIDFEmbedder:
    """Always TF-IDF. Fits on corpus at build time, persists to storage/tfidf.pkl, loads for queries."""
    def __init__(self, max_features: int = 4096):
        self.max_features = max_features
        self.vect: TfidfVectorizer | None = None

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.vect = TfidfVectorizer(max_features=self.max_features)
        X = self.vect.fit_transform(texts).astype("float32")
        dump(self.vect, TFIDF_PATH)  # persist vocab for query-time transform
        return normalize(X).toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.vect is None:
            self.vect = load(TFIDF_PATH)
        X = self.vect.transform(texts).astype("float32")
        return normalize(X).toarray()

class VectorStore:
    def __init__(self, dim: int, use_faiss: bool = True,
                 faiss_path: str = DEFAULT_FAISS,
                 fallback_path: str = DEFAULT_FALLBACK_NN):
        self.dim = dim
        self.use_faiss = bool(use_faiss and (faiss is not None))
        self.faiss_path = faiss_path
        self.fallback_path = fallback_path
        if self.use_faiss:
            self.idmap = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        else:
            self.nn = NearestNeighbors(n_neighbors=50, metric="cosine")
            self._X = None
            self._ids = None

    def build(self, X: np.ndarray, ids: np.ndarray):
        if self.use_faiss:
            self.idmap.add_with_ids(X, ids.astype(np.int64))
        else:
            self.nn.fit(X); self._X = X; self._ids = ids

    def save(self):
        if self.use_faiss:
            faiss.write_index(self.idmap, self.faiss_path)
        else:
            dump({"nn": self.nn, "X": self._X, "ids": self._ids}, self.fallback_path)

    def load(self):
        if self.use_faiss:
            self.idmap = faiss.read_index(self.faiss_path)
            self.dim = self.idmap.d
        else:
            obj = load(self.fallback_path)
            self.nn, self._X, self._ids = obj["nn"], obj["X"], obj["ids"]
            self.dim = self._X.shape[1]

    def search(self, q: np.ndarray, top_k=10):
        if self.use_faiss:
            D, I = self.idmap.search(q, top_k)
            S = np.clip(D, 0.0, 1.0)   # cosine-ish since vectors normalized
            return I[0], S[0]
        dist, idx = self.nn.kneighbors(q, n_neighbors=top_k, return_distance=True)
        sim = 1.0 - dist[0]
        ids = self._ids[idx[0]]
        return ids, sim

def build_index_from_docs(
    docs: List[Dict],
    db_path: str = DEFAULT_DB,
    use_faiss: bool = True,
    max_features: int = 4096,
) -> Tuple[str, str]:
    from utils_text import greedy_chunk

    init_sqlite(db_path)
    conn = connect(db_path)
    embedder = TFIDFEmbedder(max_features=max_features)

    all_chunks, all_ids = [], []
    for d in docs:
        doc_id = insert_document(conn, d.get("title","Untitled"), d.get("author",""),
                                 d.get("year", 0), d.get("keywords",""))
        text = (d.get("text","") or "").strip()
        paras = [p for p in text.split("\n") if p.strip()]
        chunks = greedy_chunk(paras, max_chars=900, overlap=120)
        for i, ch in enumerate(chunks):
            cid = insert_chunk(conn, doc_id, i, ch)
            all_chunks.append(ch)
            all_ids.append(cid)
    conn.commit(); conn.close()

    if not all_chunks:
        raise ValueError("No chunks ingested.")

    X = embedder.fit_transform(all_chunks)  # normalized TF-IDF vectors
    ids = np.array(all_ids, dtype=np.int64)

    vstore = VectorStore(dim=X.shape[1], use_faiss=use_faiss)
    vstore.build(X, ids)
    vstore.save()

    return db_path, (DEFAULT_FAISS if vstore.use_faiss else DEFAULT_FALLBACK_NN)
