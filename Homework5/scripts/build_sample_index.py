# scripts/build_sample_index.py

from hybrid_index import init_sqlite, DEFAULT_DB, Embedder
from utils_text import greedy_chunk
import sqlite3
import numpy as np
import os

# Define a few small documents
SAMPLES = [
    {
        "title": "Introduction to Neural Networks",
        "author": "A. Smith",
        "year": 2021,
        "keywords": "neural networks, deep learning",
        "text": """Neural networks are powerful function approximators.
        Backpropagation is used to train deep models.
        Activation functions like ReLU introduce nonlinearity.
        Applications include computer vision and NLP."""
    },
    {
        "title": "Vector Databases and FAISS",
        "author": "B. Chen",
        "year": 2023,
        "keywords": "FAISS, vector search, embeddings",
        "text": """Vector databases store dense embeddings of text or images.
        FAISS provides efficient similarity search.
        Cosine similarity and inner product are common metrics."""
    },
    {
        "title": "SQLite FTS5 and BM25",
        "author": "C. Kim",
        "year": 2022,
        "keywords": "SQLite, FTS5, BM25, keyword search",
        "text": """SQLite FTS5 supports full-text search.
        BM25 is a ranking function used in search engines.
        Keyword search excels at exact term matches."""
    }
]

def build_index(docs=SAMPLES, db_path=DEFAULT_DB):
    init_sqlite(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    embedder = Embedder()

    all_chunks, all_ids = [], []

    for doc in docs:
        cur.execute("INSERT INTO documents(title,author,year,keywords) VALUES(?,?,?,?)",
                    (doc["title"], doc["author"], doc["year"], doc["keywords"]))
        doc_id = cur.lastrowid
        chunks = greedy_chunk(doc["text"].split("\\n"))
        for i, ch in enumerate(chunks):
            cur.execute("INSERT INTO chunks(doc_id,ordinal,content) VALUES(?,?,?)",
                        (doc_id, i, ch))
            cid = cur.lastrowid
            cur.execute("INSERT INTO doc_chunks_fts(rowid,content) VALUES(?,?)", (cid, ch))
            all_chunks.append(ch)
            all_ids.append(cid)

    conn.commit(); conn.close()

    # Build embeddings (fallbacks inside Embedder)
    X = embedder.fit_transform(all_chunks)
    np.save(os.path.join(os.path.dirname(db_path), "embeddings.npy"), X)
    np.save(os.path.join(os.path.dirname(db_path), "chunk_ids.npy"), np.array(all_ids))

    print(f"Indexed {len(all_chunks)} chunks from {len(docs)} documents into {db_path}")

if __name__ == "__main__":
    build_index()
