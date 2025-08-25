# scripts/run_eval_verbose.py
from hybrid_search import hybrid_search, vector_search, keyword_search
from hybrid_index import connect, DEFAULT_DB

# Replace this mapping with your actual gold labels
gold = {
    "compression of vocabulary in computer languages": [1],
    "glottochronology protolanguage reconstruction": [2, 3],
    "language dynamics field": [4],
    "entropy of telugu": [5],
    "semantic parsing framework cornell spf": [6],
    "deep learning for nlp notes": [7],
    "meta learning for machine translation metamT": [8],
    "spoken italian treebank kiparla": [9],
    "cluster automata": [10],
    "conclusive remark on language modeling": [11],
    "sense of humour in computers": [12],
    "ukrainian writing system properties": [14],
    "richard berry paradox formal semantics": [15],
    "standardization of lexical information for nlp": [16],
}

def title(doc_id: int) -> str:
    conn = connect(DEFAULT_DB)
    cur = conn.cursor()
    cur.execute("SELECT title FROM documents WHERE doc_id=?", (doc_id,))
    t = (cur.fetchone() or ["?"])[0]
    conn.close()
    return t

def show(k: int = 3, fusion: str = "rrf"):
    for q, rel in gold.items():
        vec = vector_search(q, top_k=k)
        key = keyword_search(q, top_k=k)
        hyb = hybrid_search(q, k=k, fusion=fusion)

        print(f"\n=== Query: {q} ===")
        print("Relevant:", [(d, title(d)) for d in rel])
        print("Vector :", [(h.doc_id, title(h.doc_id)) for h in vec])
        print("Keyword:", [(h.doc_id, title(h.doc_id)) for h in key])
        print("Hybrid :", [(h['doc_id'], title(h['doc_id'])) for h in hyb["hybrid_hits"]])

if __name__ == "__main__":
    show(k=3, fusion="rrf")
