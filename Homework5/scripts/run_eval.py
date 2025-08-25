# scripts/run_eval.py
from typing import Dict, List
import os
from hybrid_search import hybrid_search, vector_search, keyword_search
from hybrid_index import DEFAULT_DB
from scripts.build_from_arxiv import download_arxiv_papers, build_docs_from_pdfs
from hybrid_index import build_index_from_docs

def ensure_index():
    # Build index from small arXiv sample if DB missing
    if not os.path.exists(DEFAULT_DB):
        meta = download_arxiv_papers(query="cs.CL", max_results=15)
        docs = build_docs_from_pdfs(meta)
        build_index_from_docs(docs)

def recall_at_k(run_hits: Dict[str, List[int]], gold: Dict[str, List[int]], k: int) -> float:
    total = len(gold)
    hits = sum(1 for q, rel in gold.items() if any(d in rel for d in run_hits.get(q, [])[:k]))
    return hits / max(1, total)


def main(k: int = 3, fusion: str = "rrf"):
    ensure_index()
    # Toy “gold” set (adjust if you want to hand-label on your corpus)
    gold = {
        # modern NLP / modeling
        "semantic parsing framework cornell spf": [6],
        "deep learning for nlp notes": [7],
        "meta learning for machine translation metamT": [8],
        "spoken italian treebank kiparla": [9],
        "cluster automata": [10],
        "conclusive remark on language modeling": [11],

        # classics / linguistics
        "compression of vocabulary in computer languages": [1],
        "glottochronology protolanguage reconstruction": [2, 3],
        "language dynamics field": [4],
        "entropy of telugu": [5],
        "sense of humour in computers": [12],
        "ukrainian writing system properties": [14],
        "richard berry paradox formal semantics": [15],
        "standardization of lexical information for nlp": [16],
    }

    vec_run, key_run, hyb_run = {}, {}, {}
    for q in gold.keys():
        vec = vector_search(q, top_k=k)
        key = keyword_search(q, top_k=k)
        hyb = hybrid_search(q, k=k, fusion=fusion)

        vec_run[q] = [h.doc_id for h in vec]
        key_run[q] = [h.doc_id for h in key]
        hyb_run[q] = [h["doc_id"] for h in hyb["hybrid_hits"]]

    # If you manually label `gold` with relevant doc_ids, these numbers will be meaningful.
    print(f"Recall@{k} (vector) :", recall_at_k(vec_run, gold, k))
    print(f"Recall@{k} (keyword):", recall_at_k(key_run, gold, k))
    print(f"Recall@{k} (hybrid {fusion}):", recall_at_k(hyb_run, gold, k))



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--fusion", type=str, default="rrf", choices=["rrf","weighted"])
    args = ap.parse_args()
    main(args.k, args.fusion)
