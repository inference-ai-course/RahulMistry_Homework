# scripts/build_from_arxiv.py
import os, json, time, re, requests, feedparser
import fitz  # PyMuPDF
from typing import List, Dict
from hybrid_index import build_index_from_docs

PAPERS_DIR = "papers"
META_JSON  = os.path.join(PAPERS_DIR, "metadata.json")

UA = {"User-Agent": "Week5-Hybrid-Retrieval/1.0 (mailto:you@example.com)"}  # <- put your email if you want
TIMEOUT = 60

def _robust_pdf_urls(arxiv_id: str) -> list[str]:
    """
    Given something like '2011.10361v1' produce a list of candidate PDF URLs
    (versioned first, then base id, and use both domains).
    """
    base = re.sub(r"v\d+$", "", arxiv_id)
    candidates = [
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://arxiv.org/pdf/{base}.pdf",
        f"http://export.arxiv.org/pdf/{arxiv_id}.pdf",
        f"http://export.arxiv.org/pdf/{base}.pdf",
    ]
    return candidates

def _try_download(url: str, dest: str) -> bool:
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and r.content.startswith(b"%PDF"):
            with open(dest, "wb") as f:
                f.write(r.content)
            return True
        # treat 403/404 gracefully; continue trying fallbacks
        return False
    except requests.RequestException:
        return False

def download_arxiv_papers(query="cs.CL", max_results=20):
    os.makedirs(PAPERS_DIR, exist_ok=True)
    base_url = "http://export.arxiv.org/api/query"
    params = {"search_query": query, "start": 0, "max_results": max_results}
    r = requests.get(base_url, params=params, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    feed = feedparser.parse(r.text)

    meta = []
    for entry in feed.entries:
        arx_id_full = entry.id.split("/")[-1]  # e.g., 2011.10361v1
        pdf_path = os.path.join(PAPERS_DIR, arx_id_full + ".pdf")
        if not os.path.exists(pdf_path):
            ok = False
            for url in _robust_pdf_urls(arx_id_full):
                if _try_download(url, pdf_path):
                    print("Downloaded", os.path.basename(pdf_path))
                    ok = True
                    break
            # back off slightly to be polite
            time.sleep(0.5)
            if not ok:
                print("Skip (no PDF):", arx_id_full)
                continue

        authors = ", ".join([a.name for a in getattr(entry, "authors", [])]) if hasattr(entry, "authors") else ""
        year = int(getattr(entry, "published", "0000")[:4]) if hasattr(entry, "published") else 0
        meta.append({
            "id": arx_id_full,
            "title": entry.title,
            "author": authors,
            "year": year,
            "keywords": f"arxiv,{query}",
            "pdf_path": pdf_path
        })

    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    return meta

def extract_text_from_pdf(pdf_path: str) -> str:
    parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)

def build_docs_from_pdfs(meta_list: List[Dict]) -> List[Dict]:
    docs = []
    for m in meta_list:
        if not os.path.exists(m["pdf_path"]):
            continue
        text = extract_text_from_pdf(m["pdf_path"])
        docs.append({
            "title": m["title"],
            "author": m["author"],
            "year": m["year"],
            "keywords": m["keywords"],
            "text": text
        })
    return docs

if __name__ == "__main__":
    meta = download_arxiv_papers(query="cs.CL", max_results=20)
    if not meta:
        print("No PDFs downloaded; nothing to index.")
    else:
        docs = build_docs_from_pdfs(meta)
        if not docs:
            print("No text extracted; nothing to index.")
        else:
            db, vec = build_index_from_docs(docs, use_faiss=True)
            print("SQLite:", db)
            print("Vector store:", vec)
