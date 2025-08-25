# api.py
from fastapi import FastAPI, Query
from hybrid_search import hybrid_search

app = FastAPI(title="Hybrid Retrieval API")

@app.get("/hybrid_search")
def route_hybrid_search(
    query: str = Query(...),
    k: int = Query(3, ge=1, le=50),
    fusion: str = Query("rrf", regex="^(rrf|weighted)$"),
    alpha: float = Query(0.6, ge=0.0, le=1.0)
):
    return hybrid_search(query=query, k=k, fusion=fusion, alpha=alpha)
