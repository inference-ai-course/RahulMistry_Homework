# scripts/inspect_index.py
import sqlite3
from hybrid_index import DEFAULT_DB

def main():
    conn = sqlite3.connect(DEFAULT_DB)
    cur = conn.cursor()
    cur.execute("SELECT doc_id, year, title FROM documents ORDER BY doc_id;")
    rows = cur.fetchall()
    for doc_id, year, title in rows:
        print(f"{doc_id}\t{year}\t{title[:120]}")
    conn.close()

if __name__ == "__main__":
    main()
