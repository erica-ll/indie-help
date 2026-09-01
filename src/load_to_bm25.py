"""
Keyword-based BM25 retrieval implementation.
"""
import json
import bm25s
from config import EMBEDDINGS_DIR, BM25_DIR

corpus = []
for path in EMBEDDINGS_DIR.glob("*.json"):
    records = json.load(open(path))
    corpus.extend(
        {"id": r["id"], "source_file": r["source_file"], "text": r["text"]}
        for r in records
    )

text_chunks = [record["text"] for record in corpus]

# bm-25
tokens_bm25 = bm25s.tokenize(text_chunks, stopwords="en")
retriever = bm25s.BM25(corpus=corpus)
retriever.index(tokens_bm25)

BM25_DIR.mkdir(parents=True, exist_ok=True)
retriever.save(str(BM25_DIR), corpus=corpus)
