import json
import glob
import chromadb

from config import EMBEDDINGS_DIR, CHROMA_DIR

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name="postmortems",
    metadata={"hnsw:space": "cosine"}
)

if __name__ == "__main__":
    for path in glob.glob(str(EMBEDDINGS_DIR / "*.json")):
        records = json.load(open(path))
        ids = [r["id"] for r in records]

        existing = set(collection.get(ids=ids)["ids"])
        new_records = [r for r in records if r["id"] not in existing]
        if not new_records:
            continue

        collection.add(
            ids=[r["id"] for r in new_records],
            embeddings=[r["embedding"] for r in new_records],
            documents=[r["text"] for r in new_records],
            metadatas=[{"source_file": r["source_file"]} for r in new_records],
        )
