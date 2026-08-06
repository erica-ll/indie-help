import json
import fitz
import tiktoken
from openai import OpenAI

from config import RAW_DIR, EMBEDDINGS_DIR

client = OpenAI() 
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def extract_text(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text, chunk_size=650, overlap=100):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start = end - overlap
    return chunks


def embed_chunks(chunks, model="text-embedding-3-small", batch_size=100):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        embeddings.extend([d.embedding for d in resp.data])
    return embeddings


if __name__ == "__main__":
    for file_path in RAW_DIR.iterdir():
        if file_path.suffix.lower() != ".pdf":
            continue

        out_path = EMBEDDINGS_DIR / f"{file_path.stem}.json"
        if out_path.exists():
            continue  # 已经处理过，跳过，省 API 调用

        input_text = extract_text(file_path)
        chunks = chunk_text(input_text)
        embeddings = embed_chunks(chunks)

        records = [
            {
                "id": f"{file_path.stem}_chunk_{i:03d}",
                "source_file": file_path.name,
                "text": chunk,
                "embedding": embedding,
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        with open(out_path, "w") as f:
            json.dump(records, f)
