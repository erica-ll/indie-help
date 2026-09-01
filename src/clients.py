"""Shared, expensive-to-construct clients and static config used across all
pipeline components. Import from here rather than re-instantiating -- loading
the BM25 index and connecting to Chroma/Cohere/OpenAI are not cheap, and each
should happen exactly once per process."""
import os
import chromadb
import yaml
import bm25s
import cohere
from openai import OpenAI

from config import CHROMA_DIR, PROMPTS_PATH, BM25_DIR

client = OpenAI()
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection("postmortems")
bm25_retriever = bm25s.BM25.load(str(BM25_DIR), load_corpus=True)
cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
with open(PROMPTS_PATH) as f:
    PROMPTS = yaml.safe_load(f)
