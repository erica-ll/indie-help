import re
import chromadb
import yaml
from openai import OpenAI

from config import CHROMA_DIR, TESTS_DIR, PROMPTS_PATH

ACTIVE_PROMPT_VERSION = "system_prompt_v1"

client = OpenAI()
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection("postmortems")

with open(PROMPTS_PATH) as f:
    PROMPTS = yaml.safe_load(f)


def ask(question, top_k=3):
    query_embedding = client.embeddings.create(
        input=[question], model="text-embedding-3-small"
    ).data[0].embedding

    # Query ChromaDB for the Top-K most similar text chunks
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    context_blocks = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_blocks.append(f"[Source: {meta['source_file']}]\n{text}")
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = PROMPTS[ACTIVE_PROMPT_VERSION]
    user_prompt = f"Retrieved Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def parse_prompts(text):
    entries = re.split(r"\n(?=\d+\.\s)", text.strip())
    prompts = {}
    for entry in entries:
        match = re.match(r"(\d+)\.\s*(.*)", entry, re.DOTALL)
        if match:
            idx, question = match.groups()
            prompts[int(idx)] = question.strip()
    return prompts


def run_tests():
    prompts_path = TESTS_DIR / "test_prompts.txt"
    answers_path = TESTS_DIR / "answers.txt"

    prompts = parse_prompts(prompts_path.read_text())

    blocks = []
    for idx in sorted(prompts):
        question = prompts[idx]
        print(f"[{idx}] {question}")
        answer = ask(question)
        blocks.append(
            f"===== Q{idx} =====\n"
            f"Question: {question}\n"
            f"Answer:\n{answer}"
        )

    answers_path.write_text("\n\n".join(blocks) + "\n")
    print(f"\nWrote {len(blocks)} answers to {answers_path}")


if __name__ == "__main__":
    run_tests()
