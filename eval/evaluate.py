import json, re
from rag import load_retriever

def hit(gold, docs):
    gold_l = gold.lower()
    return any(gold_l[:40] in d.page_content.lower() for d in docs)  # naive

if __name__ == "__main__":
    retriever = load_retriever()
    hits = total = 0
    with open("eval/qa_seed.jsonl") as f:
        for line in f:
            item = json.loads(line)
            q = item.get("question") or item.get("q")
            a = item.get("answer") or item.get("a")
            docs = retriever.get_relevant_documents(q)
            total += 1
            hits += 1 if hit(a, docs) else 0
    print(f"Retrieval hit-rate: {hits}/{total} = {hits/total:.2f}")