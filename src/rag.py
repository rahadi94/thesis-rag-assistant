import os
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# --- Embeddings wrapper (same as in ingest) ---
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
class SBERT(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(EMB_MODEL)
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

def load_retriever(store_dir="storage"):
    vs = FAISS.load_local(store_dir, SBERT(), allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    return retriever

SYSTEM_PROMPT = """You answer with grounded facts from the provided CONTEXT.
- If you don't know, say so.
- Cite sources in [source: page] style for each claim."""

USER_PROMPT_TMPL = PromptTemplate.from_template(
    "QUESTION: {question}\n\nCONTEXT:\n{context}\n\nAnswer:"
)

def format_context(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "N/A")
        lines.append(f"[{src}: {page}] {d.page_content[:1200]}")
    return "\n---\n".join(lines)

# Plug your LLM of choice here (example: OpenAI)
def llm_call(prompt: str) -> str:
    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = Ollama(model=model_name, base_url=base_url)
    return llm.invoke(prompt).strip()

def answer(question: str):
    retriever = load_retriever()
    docs = retriever.get_relevant_documents(question)
    ctx = format_context(docs)
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TMPL.format(question=question, context=ctx)}"
    out = llm_call(prompt)
    return out, docs