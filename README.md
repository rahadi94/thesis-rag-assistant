Thesis RAG Assistant
====================

Lightweight RAG pipeline for your PDFs using LangChain, FAISS, Sentence-Transformers, and Ollama for local LLM generation. Comes with a Streamlit UI and a tiny retrieval eval harness.

Project layout
--------------

```
thesis-rag-assistant/
  ├─ data/                  # PDFs to ingest
  ├─ storage/               # FAISS index + metadata
  ├─ src/
  │   ├─ ingest.py          # load → chunk → embed → index
  │   ├─ rag.py             # retrieval + generation (Ollama)
  │   └─ app.py             # Streamlit UI
  ├─ eval/
  │   ├─ qa_seed.jsonl      # seed Q/A pairs
  │   └─ evaluate.py        # simple retrieval hit-rate eval
  ├─ .env.example
  ├─ requirements.txt
  └─ README.md
```

Prerequisites
-------------
- Python 3.10+
- Ollama installed and a model available (default: `llama3`)

Install Ollama
--------------

```bash
curl -fsSL https://ollama.com/install.sh | sh
# Start service or run manually
sudo systemctl enable --now ollama || true
# or
ollama serve

ollama pull llama3
ollama run llama3 "Hello"
```

Setup
-----

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# optional: edit .env to change OLLAMA_MODEL or paths
```

Ingest PDFs → FAISS
-------------------
Put your PDFs under `data/`.

```bash
python src/ingest.py
```

Defaults:
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Chunking: 2000 chars, 300 overlap
- Vector store: FAISS at `storage/`

Run Streamlit UI
----------------

```bash
streamlit run src/app.py
```

Ask a question, view the generated answer and the retrieved context with sources `[source: page]`.

Programmatic usage
------------------

```python
from rag import answer

text, docs = answer("What is in these documents?")
print(text)
```

Evaluation
----------

Seed questions live in `eval/qa_seed.jsonl` as JSONL with keys `question` and `answer`.

```bash
python eval/evaluate.py
# Example output: Retrieval hit-rate: 9/12 = 0.75
```

Configuration
-------------
Set via `.env`:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
DATA_DIR=data
STORE_DIR=storage
```

Notes
-----
- If Ollama’s systemd unit isn’t present, run `ollama serve` in a separate terminal.
- For different models: `ollama pull mistral`, then set `OLLAMA_MODEL=mistral`.

