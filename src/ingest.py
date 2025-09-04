import os, json, glob
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
STORE_DIR = "storage"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_docs():
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        loader = PyPDFLoader(path)
        for d in loader.load():
            d.metadata["source"] = os.path.basename(path)
            docs.append(d)
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    return splitter.split_documents(docs)

def build_index(chunks):
    model = SentenceTransformer(EMB_MODEL)
    def embed(texts): return model.encode(texts, normalize_embeddings=True).tolist()
    # Wrap as LangChain Embeddings-like
    from langchain.embeddings.base import Embpeddings
    class SBERT(Embeddings):
        def embed_documents(self, texts): return embed(texts)
        def embed_query(self, text): return embed([text])[0]
    return FAISS.from_documents(chunks, SBERT())

if __name__ == "__main__":
    os.makedirs(STORE_DIR, exist_ok=True)
    docs = load_docs()
    chunks = chunk_docs(docs)
    vs = build_index(chunks)
    vs.save_local(STORE_DIR)
    print(f"Indexed {len(chunks)} chunks into {STORE_DIR}")