import streamlit as st
from rag import answer

st.set_page_config(page_title="Thesis RAG Assistant", page_icon="🔎", layout="wide")
st.title("🔎 Thesis Research RAG Assistant")
q = st.text_input("Ask a question about your papers…", placeholder="e.g., How do time-of-use tariffs influence EV smart charging?")
if st.button("Search") or (q and "auto" in st.session_state):
    with st.spinner("Retrieving and generating..."):
        ans, docs = answer(q)
    st.subheader("Answer")
    st.write(ans)
    with st.expander("Show retrieved context"):
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "N/A")
            st.markdown(f"**{i}. {src} — p.{page}**")
            st.write(d.page_content[:1200] + ("…" if len(d.page_content) > 1200 else ""))

st.caption("Built with LangChain + FAISS + Streamlit")