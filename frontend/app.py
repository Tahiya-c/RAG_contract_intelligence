import streamlit as st
import requests

API_URL = "http://contract_api:8000"

st.set_page_config(page_title="Contract Intelligence", layout="wide")
st.title("Contract Intelligence System")
st.caption("Ask questions about uploaded contracts. Answers are grounded in retrieved contract text only.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Corpus")
    try:
        res = requests.get(f"{API_URL}/contracts")
        data = res.json()
        st.metric("Documents", data["total_documents"])
        st.metric("Total chunks", data["total_chunks"])
        st.divider()
        for c in data["contracts"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(c["source"])
                st.caption(f"{c['total_chunks']} chunks")
            with col2:
                if st.button("🗑", key=f"del_{c['source']}"):
                    del_res = requests.delete(f"{API_URL}/contracts/{c['source']}")
                    if del_res.status_code == 200:
                        st.success("Moved to trash")
                        st.rerun()
                    else:
                        st.error("Failed")
    except Exception as e:
        st.error(f"Could not reach API: {e}")

    st.divider()
    st.header("Upload contract")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file and st.button("Upload"):
        with st.spinner("Processing..."):
            upload_res = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )
            if upload_res.status_code == 200:
                result = upload_res.json()
                st.success(f"Added {result['chunks_added']} chunks")
                st.rerun()
            else:
                st.error(f"Upload failed: {upload_res.text}")

    st.divider()
    st.header("Trash")
    try:
        del_res = requests.get(f"{API_URL}/deleted")
        deleted = del_res.json().get("deleted", [])
        if not deleted:
            st.caption("No deleted contracts")
        for d in deleted:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(d["source"])
                st.caption(f"{d['days_remaining']} days left")
            with col2:
                if st.button("↩", key=f"restore_{d['source']}"):
                    res = requests.post(f"{API_URL}/restore/{d['source']}")
                    if res.status_code == 200:
                        st.success("Restored")
                        st.rerun()
                    else:
                        st.error("Failed")
    except Exception as e:
        st.error(f"Trash error: {e}")

    if st.session_state.history:
        st.divider()
        st.header("History")
        for item in reversed(st.session_state.history):
            with st.expander(f"Q: {item['question'][:50]}..."):
                st.markdown(item["answer"])

question = st.text_input(
    "Ask a question about your contracts",
    placeholder="What happens if we terminate early?"
)

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating..."):
        try:
            res = requests.post(
                f"{API_URL}/query",
                json={"question": question}
            )
            result = res.json()

            st.session_state.history.append({
                "question": question,
                "answer": result["answer"]
            })

            st.subheader("Answer")
            st.markdown(result["answer"])

            st.subheader("Sources")
            for s in result["sources"]:
                score_pct = int(s["score"] * 100)
                st.markdown(
                    f"- `{s['source']}` — chunk {s['chunk_index']}"
                    f"&nbsp; `relevance: {score_pct}%`"
                )

        except Exception as e:
            st.error(f"Error: {e}")