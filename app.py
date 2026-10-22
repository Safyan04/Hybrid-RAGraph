import importlib
import re
import sys
import traceback
import uuid
from urllib.parse import urlparse

import streamlit as st

PIPELINE = ["i_loading", "ii_chunking", "iii_embedding", "iv_vectordb",
            "v_retrieval", "vi_tools", "vii_prompts", "viii_agents"]
TAG = re.compile(r"\[SOURCE:\s*([^\]]*)\]", re.I)

st.set_page_config(page_title="RAGraph", page_icon="⚡")
st.title("⚡ RAG Tool Agent")

st.session_state.setdefault("agent", None)
st.session_state.setdefault("pdf_signature", None)
st.session_state.setdefault("pdf_name", "")
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("messages", [])


def load_pipeline():
    """Re-run all pipeline files so the index is rebuilt from the new PDF."""
    for name in PIPELINE:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)
    return sys.modules["viii_agents"].agent


def get_text(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content
                   if isinstance(b, dict) and b.get("type") == "text")


def collect_sources(messages):
    """Gather what every tool returned for the latest question."""
    last = max(i for i, m in enumerate(messages) if m.type == "human")
    found = {}
    for m in messages[last + 1:]:
        if m.type != "tool":
            continue
        if m.name == "document_search":
            pages = {int(d.metadata.get("page", 0)) + 1 for d in (m.artifact or [])}
            found.setdefault("document", set()).update(pages)
        elif m.name == "web_search":
            links = re.findall(r"link:\s*(https?://[^\s,\]]+)", get_text(m.content))
            found.setdefault("web", set()).update(links)
        elif m.name == "calculator":
            found.setdefault("calculator", set())
        elif m.name == "get_datetime":
            found.setdefault("datetime", set())
    return found


def split_answer(text):
    """Separate the [SOURCE: ...] tag from the visible answer."""
    tags = set()
    for t in TAG.findall(text):
        tags |= {x.strip().lower() for x in t.split(",") if x.strip()}
    return TAG.sub("", text).strip(), tags


def show_source(msg):
    src = msg.get("sources")
    if src is None:          # error message, nothing to show
        return
    used = set(msg["used"])
    lines = []
    if "document" in used and src.get("document"):
        pages = ", ".join(map(str, src["document"]))
        lines.append(f"📄 **Document:** {st.session_state.pdf_name} (page {pages})")
    if "web" in used:
        links = src.get("web", [])
        lines.append("🌐 **Web search:** " + (" · ".join(
            f"[{urlparse(u).netloc}]({u})" for u in links) or "no links returned"))
    if "calculator" in used:
        lines.append("🧮 **Calculator tool**")
    if "datetime" in used:
        lines.append("🕒 **Time tool**")

    if lines:
        st.caption("  \n".join(lines))
    elif src:
        st.caption("🔎 Searched " + ", ".join(src) + " but found nothing usable.")
    else:
        st.caption("ℹ️ No tool used.")


# ---------- Sidebar: PDF upload ----------
with st.sidebar:
    pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is None:
    st.session_state.agent = None
    st.session_state.pdf_signature = None
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

signature = (pdf.name, pdf.size)
if st.session_state.pdf_signature != signature:
    with st.spinner("Reading the PDF and building the index..."):
        try:
            with open("uploaded.pdf", "wb") as f:
                f.write(pdf.getvalue())
            st.session_state.agent = load_pipeline()
            st.session_state.pdf_signature = signature
            st.session_state.pdf_name = pdf.name
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
        except Exception as e:
            traceback.print_exc()
            st.error(f"Could not process the PDF: {e}")
            st.stop()

with st.sidebar:
    st.success("Document ready")

# ---------- Chat ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_source(msg)

if query := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                found = collect_sources(result["messages"])
                answer, tags = split_answer(get_text(result["messages"][-1].content))

                declared = tags - {"none"}
                if declared:                 # trust only tools that were actually called
                    used = declared & set(found)
                elif tags:                   # model said "none"
                    used = set()
                else:                        # model forgot the tag
                    used = set(found)
                sources = {k: sorted(v) for k, v in found.items()}
            except Exception as e:
                traceback.print_exc()
                answer, sources, used = f"⚠️ Error: {e}", None, set()

        st.markdown(answer)
        msg = {"role": "assistant", "content": answer,
               "sources": sources, "used": sorted(used)}
        show_source(msg)
    st.session_state.messages.append(msg)