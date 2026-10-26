"""
Streamlit web UI for the RAG app.

Run with:  streamlit run app.py
"""

import html
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

# rag/ is a plain folder of modules (not a package), so make it importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "rag"))

from config import DOCS_DIR, ROOT  # noqa: E402
from generator import build_messages, generate  # noqa: E402
from highlight import find_best_sentence  # noqa: E402
from ingest import ingest_folder  # noqa: E402
from loaders import format_location  # noqa: E402
from retriever import retrieve  # noqa: E402
from store import get_collection, list_sources  # noqa: E402

st.set_page_config(page_title="Ask your documents", page_icon="📄")


# ---------------------------------------------------------------------------
# Cached helpers (Streamlit re-runs this whole script on every interaction)
# ---------------------------------------------------------------------------


@st.cache_resource
def cached_collection():
    """Open the Chroma collection once, not on every re-run."""
    return get_collection()


# Runs in a separate Python process: MuPDF crashes the whole process on some PDF pages,
# and this way only the helper dies, not the app.
RENDER_SCRIPT = """
import re
import sys
import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)  # MuPDF warnings would otherwise mix into the PNG on stdout

def search_candidates(sentence):
    # The sentence can start with page-header junk ("UNOFFICIAL", a page number) that sits
    # elsewhere on the page, so try progressively shorter versions of it
    lines = [line.strip() for line in sentence.splitlines() if line.strip()]
    while len(lines) > 1 and len(lines[0]) < 20:
        lines.pop(0)
    core = re.sub(r"^\\d+\\s+", "", " ".join(lines))
    return [sentence, core, core[:60], core[-60:]]


path, page_number, sentence = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with pymupdf.open(path) as doc:
    page = doc[page_number - 1]
    if sentence:
        for candidate in search_candidates(sentence):
            rects = page.search_for(candidate)
            if rects:
                page.add_highlight_annot(rects)  # only in memory, the file isn't changed
                break
    sys.stdout.buffer.write(page.get_pixmap(dpi=110).tobytes("png"))
"""


@st.cache_data(show_spinner=False)
def render_pdf_page(path: str, page_number: int, sentence: str | None) -> bytes | None:
    """Render one PDF page as a PNG with the sentence highlighted, or None if rendering fails."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                RENDER_SCRIPT,
                path,
                str(page_number),
                sentence or "",
            ],
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    is_png = result.stdout.startswith(b"\x89PNG")
    return result.stdout if result.returncode == 0 and is_png else None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def md_escape(text: str) -> str:
    """Stop Streamlit's markdown from turning "$7.50 ... $10" into a math formula."""
    return text.replace("$", r"\$")


def highlighted_html(text: str, sentence: str | None) -> str:
    """The chunk text as HTML, with the matched sentence wrapped in <mark>."""
    if sentence and sentence in text:
        before, _, after = text.partition(sentence)
        mark = '<mark style="background: #ffe066; color: #1a1a1a; padding: 0 2px; border-radius: 3px">'
        body = f"{html.escape(before)}{mark}{html.escape(sentence)}</mark>{html.escape(after)}"
    else:
        body = html.escape(text)
    return f'<div style="white-space: pre-wrap; font-size: 0.9rem; line-height: 1.5">{body}</div>'


def resolve_path(path: str) -> Path:
    """Chunk paths may be relative to the project folder."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_sources(answer: str, hits: list[dict]) -> list[dict]:
    """Everything needed to display the sources an answer cited."""
    cited = sorted(
        {int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(hits)}
    )
    sources = []
    for n in cited:
        hit = hits[n - 1]
        match = find_best_sentence(answer, hit["text"])
        sources.append(
            {
                "n": n,
                "source": hit["source"],
                "location": format_location(hit),
                "text": hit["text"],
                "sentence": match[0] if match else None,
                "path": hit["path"],
                "page_number": hit["page_number"],
            }
        )
    return sources


def render_sources(sources: list[dict]):
    for src in sources:
        with st.expander(f"[{src['n']}] {src['source']}, {src['location']}"):
            st.html(highlighted_html(src["text"], src["sentence"]))

            path = resolve_path(src["path"])
            if (
                path.suffix.lower() == ".pdf"
                and src["page_number"] != -1
                and path.exists()
            ):
                image = render_pdf_page(str(path), src["page_number"], src["sentence"])
                if image:
                    st.image(image, caption=f"{src['source']}, {src['location']}")
                else:
                    st.caption("This PDF page couldn't be rendered as an image.")


# ---------------------------------------------------------------------------
# Sidebar: document library
# ---------------------------------------------------------------------------

collection = cached_collection()
Path(DOCS_DIR).mkdir(exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # changing the key clears the file uploader


def run_ingest() -> str:
    with st.spinner("Indexing documents..."):
        results, deleted = ingest_folder(DOCS_DIR, collection)
    counts = {}
    for status in results.values():
        counts[status] = counts.get(status, 0) + 1
    summary = (
        ", ".join(f"{count} {status}" for status, count in counts.items()) or "no files"
    )
    if deleted:
        summary += f", {len(deleted)} removed"
    return summary


with st.sidebar:
    st.header("Library")

    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    uploads = st.file_uploader(
        "Add documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploads and st.button("Add to library", type="primary"):
        for upload in uploads:
            (Path(DOCS_DIR) / upload.name).write_bytes(upload.getbuffer())
        st.session_state.flash = run_ingest()
        st.session_state.uploader_key += 1
        st.rerun()

    if st.button("Re-scan docs folder"):
        st.session_state.flash = run_ingest()
        st.rerun()

    st.divider()
    sources_in_db = sorted(list_sources(collection))
    st.caption(f"{len(sources_in_db)} documents, {collection.count()} chunks indexed")
    for name in sources_in_db:
        st.markdown(f"- {md_escape(name)}")

    st.divider()
    top_k = st.slider("Passages to retrieve", min_value=3, max_value=10, value=5)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

st.title("Ask your documents")

# Redraw the conversation so far (the script re-ran, so the page is empty)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(md_escape(message["content"]))
        if message.get("sources"):
            render_sources(message["sources"])

if question := st.chat_input("e.g. When does the lease expire?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md_escape(question))

    with st.chat_message("assistant"):
        if collection.count() == 0:
            answer = "Your library is empty. Add some documents in the sidebar first."
            st.markdown(answer)
            sources = []
        else:
            with st.spinner("Searching your documents..."):
                hits = retrieve(
                    question, k=min(top_k, collection.count()), collection=collection
                )

            raw_pieces = []

            def stream():
                # Keep the raw text for citations, but escape "$" for display
                for piece in generate(build_messages(question, hits)):
                    raw_pieces.append(piece)
                    yield md_escape(piece)

            st.write_stream(stream())
            answer = "".join(raw_pieces)
            sources = build_sources(answer, hits)
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
