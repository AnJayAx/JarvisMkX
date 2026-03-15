"""
Jarvis Mk.X — Smart Research Paper Q&A Chatbot
Streamlit Application (v2 — Multi-PDF + Visualizations)

Run: streamlit run app.py
"""

import streamlit as st
import sys
import os
import time
import json
import re
import fitz
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")

from database import (
    create_session, get_all_sessions, search_sessions, get_session,
    update_session_title, update_session_settings, delete_session,
    add_message, get_messages, add_pdf, get_active_pdfs, remove_pdf,
    get_corrections,
)
from pdf_export import export_chat_to_pdf

# ─── Page Config ───
st.set_page_config(
    page_title="Jarvis Mk.X",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───
st.markdown("""
<style>
    .stApp { 
        font-family: 'Segoe UI', sans-serif;
        background: radial-gradient(circle at 20% 5%, #1b2433 0%, #0d1117 40%, #0a0f14 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    /* Prevent a leftover gutter when the sidebar is collapsed */
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 340px !important;
        max-width: 340px !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        max-width: 0 !important;
        width: 0 !important;
        border-right: none !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    [data-testid="stSidebarResizer"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        letter-spacing: 0.2px;
        font-weight: 700;
    }
    [data-testid="stSidebar"] .stTextInput input {
        background-color: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 12px;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] .stTextInput input:focus {
        border-color: #60a5fa;
        box-shadow: 0 0 0 1px #60a5fa;
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        background: rgba(148, 163, 184, 0.08);
        color: #e2e8f0;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(96, 165, 250, 0.85);
        background: rgba(96, 165, 250, 0.15);
        transform: translateY(-1px);
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        border-color: rgba(96, 165, 250, 0.9);
        color: #f8fafc;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] p {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    [data-testid="stSidebar"] [class*="st-key-session_title_"] .stButton > button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #d1d5db !important;
        border-radius: 0 !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        text-align: left !important;
        width: 100% !important;
        min-height: 2rem !important;
        /* Let the tooltip overflow the button; ellipsis is applied to the inner text element */
        overflow: visible !important;
        transform: none !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    [data-testid="stSidebar"] [class*="st-key-session_title_"] .stButton > button:hover {
        background: transparent !important;
        color: #e5e7eb !important;
    }
    [data-testid="stSidebar"] [class*="st-key-session_title_"] .stButton > button p {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        margin: 0 !important;
        width: 100%;
        text-align: left !important;
    }
    [data-testid="stSidebar"] [class*="st-key-session_title_"] .stButton > button * {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        text-align: left !important;
    }

    /* Fallback selectors (Streamlit markup varies by version) */
    [data-testid="stSidebar"] [class*="st-key-session_title_"] button,
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] button {
        /* Remove the “boxed button” look */
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        /* Allow hover tooltip to overflow */
        overflow: visible !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        position: relative !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* Streamlit often wraps button labels in inner divs/spans; force them left */
    [data-testid="stSidebar"] [class*="st-key-session_title_"] button > div,
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] button > div {
        flex: 1 1 auto !important;
        width: 100% !important;
        text-align: left !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [class*="st-key-session_title_"] button div,
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] button div {
        text-align: left !important;
        justify-content: flex-start !important;
    }

    /* Force a true 1-line render even if inner markup wraps */
    [data-testid="stSidebar"] [class*="st-key-session_title_"] button p,
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] button p,
    [data-testid="stSidebar"] [class*="st-key-session_title_"] button span,
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] button span {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 100% !important;
    }

    /* Full title on hover is handled by the browser's native tooltip via the button's `title`
       attribute (Streamlit sets this from `help=`). */
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] .stButton > button {
        color: #bfdbfe !important;
        font-weight: 700 !important;
    }

    /* Active session highlight (robust across Streamlit DOM variations) */
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] {
        border-left: 3px solid rgba(96, 165, 250, 0.85) !important;
        padding-left: 0.55rem !important;
    }
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] button {
        color: #bfdbfe !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [class*="st-key-session_delete_"] .stButton > button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #9ca3af !important;
        width: 100% !important;
        min-width: 100% !important;
        justify-content: flex-end !important;
        padding-left: 0 !important;
        padding-right: 0.1rem !important;
        transform: none !important;
    }
    [data-testid="stSidebar"] [class*="st-key-session_delete_"] .stButton > button:hover {
        background: transparent !important;
        color: #f87171 !important;
    }
    .chat-user { background: #1e3a5f; padding: 12px 16px; border-radius: 12px;
                 margin: 4px 0; border-left: 3px solid #4da6ff; }
    .chat-bot { background: #1a3328; padding: 12px 16px; border-radius: 12px;
                margin: 4px 0; border-left: 3px solid #4dff88; }
    .pdf-summary { background: #2a2a3e; padding: 14px; border-radius: 10px;
                   margin: 6px 0; border: 1px solid #444; }
    .stat-card { background: #1e1e2e; padding: 10px 14px; border-radius: 8px;
                 text-align: center; border: 1px solid #333; }

    /* Sticky composer (prompt + settings)
       NOTE: Streamlit's DOM varies by version; target the keyed container directly. */
    section.main {
        /* Prevent chat messages from being hidden behind the fixed composer */
        padding-bottom: 320px !important;
    }
    .st-key-composer {
        position: fixed !important;
        bottom: 0 !important;
        right: 0 !important;
        left: 0 !important;
        z-index: 999 !important;
        box-sizing: border-box !important;
        width: auto !important;
        max-width: none !important;
        padding: 0.9rem 1.1rem 1.1rem 1.1rem !important;
        background: rgba(13, 17, 23, 0.88) !important;
        backdrop-filter: blur(10px) !important;
        border-top: 1px solid rgba(148, 163, 184, 0.18) !important;
    }
    /* Align composer with main area when sidebar is expanded.
       `:has()` is supported in modern Chromium and is much more reliable than sibling selectors
       across Streamlit DOM variations. */
    body:has([data-testid="stSidebar"][aria-expanded="true"]) .st-key-composer {
        left: 340px !important;
        width: calc(100% - 340px) !important;
    }
    body:has([data-testid="stSidebar"][aria-expanded="false"]) .st-key-composer {
        left: 0 !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ───
def init_state():
    defaults = {
        "current_session": None,
        "bot": None,
        "bot_loaded": False,
        "papers_loaded_key": None,
        "suggested_questions": [],
        "all_chunks_cache": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Bot Loading ───
@st.cache_resource(show_spinner=False)
def load_bot():
    from bot import JarvisBot

    # Embeddings selection:
    # - Set JARVIS_EMBED_MODEL to force a specific backend (e.g., "voyage-3-large").
    # - Otherwise, use Voyage only when VOYAGE_API_KEY is configured.
    forced_embed_model = os.environ.get("JARVIS_EMBED_MODEL", "").strip()
    if forced_embed_model:
        embed_model_name = forced_embed_model
    else:
        embed_model_name = (
            "voyage-4-large"
            if os.environ.get("VOYAGE_API_KEY")
            else "sentence-transformers/allenai-specter"
        )

    bot = JarvisBot(
        base_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        adapter_path="models/jarvis-mkx-adapter-v2",
        embed_model_name=embed_model_name,
        chunk_size=512, chunk_overlap=50, load_in_4bit=True,
    )
    bot.load_model()
    return bot


def get_bot():
    if st.session_state.bot is None:
        with st.spinner("🧠 Loading Jarvis AI (first time ~30s)..."):
            st.session_state.bot = load_bot()
            st.session_state.bot_loaded = True
    return st.session_state.bot


# ─── Multi-PDF Loading ───
def load_all_pdfs_into_bot(session_id, active_pdfs, messages):
    """Load all active PDFs by combining their chunks into one index."""
    if not active_pdfs:
        return

    cache_key = f"{session_id}_{'_'.join(p['id'] for p in active_pdfs)}"
    if st.session_state.papers_loaded_key == cache_key:
        return  # Already loaded this exact combination

    bot = get_bot()
    from processor import PaperProcessor

    processor = PaperProcessor(chunk_size=512, chunk_overlap=50)
    all_chunks = []

    for pdf in active_pdfs:
        if os.path.exists(pdf["filepath"]):
            paper = processor.process(pdf["filepath"])
            # Tag chunks with source PDF name
            for chunk in paper.chunks:
                chunk.section = f"[{pdf['filename']}] {chunk.section}"
            all_chunks.extend(paper.chunks)
            # Keep the last paper's metadata for meta questions
            bot.current_paper = paper

    if all_chunks:
        bot.retriever.build_index(all_chunks)
        st.session_state.all_chunks_cache = all_chunks

    # Reload conversation history
    from bot import ConversationTurn
    bot.conversation_history = []
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            bot.conversation_history.append(
                ConversationTurn(role=msg["role"], content=msg["content"])
            )
    if len(bot.conversation_history) > bot.max_history_turns:
        bot.conversation_history = bot.conversation_history[-bot.max_history_turns:]

    # Reload corrections
    corrections = get_corrections(session_id)
    bot.corrections = [
        {"question": c["question"], "wrong_answer": c["wrong_answer"],
         "correction": c["correction"]}
        for c in corrections
    ]

    st.session_state.papers_loaded_key = cache_key


# ─── Helpers ───

def generate_suggested_questions(paper_info):
    sections = paper_info.get("sections", [])
    suggestions = [
        "Can you give me a summary of this paper?",
        "What are the key findings?",
    ]
    for sec in sections:
        sl = sec.lower()
        if "method" in sl: suggestions.append(f"What methodology is used in '{sec}'?")
        elif "result" in sl: suggestions.append("What are the main results?")
        elif "conclusion" in sl: suggestions.append("What conclusions does the paper draw?")
        elif "experiment" in sl: suggestions.append("What experiments were conducted?")
    return suggestions[:6]


def format_sources_for_db(sources):
    return [
        {"section": s.section, "page_numbers": s.page_numbers,
         "score": round(s.score, 4),
         "method": getattr(s, "retrieval_method", "unknown"),
         # Store full text so the analytics panel can show complete passages.
         # Keep a short preview for charts / lightweight displays.
         "text": s.text,
         "text_preview": s.text[:300]}
        for s in sources
    ]


def _normalize_pdf_search_text(text: str) -> str:
    if not text:
        return ""
    # PDFs may contain line breaks/hyphenation; collapse whitespace for matching.
    return " ".join(str(text).replace("\u00ad", "").split())


def _normalize_token(token: str) -> str:
    token = (token or "").lower()
    # Keep alphanumerics; drop punctuation to be resilient to PDF quirks.
    token = re.sub(r"[^a-z0-9]+", "", token)
    return token


def _first_sentence(text: str, max_chars: int = 260) -> str:
    """Return a short, sentence-like snippet to highlight.

    Highlighting entire chunks is brittle (columns, hyphenation, line wraps).
    A short snippet greatly increases match reliability.
    """
    t = _normalize_pdf_search_text(text)
    if not t:
        return ""
    # Try to cut at sentence boundary.
    m = re.search(r"(.+?[.!?])\s", t)
    if m:
        snippet = m.group(1)
    else:
        snippet = t
    return snippet[:max_chars].strip()


def _rect_union(rects):
    if not rects:
        return None
    r = fitz.Rect(rects[0])
    for rr in rects[1:]:
        r |= fitz.Rect(rr)
    return r


def _find_highlight_rects_by_words(page, text: str, max_words: int = 35):
    """Best-effort mapping from passage text -> highlight rectangles.

    Uses word-level coordinates rather than string search to reduce drift.
    Returns a list of fitz.Rect (usually one per line).
    """
    snippet = _first_sentence(text)
    if not snippet:
        return []

    target_tokens = [_normalize_token(t) for t in snippet.split()]
    target_tokens = [t for t in target_tokens if t]
    if not target_tokens:
        return []
    target_tokens = target_tokens[:max_words]

    words = page.get_text("words") or []
    if not words:
        return []

    page_tokens = []
    for w in words:
        tok = _normalize_token(w[4])
        page_tokens.append(tok)

    # Anchor match on first N tokens.
    anchor_len = min(8, len(target_tokens))
    anchor = target_tokens[:anchor_len]
    if anchor_len < 3:
        return []

    candidate_starts = []
    for i in range(0, len(page_tokens) - anchor_len + 1):
        if page_tokens[i:i + anchor_len] == anchor:
            candidate_starts.append(i)
            if len(candidate_starts) >= 5:
                break

    # If no exact anchor match, give up (caller can fall back to search_for).
    if not candidate_starts:
        return []

    # Pick the start that yields the longest forward match.
    best = None
    best_len = 0
    for start in candidate_starts:
        j = 0
        while (start + j) < len(page_tokens) and j < len(target_tokens):
            if page_tokens[start + j] != target_tokens[j]:
                break
            j += 1
        if j > best_len:
            best_len = j
            best = (start, j)

    if not best or best_len < 4:
        return []

    start, length = best
    matched_words = words[start:start + length]
    # Group into line-ish rectangles by y coordinate.
    line_groups = []
    current = []
    current_y = None

    for w in matched_words:
        rect = fitz.Rect(w[0], w[1], w[2], w[3])
        y = round(rect.y0, 1)
        if current_y is None:
            current_y = y
            current.append(rect)
            continue
        if abs(y - current_y) <= 2.5:
            current.append(rect)
        else:
            line_groups.append(current)
            current = [rect]
            current_y = y
    if current:
        line_groups.append(current)

    rects = []
    for group in line_groups:
        u = _rect_union(group)
        if u:
            rects.append(u)
    return rects


def _build_search_snippets(text: str) -> list:
    """Build short snippets likely to match PDF text extraction."""
    t = _normalize_pdf_search_text(text)
    if not t:
        return []

    words = t.split()
    snippets = []

    # Prefer word-based snippets (more stable than raw character slices).
    if len(words) >= 6:
        snippets.append(" ".join(words[:12]))
    if len(words) >= 24:
        mid = len(words) // 2
        snippets.append(" ".join(words[mid: mid + 12]))
    if len(words) >= 12:
        snippets.append(" ".join(words[-12:]))

    # Fallback: first ~120 characters.
    snippets.append(t[:120])

    # De-dupe + drop very short snippets.
    seen = set()
    out = []
    for s in snippets:
        s = s.strip()
        if len(s) < 15:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def render_pdf_page(filepath, page_num, highlight_texts=None):
    try:
        doc = fitz.open(filepath)
        page = doc[min(page_num, len(doc) - 1)]

        # Best-effort highlighting: search for snippets of the retrieved passage.
        if highlight_texts:
            try:
                for text in highlight_texts:
                    # 1) Try coordinate-aware matching via words
                    rects = _find_highlight_rects_by_words(page, text)
                    if rects:
                        for r in rects:
                            try:
                                page.add_highlight_annot(r)
                            except Exception:
                                pass
                        break

                    # 2) Fallback: plain text search
                    found_any = False
                    for snippet in _build_search_snippets(text):
                        try:
                            hits = page.search_for(snippet, quads=True)
                        except TypeError:
                            hits = page.search_for(snippet)
                        if hits:
                            try:
                                page.add_highlight_annot(hits)
                            except Exception:
                                for h in hits:
                                    try:
                                        page.add_highlight_annot([h])
                                    except Exception:
                                        pass
                            found_any = True
                            break
                    if found_any:
                        break
            except Exception:
                # If highlighting fails, still return a normal rendered page.
                pass

        pix = page.get_pixmap(
            matrix=fitz.Matrix(1.5, 1.5),
            annots=True if highlight_texts else False,
        )
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except:
        return None


# ─── Visualization Functions ───

def create_confidence_gauge(confidence):
    color = "#4dff88" if confidence > 0.5 else "#ffcc00" if confidence > 0.3 else "#ff4444"
    value = max(0.0, min(1.0, float(confidence))) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": "Confidence %", "font": {"size": 14}},
        number={"suffix": "%", "valueformat": ".0f", "font": {"size": 46}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "bgcolor": "#1e1e2e",
               "steps": [{"range": [0, 30], "color": "#3a1a1a"},
                         {"range": [30, 60], "color": "#3a3a1a"},
                         {"range": [60, 100], "color": "#1a3a1a"}]},
    ))
    # Extra right margin prevents the number text from being clipped on narrow layouts.
    fig.update_layout(height=200, margin=dict(l=10, r=60, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ccc"})
    return fig


def create_source_chart(sources):
    if not sources: return None
    sections = [s.get("section", "?")[:30] for s in sources]
    scores = [s.get("score", 0) for s in sources]
    methods = [s.get("method", "?") for s in sources]
    colors = ["#4da6ff" if m == "dense" else "#ff9944" if m == "sparse" else "#44ff88"
              for m in methods]
    fig = go.Figure(go.Bar(x=scores, y=sections, orientation="h",
                           marker_color=colors,
                           text=[f"{s:.3f}" for s in scores],
                           textposition="outside"))
    fig.update_layout(title="Retrieval Scores by Source", xaxis_title="Score",
                      height=max(200, len(sources) * 50),
                      margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#ccc"})
    return fig


def create_method_pie(sources):
    """Pie chart of retrieval methods used."""
    if not sources: return None
    methods = [s.get("method", "unknown") for s in sources]
    from collections import Counter
    counts = Counter(methods)
    colors = {"dense": "#4da6ff", "sparse": "#ff9944", "hybrid": "#44ff88", "unknown": "#888"}
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()), values=list(counts.values()),
        marker_colors=[colors.get(m, "#888") for m in counts.keys()],
        hole=0.4,
    ))
    fig.update_layout(title="Retrieval Method Distribution", height=250,
                      margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ccc"})
    return fig


def create_3d_vector_space(query_text, sources, embed_model):
    """3D UMAP/PCA projection of query vs retrieved chunks."""
    if not sources or len(sources) < 2:
        return None

    try:
        from sklearn.decomposition import PCA

        # Collect texts
        texts = [query_text] + [s.get("text_preview", "")[:200] for s in sources]
        labels = ["Query"] + [f"Chunk {i+1}: {s.get('section', '?')[:20]}" for i, s in enumerate(sources)]
        types = ["query"] + ["chunk"] * len(sources)
        scores = [1.0] + [s.get("score", 0) for s in sources]

        # Embed
        embeddings = embed_model.encode(texts, normalize_embeddings=True)

        # PCA to 3D
        pca = PCA(n_components=3)
        coords = pca.fit_transform(embeddings)

        colors = ["#ff4444"] + ["#44ff88" if sc > 0.5 else "#ffcc00" if sc > 0.3 else "#4da6ff"
                                 for sc in scores[1:]]
        sizes = [15] + [max(8, sc * 20) for sc in scores[1:]]

        fig = go.Figure()

        # Chunks
        for i in range(1, len(coords)):
            fig.add_trace(go.Scatter3d(
                x=[coords[i, 0]], y=[coords[i, 1]], z=[coords[i, 2]],
                mode="markers+text", text=[labels[i]],
                textposition="top center",
                marker=dict(size=sizes[i], color=colors[i], opacity=0.8),
                name=labels[i],
            ))
            # Line from query to chunk
            fig.add_trace(go.Scatter3d(
                x=[coords[0, 0], coords[i, 0]],
                y=[coords[0, 1], coords[i, 1]],
                z=[coords[0, 2], coords[i, 2]],
                mode="lines",
                line=dict(color=colors[i], width=2, dash="dash"),
                showlegend=False, opacity=0.4,
            ))

        # Query point (larger, red)
        fig.add_trace(go.Scatter3d(
            x=[coords[0, 0]], y=[coords[0, 1]], z=[coords[0, 2]],
            mode="markers+text", text=["YOUR QUERY"],
            textposition="top center",
            marker=dict(size=15, color="#ff4444", symbol="diamond", opacity=1.0),
            name="Query",
        ))

        fig.update_layout(
            title="3D Vector Space: Query vs Retrieved Chunks",
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})",
                bgcolor="rgba(0,0,0,0)",
            ),
            height=500, margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ccc"},
        )
        return fig
    except Exception as e:
        st.caption(f"3D viz error: {e}")
        return None


def create_score_heatmap(sources):
    """Heatmap showing dense vs sparse scores for each source."""
    if not sources or len(sources) < 2: return None

    sections = [s.get("section", "?")[:25] for s in sources]
    hybrid = [s.get("score", 0) for s in sources]

    # Estimate dense/sparse split from method
    dense_est = []
    sparse_est = []
    for s in sources:
        h = s.get("score", 0)
        m = s.get("method", "hybrid")
        if m == "dense":
            dense_est.append(h)
            sparse_est.append(0.0)
        elif m == "sparse":
            dense_est.append(0.0)
            sparse_est.append(h)
        else:
            dense_est.append(h * 0.6)
            sparse_est.append(h * 0.4)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Dense (Semantic)", y=sections, x=dense_est,
                         orientation="h", marker_color="#4da6ff"))
    fig.add_trace(go.Bar(name="Sparse (BM25)", y=sections, x=sparse_est,
                         orientation="h", marker_color="#ff9944"))

    fig.update_layout(barmode="stack", title="Dense vs Sparse Score Breakdown",
                      xaxis_title="Score", height=max(200, len(sources) * 50),
                      margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#ccc"}, legend=dict(orientation="h", yanchor="bottom"))
    return fig


def create_chunk_length_chart(sources):
    """Bar chart showing text length of each retrieved chunk."""
    if not sources: return None
    sections = [f"Chunk {i+1}" for i in range(len(sources))]
    lengths = [len(s.get("text_preview", "")) for s in sources]
    fig = go.Figure(go.Bar(x=sections, y=lengths, marker_color="#7c4dff",
                           text=lengths, textposition="outside"))
    fig.update_layout(title="Retrieved Chunk Sizes (chars)", yaxis_title="Characters",
                      height=250, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#ccc"})
    return fig


def create_answer_word_cloud_data(answer_text):
    """Simple word frequency bar chart (pseudo word-cloud)."""
    import re
    from collections import Counter
    words = re.findall(r'\b[a-zA-Z]{4,}\b', answer_text.lower())
    stopwords = {"this", "that", "with", "from", "have", "been", "were", "they",
                 "their", "which", "would", "could", "about", "into", "more",
                 "also", "than", "other", "some", "such", "when", "what", "there"}
    words = [w for w in words if w not in stopwords]
    counts = Counter(words).most_common(12)
    if not counts: return None
    words, freqs = zip(*counts)
    fig = go.Figure(go.Bar(x=list(freqs), y=list(words), orientation="h",
                           marker_color="#e040fb"))
    fig.update_layout(title="Key Terms in Answer", xaxis_title="Frequency",
                      height=300, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#ccc"})
    return fig


# ──────────────────────────────────────────────
#                  SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Jarvis Mk.X")
    st.caption("Smart Research Paper Q&A")

    search_query = st.text_input("Search", placeholder="Find a chat...")

    if st.button("+ New Chat", use_container_width=True, type="primary"):
        sid = create_session("New Chat")
        st.session_state.current_session = sid
        st.session_state.papers_loaded_key = None
        st.session_state.suggested_questions = []
        st.rerun()

    st.divider()

    sessions = search_sessions(search_query) if search_query else get_all_sessions()

    if not sessions:
        st.info("No chats yet. Click 'New Chat'!")
    else:
        with st.container(key="session_list"):
            for sess in sessions:
                c1, c2 = st.columns([5, 1])
                with c1:
                    is_active = st.session_state.current_session == sess["id"]
                    full_title = sess["title"]
                    # Render the full title; CSS enforces a 1-line ellipsis.
                    # Keeping the full text in the DOM ensures hover tooltips can show the whole title.
                    display_title = full_title
                    title_container_key = (
                        f"session_title_active_{sess['id']}"
                        if is_active else f"session_title_{sess['id']}"
                    )
                    with st.container(key=title_container_key):
                        if st.button(display_title, key=f"s_{sess['id']}",
                                     use_container_width=True,
                                     help=full_title,
                                     type="secondary"):
                            st.session_state.current_session = sess["id"]
                            st.session_state.papers_loaded_key = None
                            st.session_state.suggested_questions = []
                            st.rerun()
                with c2:
                    with st.container(key=f"session_delete_{sess['id']}"):
                        if st.button("🗑️", key=f"d_{sess['id']}", use_container_width=True):
                            delete_session(sess["id"])
                            if st.session_state.current_session == sess["id"]:
                                st.session_state.current_session = None
                            st.rerun()

    st.divider()



# ──────────────────────────────────────────────
#                  MAIN AREA
# ──────────────────────────────────────────────

if st.session_state.current_session is None:
    st.markdown("# Jarvis Mk.X")
    st.markdown("### Smart Research Paper Chatbot")
    st.markdown("""
    Upload research papers and ask questions. Jarvis uses fine-tuned Mistral 7B
    with hybrid retrieval (Voyage 3 Large + ChromaDB + BM25).

    **Features:** Up to 3 PDFs | Persistent memory | Adjustable settings |
    Visualizations | Answer correction | PDF export

    👉 Click **New Chat** to start!
    """)

else:
    session = get_session(st.session_state.current_session)
    if not session:
        st.error("Session not found.")
        st.stop()

    session_id = session["id"]

    # ─── Title + Export ───
    c_title, c_export = st.columns([6, 1])
    with c_title:
        new_title = st.text_input("Title", value=session["title"],
                                   label_visibility="collapsed")
        if new_title != session["title"]:
            update_session_title(session_id, new_title)
    with c_export:
        if st.button("📥 Export"):
            msgs = get_messages(session_id)
            if msgs:
                path = export_chat_to_pdf(session["title"], msgs)
                with open(path, "rb") as f:
                    st.download_button("Download", f.read(),
                                       file_name=os.path.basename(path),
                                       mime="application/pdf")

    # ─── PDF Upload ───
    st.markdown("---")
    active_pdfs = get_active_pdfs(session_id)

    cu, cm = st.columns([3, 2])
    with cu:
        if len(active_pdfs) < 3:
            # Streamlit file_uploader preserves selected files across reruns.
            # To clear it after a successful upload, rotate its key using a per-session nonce.
            nonce_key = f"up_nonce_{session_id}"
            if nonce_key not in st.session_state:
                st.session_state[nonce_key] = 0
            uploader_key = f"up_{session_id}_{st.session_state[nonce_key]}"

            uploaded = st.file_uploader(
                f"📄 Upload PDF ({len(active_pdfs)}/3)",
                type=["pdf"],
                accept_multiple_files=True,
                key=uploader_key,
            )
            if uploaded:
                bot = get_bot()
                from processor import PaperProcessor
                processor = PaperProcessor(chunk_size=512, chunk_overlap=50)

                for uf in uploaded:
                    if len(get_active_pdfs(session_id)) >= 3:
                        st.warning("Max 3 PDFs.")
                        break
                    existing = [p["filename"] for p in get_active_pdfs(session_id)]
                    if uf.name in existing:
                        st.info(f"{uf.name} already uploaded.")
                        continue

                    upload_dir = f"data/uploads/{session_id}"
                    os.makedirs(upload_dir, exist_ok=True)
                    filepath = os.path.join(upload_dir, uf.name)
                    with open(filepath, "wb") as f:
                        f.write(uf.getbuffer())

                    with st.spinner(f"Processing {uf.name}..."):
                        paper = processor.process(filepath)
                        # Generate summary using bot
                        bot.current_paper = paper
                        bot.retriever.build_index(paper.chunks)
                        summary_resp = bot.ask("Give me a brief summary of this paper.",
                                                leniency=100)
                        summary = summary_resp.answer if summary_resp.answer else "Summary unavailable."

                    add_pdf(session_id=session_id, filename=uf.name, filepath=filepath,
                            summary=summary, num_pages=paper.num_pages,
                            num_chunks=len(paper.chunks),
                            sections=list(paper.sections.keys()))

                    st.session_state.suggested_questions = generate_suggested_questions(
                        {"sections": list(paper.sections.keys())})
                    st.session_state.papers_loaded_key = None  # Force reload
                    st.success(f"✅ {uf.name}")

                # Rotate the uploader key to clear the selected files on the next run.
                st.session_state[nonce_key] += 1
                st.rerun()
        else:
            st.info("📄 3/3 PDFs uploaded.")

    with cm:
        if active_pdfs:
            st.markdown("**Active PDFs:**")
            for pdf in active_pdfs:
                pc1, pc2 = st.columns([4, 1])
                with pc1:
                    st.caption(f"📄 {pdf['filename']} ({pdf['num_pages']}p, {pdf['num_chunks']} chunks)")
                with pc2:
                    if st.button("❌", key=f"rm_{pdf['id']}"):
                        remove_pdf(pdf["id"])
                        st.session_state.papers_loaded_key = None
                        st.rerun()

    # ─── Summaries ───
    active_pdfs = get_active_pdfs(session_id)
    if active_pdfs:
        with st.expander("📋 PDF Summaries", expanded=False):
            for pdf in active_pdfs:
                st.markdown(f"""<div class="pdf-summary">
                    <strong>📄 {pdf['filename']}</strong> — {pdf['num_pages']}p, {pdf['num_chunks']} chunks<br>
                    <em>{pdf['summary']}</em></div>""", unsafe_allow_html=True)

    # ─── Suggested Questions ───
    if st.session_state.suggested_questions and active_pdfs:
        st.markdown("**💡 Suggested:**")
        cols = st.columns(3)
        for idx, q in enumerate(st.session_state.suggested_questions[:6]):
            with cols[idx % 3]:
                if st.button(q, key=f"sq_{idx}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

    st.markdown("---")

    # ─── Load ALL PDFs into bot ───
    messages = get_messages(session_id)
    if active_pdfs:
        load_all_pdfs_into_bot(session_id, active_pdfs, messages)

    # ─── Chat Messages ───
    for msg in messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("confidence", 0) > 0:
                sources = msg.get("sources", [])

                # ─── VISUALIZATIONS ───
                with st.expander("📊 Answer Analytics", expanded=False):
                    # Row 1: Confidence + Source Scores
                    v1, v2 = st.columns(2)
                    with v1:
                        st.plotly_chart(create_confidence_gauge(msg["confidence"]),
                                         use_container_width=True,
                                         key=f"chart_conf_{msg['id']}"
                                         )
                        st.caption(f"⏱️ {msg.get('generation_time', 0):.1f}s | "
                                   f"🔗 {msg.get('retrieval_methods', 'N/A')}")
                    with v2:
                        fig = create_source_chart(sources)
                        if fig:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"chart_sources_{msg['id']}"
                            )

                    # Row 2: Method Pie + Score Heatmap
                    v3, v4 = st.columns(2)
                    with v3:
                        fig = create_method_pie(sources)
                        if fig:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"chart_methodpie_{msg['id']}"
                            )
                    with v4:
                        fig = create_score_heatmap(sources)
                        if fig:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"chart_heatmap_{msg['id']}"
                            )

                    # Row 3: 3D Vector Space
                    if sources and len(sources) >= 2:
                        st.markdown("**🌐 3D Vector Space Projection (PCA)**")
                        st.caption("Red diamond = your query. Green = high score chunks. "
                                   "Dashed lines = similarity connections.")
                        # Get the query from the previous user message
                        prev_user = [m for m in messages if m["id"] < msg["id"] and m["role"] == "user"]
                        query_text = prev_user[-1]["content"] if prev_user else "query"
                        bot = get_bot()
                        fig = create_3d_vector_space(query_text, sources, bot.retriever.embed_model)
                        if fig:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"chart_3d_{msg['id']}"
                            )

                    # Row 4: Chunk Lengths + Answer Keywords
                    v5, v6 = st.columns(2)
                    with v5:
                        fig = create_chunk_length_chart(sources)
                        if fig:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"chart_chunklens_{msg['id']}"
                            )
                    with v6:
                        fig = create_answer_word_cloud_data(msg["content"])
                        if fig:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"chart_wordcloud_{msg['id']}"
                            )

                    # Source Text Highlights
                    if sources:
                        st.markdown("**📝 Source Passages:**")
                        for idx, src in enumerate(sources[:5]):
                            badge = ("🔵 Dense" if src.get("method") == "dense"
                                     else "🟠 Sparse" if src.get("method") == "sparse"
                                     else "🟢 Hybrid")
                            st.markdown(f"**Source {idx+1}** [{src.get('section', '?')}] "
                                        f"p.{src.get('page_numbers', '?')} | "
                                        f"Score: {src.get('score', 0):.3f} | {badge}")
                            full_text = src.get("text") or src.get("text_preview", "")
                            st.caption(full_text)

                    # PDF Page Viewer
                    if sources and active_pdfs:
                        st.markdown("**📄 Referenced Pages:**")
                        # Let user pick which PDF to view
                        pdf_names = [p["filename"] for p in active_pdfs]
                        selected_pdf_name = st.selectbox(
                            "View PDF:", pdf_names,
                            key=f"pdfsel_{msg['id']}"
                        )
                        selected_pdf = next(
                            (p for p in active_pdfs if p["filename"] == selected_pdf_name), None
                        )
                        if selected_pdf and os.path.exists(selected_pdf["filepath"]):
                            for src in sources[:2]:
                                pages = src.get("page_numbers", [])
                                if pages:
                                    pnum = pages[0] - 1
                                    # If the source section is tagged like "[file.pdf] ...",
                                    # prefer rendering/highlighting against that PDF.
                                    pdf_for_source = selected_pdf
                                    sec = (src.get("section") or "").strip()
                                    if sec.startswith("[") and "]" in sec:
                                        tagged_name = sec[1:sec.index("]")].strip()
                                        match = next(
                                            (p for p in active_pdfs if p.get("filename") == tagged_name),
                                            None,
                                        )
                                        if match and os.path.exists(match.get("filepath", "")):
                                            pdf_for_source = match

                                    highlight_text = src.get("text") or src.get("text_preview", "")
                                    img = render_pdf_page(
                                        pdf_for_source["filepath"],
                                        pnum,
                                        highlight_texts=[highlight_text] if highlight_text else None,
                                    )
                                    if img:
                                        st.image(img,
                                                 caption=f"Page {pages[0]} — {src.get('section', '')}",
                                                 width=550)
                                    break

    # ─── Sticky Composer (Prompt + Settings) ───
    pending = st.session_state.pop("pending_question", None)

    # If a suggested question was clicked, prefill the input.
    prompt_key = f"prompt_{session_id}"
    if pending:
        st.session_state[prompt_key] = pending

    with st.container(key="composer"):
        with st.form(key=f"composer_form_{session_id}", clear_on_submit=True):
            prompt_text = st.text_input(
                "Ask about your research papers...",
                key=prompt_key,
                label_visibility="collapsed",
                placeholder="Ask about your research papers...",
            )

            c1, c2 = st.columns(2)
            with c1:
                leniency = st.slider(
                    "🎚️ Leniency",
                    0,
                    100,
                    int(session.get("leniency", 50)),
                    help="0=strict, 100=lenient",
                    key=f"leniency_{session_id}",
                )
            with c2:
                top_k = st.slider(
                    "🔍 Top-K",
                    1,
                    10,
                    int(session.get("top_k", 5)),
                    help="Chunks to retrieve",
                    key=f"topk_{session_id}",
                )

            sent = st.form_submit_button("Send")

    # Avoid updating `updated_at` (and reordering sidebar sessions) unless settings actually changed.
    if (leniency != session.get("leniency", 50)) or (top_k != session.get("top_k", 5)):
        update_session_settings(session_id, leniency, top_k)

    if sent:
        prompt = (prompt_text or "").strip()
    elif pending:
        # Preserve the old behavior: suggested questions send immediately.
        prompt = (pending or "").strip()
        st.session_state[prompt_key] = ""
    else:
        prompt = None
    if prompt:
        if not active_pdfs:
            st.warning("📄 Upload a PDF first!")
        else:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            add_message(session_id, "user", prompt)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🧠 Thinking..."):
                    bot = get_bot()
                    response = bot.ask(prompt, top_k=top_k, leniency=leniency)
                st.markdown(response.answer)

                sources_data = format_sources_for_db(response.sources)
                methods = list(set(getattr(s, "retrieval_method", "?") for s in response.sources))

                add_message(
                    session_id,
                    "assistant",
                    response.answer,
                    confidence=response.confidence,
                    generation_time=response.generation_time,
                    sources=sources_data,
                    retrieval_methods=", ".join(methods),
                )

            if len(get_messages(session_id)) <= 2:
                update_session_title(session_id, prompt[:50] + ("..." if len(prompt) > 50 else ""))

            st.rerun()
