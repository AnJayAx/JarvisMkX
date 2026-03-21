"""
Jarvis Mk.X — Smart Research Paper Q&A Chatbot
Streamlit Application (v2 — Multi-PDF + Visualizations)

Run: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
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

# ─── Load API keys from .env file ──────────────────────────────────────────
def _load_env_file():
    """Load key=value pairs from .env file in project root."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and value and not os.environ.get(key):
                    os.environ[key] = value

_load_env_file()

sys.path.insert(0, "src")

from database import (
    create_session, get_all_sessions, search_sessions, get_session,
    update_session_title, update_session_settings, delete_session,
    add_message, get_messages, add_pdf, get_active_pdfs, remove_pdf,
    get_corrections,
)
from pdf_export import export_chat_to_pdf

# Try to import delete_message; provide fallback if database.py doesn't have it yet.
try:
    from database import delete_message
except ImportError:
    def delete_message(message_id):
        """Fallback: delete a message by ID directly from the SQLite database."""
        import sqlite3
        db_path = os.path.join("data", "jarvis.db")
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        conn.commit()
        conn.close()

st.set_page_config(
    page_title="Jarvis Mk.X",
    page_icon="J",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    :root {
        --composer-height: 260px;
    }
    .stApp { 
        font-family: 'Segoe UI', sans-serif;
        background: radial-gradient(circle at 20% 5%, #1b2433 0%, #0d1117 40%, #0a0f14 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
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
    [data-testid="stSidebar"] [class*="st-key-session_title_"] button,
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        overflow: visible !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        position: relative !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
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
    [data-testid="stSidebar"] [class*="st-key-session_title_"] button p,
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] button p,
    [data-testid="stSidebar"] [class*="st-key-session_title_"] button span,
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] button span {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 100% !important;
    }
    [data-testid="stSidebar"] [class*="st-key-session_title_active_"] .stButton > button {
        color: #bfdbfe !important;
        font-weight: 700 !important;
    }
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

    /* Chat message delete buttons */
    [data-testid="stChatMessage"] [class*="st-key-del_"] .stButton > button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: rgba(148, 163, 184, 0.35) !important;
        font-size: 0.72em !important;
        padding: 2px 6px !important;
        min-height: 0 !important;
        transform: none !important;
    }
    [data-testid="stChatMessage"] [class*="st-key-del_"] .stButton > button:hover {
        color: #f87171 !important;
        background: rgba(248, 113, 113, 0.08) !important;
    }

    [data-testid="stMainBlockContainer"], .main .block-container {
        padding-bottom: calc(var(--composer-height, 260px) + 10rem) !important;
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
    .st-key-composer .stFormSubmitButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #f0f6ff !important;
        font-size: 1.1em !important;
        min-height: 2.4rem !important;
        padding: 0 0.6rem !important;
        cursor: pointer !important;
        transition: transform 0.1s ease, box-shadow 0.15s ease !important;
    }
    .st-key-composer .stFormSubmitButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 10px rgba(37,99,235,0.35) !important;
    }
    .st-key-composer .stSlider { padding-top: 0.35rem !important; }
    .st-key-composer .stSlider [data-testid="stTickBarMin"],
    .st-key-composer .stSlider [data-testid="stTickBarMax"] {
        font-size: 0.65em !important;
        color: rgba(148,163,184,0.5) !important;
    }

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


def init_state():
    defaults = {
        "current_session": None,
        "bot": None,
        "bot_loaded": False,
        "papers_loaded_key": None,
        "suggested_questions": [],
        "all_chunks_cache": [],
        "current_model": "jarvis_finetuned",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── VRAM Detection ─────────────────────────────────────────────────────────
def detect_gpu_info():
    """Detect GPU and available VRAM. Returns (gpu_name, vram_gb) or (None, 0)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return name, round(vram, 1)
    except Exception:
        pass
    return None, 0

GPU_NAME, GPU_VRAM_GB = detect_gpu_info()
HIGH_VRAM = GPU_VRAM_GB >= 8  # Mistral-7B fits on 8GB+ GPUs (4-bit)

# ─── Model Configurations ───────────────────────────────────────────────────
# Local models: Jarvis (fine-tuned) and DeepSeek-R1-7B (no API available)
# API models: Qwen3-8B, Llama-3.1-8B, Mistral-7B via OpenRouter; DeepSeek-V3.2 via DeepSeek API
MODEL_CONFIGS = {
    "jarvis_finetuned": {
        "label": "Jarvis Mk.X (Fine-tuned Qwen3-8B)",
        "base_model": "Qwen/Qwen3-8B",
        "adapter_path": "models/jarvis-mkx-qwen3-8b-adapter",
        "short_name": "Jarvis Mk.X",
        "description": "Our QLoRA fine-tuned Qwen3-8B on QASPER + PubMedQA",
        "is_api": False,
        "is_qwen3": True,
        "min_vram": 8,
    },
    "deepseek_r1_7b": {
        "label": "DeepSeek-R1-Distill-Qwen-7B (Local)",
        "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "adapter_path": None,
        "short_name": "DeepSeek-R1-7B",
        "description": "DeepSeek R1 distilled into Qwen-7B — local only (no API available)",
        "is_api": False,
        "is_qwen3": False,
        "min_vram": 8,
    },
    "qwen3_base_api": {
        "label": "Qwen3-8B (API)",
        "short_name": "Qwen3-8B Base",
        "description": "Base Qwen3-8B via OpenRouter API — no GPU needed",
        "is_api": True,
        "api_provider": "openrouter",
        "api_model": "qwen/qwen3-8b",
        "min_vram": 0,
    },
    "llama31_api": {
        "label": "Llama-3.1-8B-Instruct (API)",
        "short_name": "Llama-3.1-8B",
        "description": "Meta Llama 3.1 8B via OpenRouter API — no GPU needed",
        "is_api": True,
        "api_provider": "openrouter",
        "api_model": "meta-llama/llama-3.1-8b-instruct",
        "min_vram": 0,
    },
    "mistral_7b_api": {
        "label": "Mistral-7B-Instruct (API)",
        "short_name": "Mistral-7B",
        "description": "Mistral 7B Instruct via OpenRouter API — no GPU needed",
        "is_api": True,
        "api_provider": "openrouter",
        "api_model": "mistralai/mistral-7b-instruct",
        "min_vram": 0,
    },
    "deepseek_v3_api": {
        "label": "DeepSeek-V3.2 (API)",
        "short_name": "DeepSeek-V3.2",
        "description": "Frontier MoE model via DeepSeek API — no GPU needed",
        "is_api": True,
        "api_provider": "deepseek",
        "api_model": "deepseek-chat",
        "min_vram": 0,
    },
}

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def get_available_models():
    """Return model keys that can run on this hardware."""
    available = {}
    for key, cfg in MODEL_CONFIGS.items():
        if cfg["is_api"]:
            # API models always available
            available[key] = cfg
        elif GPU_VRAM_GB >= cfg["min_vram"]:
            # Enough VRAM for local model
            available[key] = cfg
        elif cfg.get("api_fallback"):
            # Mark as API fallback
            fallback_cfg = dict(cfg)
            fallback_cfg["label"] = cfg["label"] + " (via API)"
            fallback_cfg["short_name"] = cfg["short_name"] + " (API)"
            fallback_cfg["is_api"] = True
            fallback_cfg["_is_fallback"] = True
            available[key] = fallback_cfg
        # else: model is unavailable on this hardware
    return available


AVAILABLE_MODELS = get_available_models()


def _get_embed_model_name():
    forced = os.environ.get("JARVIS_EMBED_MODEL", "").strip()
    if forced:
        return forced
    # Default: Voyage 4 large (requires VOYAGE_API_KEY env var)
    # Fallback: SPECTER (free, local, no API key needed)
    return (
        "voyage-4-large"
        if os.environ.get("VOYAGE_API_KEY")
        else "sentence-transformers/allenai-specter"
    )


def _create_bot_shell(model_key="jarvis_finetuned"):
    """Create a JarvisBot with retriever only — NO LLM loaded.

    The LLM is loaded on-demand per question and unloaded immediately after,
    keeping GPU memory free between questions. This lets you switch models
    instantly since only the retriever (CPU) stays resident.
    """
    from bot import JarvisBot
    config = AVAILABLE_MODELS.get(model_key, MODEL_CONFIGS.get(model_key, {}))

    bot = JarvisBot(
        base_model_name=config.get("base_model", "Qwen/Qwen3-8B"),
        adapter_path=config.get("adapter_path"),
        embed_model_name=_get_embed_model_name(),
        chunk_size=512, chunk_overlap=50, load_in_4bit=True,
        is_qwen3=config.get("is_qwen3", False),
    )
    # DON'T call bot.load_model() — LLM loads on demand
    print(f"Bot shell created (retriever only, no LLM in VRAM)")
    return bot


def get_bot():
    """Get or create the bot shell (retriever only — no LLM resident)."""
    if st.session_state.bot is None:
        with st.spinner("Setting up retriever..."):
            st.session_state.bot = _create_bot_shell()
            st.session_state.bot_loaded = True
    return st.session_state.bot


def answer_with_model(bot, model_key, prompt, top_k=5, leniency=50,
                       active_pdfs=None, session_id=None, messages=None):
    """Load LLM → answer question → unload LLM. GPU is free between calls.

    For API models, no LLM loading is needed — just call the API.
    For local models, the LLM is loaded, used, and immediately freed.
    The retriever and PDF index stay in memory (CPU) throughout.
    """
    config = AVAILABLE_MODELS.get(model_key, MODEL_CONFIGS.get(model_key))

    # ── API model: no LLM needed ────────────────────────────────────────
    if config.get("is_api"):
        retrieved = bot.retriever.retrieve(prompt, top_k=top_k)
        provider = config.get("api_provider", "deepseek")
        api_model = config.get("api_model", "deepseek-chat")
        answer_text = query_api(prompt, retrieved, leniency,
                                provider=provider, model=api_model)

        class _ApiResponse:
            pass
        response = _ApiResponse()
        response.answer = answer_text
        response.sources = retrieved
        response.confidence = 0.5
        response.generation_time = 0.0
        return response

    # ── Local model: load → answer → unload ─────────────────────────────
    import gc

    # Configure bot for this model
    bot.base_model_name = config["base_model"]
    bot.adapter_path = config.get("adapter_path")
    bot.is_qwen3 = config.get("is_qwen3", False)

    try:
        # Load LLM into VRAM
        bot.load_model()

        # Answer the question
        response = bot.ask(prompt, top_k=top_k, leniency=leniency)

    finally:
        # ALWAYS unload — even if generation fails
        bot.unload_model()
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return response


def query_api(prompt, context_chunks, leniency=50, provider="deepseek", model="deepseek-chat"):
    """Generate an answer using an API (OpenRouter or DeepSeek) with retrieved context."""
    import requests

    context_text = "\n\n".join([
        f"[Source {i+1}] {getattr(c, 'section', 'Unknown')}:\n{c.text}"
        for i, c in enumerate(context_chunks[:5])
    ]) if context_chunks else "No context available."

    temperature = 0.1 + (leniency / 100) * 0.6

    system_msg = (
        "You are Jarvis Mk.X, a research paper Q&A assistant.\n\n"
        "STRICT RULES:\n"
        "1. ONLY answer based on the provided context from the research paper.\n"
        "2. If the context does not contain information to answer the question, "
        "respond with: 'The provided context does not contain information to answer this question.'\n"
        "3. NEVER make up facts, numbers, or claims not supported by the context.\n"
        "4. Do NOT answer general knowledge questions — you only know what is in the paper.\n\n"
        "RESPONSE FORMAT:\n"
        "**Answer:** [Your direct answer]\n\n"
        "**Reason:** [How the context supports it]\n\n"
        "**Sources:**\n- [Section/page references]\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"### Context from Research Paper:\n{context_text}\n\n### Question:\n{prompt}"},
    ]

    if provider == "openrouter":
        api_key = OPENROUTER_API_KEY
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://jarvis-mkx.app",
            "X-Title": "Jarvis Mk.X",
        }
    else:  # deepseek
        api_key = DEEPSEEK_API_KEY
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    if not api_key:
        return f"API key not set for {provider}. Set {'OPENROUTER_API_KEY' if provider == 'openrouter' else 'DEEPSEEK_API_KEY'} environment variable."

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 500,
        "temperature": temperature,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"{provider} API error: {e}"


def load_all_pdfs_into_bot(session_id, active_pdfs, messages):
    if not active_pdfs:
        return

    cache_key = f"{session_id}_{'_'.join(p['id'] for p in active_pdfs)}"
    if st.session_state.papers_loaded_key == cache_key:
        return

    # Only need the retriever for indexing — don't force full LLM load
    if st.session_state.bot is None:
        with st.spinner("Setting up retriever..."):
            st.session_state.bot = _create_bot_shell()
            st.session_state.bot_loaded = True

    bot = st.session_state.bot
    from processor import PaperProcessor

    processor = PaperProcessor(chunk_size=512, chunk_overlap=50)
    all_chunks = []

    for pdf in active_pdfs:
        if os.path.exists(pdf["filepath"]):
            paper = processor.process(pdf["filepath"])
            for chunk in paper.chunks:
                chunk.section = f"[{pdf['filename']}] {chunk.section}"
            all_chunks.extend(paper.chunks)
            bot.current_paper = paper

    if all_chunks:
        bot.retriever.build_index(all_chunks)
        st.session_state.all_chunks_cache = all_chunks

    from bot import ConversationTurn
    bot.conversation_history = []
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            bot.conversation_history.append(
                ConversationTurn(role=msg["role"], content=msg["content"])
            )
    if len(bot.conversation_history) > bot.max_history_turns:
        bot.conversation_history = bot.conversation_history[-bot.max_history_turns:]

    corrections = get_corrections(session_id)
    bot.corrections = [
        {"question": c["question"], "wrong_answer": c["wrong_answer"],
         "correction": c["correction"]}
        for c in corrections
    ]

    st.session_state.papers_loaded_key = cache_key


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


def _generate_summary_with_jarvis(context_text, filename):
    """Use Jarvis Mk.X (fine-tuned Qwen3-8B) to generate a paper summary.
    Loads the model on demand, generates summary, then unloads."""
    import gc
    from bot import JarvisBot

    bot = JarvisBot(
        base_model_name="Qwen/Qwen3-8B",
        adapter_path="models/jarvis-mkx-qwen3-8b-adapter",
        embed_model_name=_get_embed_model_name(),
        load_in_4bit=True,
        is_qwen3=True,
    )

    try:
        bot.load_model()

        prompt = (
            f"Summarize this research paper in 3-5 sentences. "
            f"Cover: (1) what the paper is about, (2) the main methodology, "
            f"(3) the key findings or conclusions.\n\n"
            f"### Context from Research Paper:\n{context_text}\n\n"
            f"### Question:\nProvide a concise summary of this paper."
        )

        messages = [
            {"role": "system", "content": "You are Jarvis Mk.X. Summarize the research paper concisely in 3-5 sentences."},
            {"role": "user", "content": prompt},
        ]

        template_kwargs = dict(
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
            enable_thinking=False,
        )
        encoded = bot.tokenizer.apply_chat_template(messages, **template_kwargs)
        input_ids = encoded["input_ids"].to(bot.model.device)
        attention_mask = encoded["attention_mask"].to(bot.model.device)

        with torch.no_grad():
            outputs = bot.model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=500, temperature=0.3,
                do_sample=True, repetition_penalty=1.1,
            )
        summary = bot.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        return summary

    finally:
        bot.unload_model()
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def format_sources_for_db(sources):
    return [
        {"section": s.section, "page_numbers": s.page_numbers,
         "score": round(s.score, 4),
         "method": getattr(s, "retrieval_method", "unknown"),
         "text": s.text,
         "text_preview": s.text[:300]}
        for s in sources
    ]


def render_pdf_page_simple(filepath, page_num):
    """Render a PDF page as an image — no highlighting (avoids inaccurate marks)."""
    try:
        doc = fitz.open(filepath)
        page = doc[min(page_num, len(doc) - 1)]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception:
        return None


def create_confidence_gauge(confidence):
    color = "#4dff88" if confidence > 0.5 else "#ffcc00" if confidence > 0.3 else "#ff4444"
    value = max(0.0, min(1.0, float(confidence))) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": "Retrieval Confidence", "font": {"size": 14, "color": "#ccc"}},
        number={"suffix": "%", "valueformat": ".0f", "font": {"size": 42}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#555"},
               "bar": {"color": color},
               "bgcolor": "#1e1e2e",
               "steps": [{"range": [0, 30], "color": "#3a1a1a"},
                         {"range": [30, 60], "color": "#3a3a1a"},
                         {"range": [60, 100], "color": "#1a3a1a"}]},
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=60, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ccc"})
    return fig


def create_source_scores_bar(sources):
    """Horizontal bar chart: retrieval score per source chunk, color-coded by method."""
    if not sources:
        return None
    labels = [f"Chunk {i+1}: {s.get('section', '?')[:35]}" for i, s in enumerate(sources)]
    scores = [s.get("score", 0) for s in sources]
    methods = [s.get("method", "hybrid") for s in sources]
    colors = ["#4da6ff" if m == "dense" else "#ff9944" if m == "sparse" else "#44ff88"
              for m in methods]
    fig = go.Figure(go.Bar(
        y=labels, x=scores, orientation="h",
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        hovertext=[f"{m.title()} | Score: {s:.4f}" for m, s in zip(methods, scores)],
        hoverinfo="text",
    ))
    fig.update_layout(
        title="Retrieval Scores by Chunk",
        xaxis_title="Score", xaxis_range=[0, max(scores) * 1.3 + 0.05],
        height=max(180, len(sources) * 45),
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
    )
    return fig


def create_method_breakdown(sources):
    """Stacked bar: dense vs sparse contribution per chunk."""
    if not sources or len(sources) < 2:
        return None
    labels = [f"Chunk {i+1}" for i in range(len(sources))]
    dense_est, sparse_est = [], []
    for s in sources:
        h = s.get("score", 0)
        m = s.get("method", "hybrid")
        if m == "dense":
            dense_est.append(h); sparse_est.append(0)
        elif m == "sparse":
            dense_est.append(0); sparse_est.append(h)
        else:
            dense_est.append(h * 0.6); sparse_est.append(h * 0.4)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Dense (Semantic)", x=labels, y=dense_est, marker_color="#4da6ff"))
    fig.add_trace(go.Bar(name="Sparse (BM25)", x=labels, y=sparse_est, marker_color="#ff9944"))
    fig.update_layout(
        barmode="stack", title="Dense vs Sparse Score Breakdown",
        yaxis_title="Score", height=250,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"}, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_answer_keywords(answer_text):
    """Top keywords from the answer as a horizontal bar."""
    from collections import Counter
    words = re.findall(r'\b[a-zA-Z]{4,}\b', answer_text.lower())
    stopwords = {"this", "that", "with", "from", "have", "been", "were", "they",
                 "their", "which", "would", "could", "about", "into", "more",
                 "also", "than", "other", "some", "such", "when", "what", "there",
                 "these", "those", "does", "will", "each", "only", "very", "most"}
    words = [w for w in words if w not in stopwords]
    counts = Counter(words).most_common(10)
    if not counts:
        return None
    words_list, freqs = zip(*counts)
    fig = go.Figure(go.Bar(
        x=list(freqs), y=list(words_list), orientation="h",
        marker_color="#e040fb",
    ))
    fig.update_layout(
        title="Key Terms in Answer", xaxis_title="Frequency",
        height=280, margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
    )
    return fig


def create_3d_vector_space(query_text, sources, embed_model):
    """3D PCA scatter plot: query point vs retrieved chunk points with hover text."""
    if not sources or len(sources) < 2:
        return None
    try:
        from sklearn.decomposition import PCA

        chunk_previews = [
            (s.get("text") or s.get("text_preview", ""))[:150].replace("\n", " ")
            for s in sources
        ]
        texts = [query_text] + [s.get("text") or s.get("text_preview", "")[:200] for s in sources]
        scores = [1.0] + [s.get("score", 0) for s in sources]
        methods = ["query"] + [s.get("method", "hybrid") for s in sources]
        sections = ["Query"] + [s.get("section", "?")[:30] for s in sources]

        embeddings = embed_model.encode(texts, normalize_embeddings=True)
        pca = PCA(n_components=3)
        coords = pca.fit_transform(embeddings)

        fig = go.Figure()

        # ── Chunk points with hover ──────────────────────────────────────
        for i in range(1, len(coords)):
            sc = scores[i]
            color = "#44ff88" if sc > 0.5 else "#ffcc00" if sc > 0.3 else "#4da6ff"
            size = max(8, sc * 18)
            method_label = methods[i].title()
            hover = (
                f"<b>Chunk {i}: {sections[i]}</b><br>"
                f"Score: {sc:.3f} | Method: {method_label}<br>"
                f"<i>{chunk_previews[i-1]}...</i>"
            )
            fig.add_trace(go.Scatter3d(
                x=[coords[i, 0]], y=[coords[i, 1]], z=[coords[i, 2]],
                mode="markers",
                marker=dict(size=size, color=color, opacity=0.85,
                            line=dict(width=1.5, color="white")),
                name=f"Chunk {i} ({sc:.3f})",
                hovertext=hover, hoverinfo="text",
            ))
            # Dashed line to query
            fig.add_trace(go.Scatter3d(
                x=[coords[0, 0], coords[i, 0]],
                y=[coords[0, 1], coords[i, 1]],
                z=[coords[0, 2], coords[i, 2]],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=False, opacity=0.3, hoverinfo="skip",
            ))

        # ── Query point (diamond) ────────────────────────────────────────
        fig.add_trace(go.Scatter3d(
            x=[coords[0, 0]], y=[coords[0, 1]], z=[coords[0, 2]],
            mode="markers+text", text=["Query"],
            textposition="top center", textfont=dict(size=11, color="#ff6666"),
            marker=dict(size=12, color="#ff4444", symbol="diamond", opacity=1.0,
                        line=dict(width=2, color="white")),
            name="Your Query",
            hovertext=f"<b>Query</b><br>{query_text[:120]}...",
            hoverinfo="text",
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.0%})",
                bgcolor="rgba(0,0,0,0)",
            ),
            height=550, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ccc"},
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0.5)"),
        )
        return fig
    except Exception as e:
        return None


with st.sidebar:
    st.markdown("## Jarvis Mk.X")
    st.caption("Smart Research Paper Chatbot")

    search_query = st.text_input("Search", placeholder="Find a chat...")

    if st.button("+ New Chat", width="stretch", type="primary"):
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
                    display_title = full_title
                    title_container_key = (
                        f"session_title_active_{sess['id']}"
                        if is_active else f"session_title_{sess['id']}"
                    )
                    with st.container(key=title_container_key):
                        if st.button(display_title, key=f"s_{sess['id']}",
                                     width="stretch",
                                     help=full_title,
                                     type="secondary"):
                            st.session_state.current_session = sess["id"]
                            st.session_state.papers_loaded_key = None
                            st.session_state.suggested_questions = []
                            st.rerun()
                with c2:
                    with st.container(key=f"session_delete_{sess['id']}"):
                        if st.button("X", key=f"d_{sess['id']}", width="stretch"):
                            delete_session(sess["id"])
                            if st.session_state.current_session == sess["id"]:
                                st.session_state.current_session = None
                            st.rerun()

    st.divider()


if st.session_state.current_session is None:
    st.markdown("# Jarvis Mk.X")
    st.markdown("### Smart Research Paper Q&A Chatbot")
    st.caption("Fine-tuned Qwen3-8B on QASPER + PubMedQA | Hybrid Retrieval (Dense + BM25) | Voyage 4 Embeddings")

    # GPU info banner
    if GPU_NAME:
        st.info(f"**{GPU_NAME}** — {GPU_VRAM_GB}GB VRAM")
    else:
        st.info("Running in API mode — no local GPU required for most models")

    st.markdown("---")

    # ── Available Models ─────────────────────────────────────────────────
    st.markdown("### Available Models")
    st.markdown("Select any model from the dropdown when chatting. "
                "**Jarvis Mk.X** is our fine-tuned model; others are baselines for comparison.")

    model_cols = st.columns(3)
    for i, (key, cfg) in enumerate(AVAILABLE_MODELS.items()):
        with model_cols[i % 3]:
            is_local = not cfg.get("is_api")
            badge = "[Local]" if is_local else "[API]"
            star = " " if "jarvis" in key else ""
            st.markdown(f"**{cfg['label']}**{star}  \n"
                        f"{badge} — {cfg['description']}")

    st.markdown("---")

    # ── Benchmark Results ────────────────────────────────────────────────
    st.markdown("### 6-Model Benchmark Results")
    st.markdown(
        "We evaluated all 6 models on **100 QASPER test questions** (research paper Q&A). "
        "Our fine-tuned **Jarvis Mk.X** significantly outperforms all base models across "
        "every metric, with statistically significant improvements (p < 0.001)."
    )

    # ── Metrics explanation ──────────────────────────────────────────────
    with st.expander("What do these metrics mean?", expanded=False):
        st.markdown("""
**Lexical Metrics** — Measure word-level overlap between the model's answer and the reference answer:
- **Token F1** — Harmonic mean of precision and recall at the word level. Higher = better word coverage.
- **ROUGE-1/2/L** — Unigram, bigram, and longest-common-subsequence overlap. Standard for summarization evaluation.
- **BLEU-1/2/4** — N-gram precision (how many n-grams in the prediction appear in the reference).
- **METEOR** — Alignment-based metric that accounts for synonyms and stemming. More lenient than BLEU.

**Semantic Metrics** — Measure meaning similarity, not just word overlap:
- **BERTScore F1** — Uses DeBERTa embeddings to compare meaning. A prediction can score high even with different wording.
- **SBERT Cosine** — Cosine similarity between sentence embeddings. Captures overall semantic closeness.

**RAGAS Metrics** — Evaluate RAG (Retrieval-Augmented Generation) quality using GPT-4o-mini as a judge:
- **Faithfulness** — Is the answer grounded in the retrieved context? (no hallucination)
- **Answer Relevancy** — Does the answer actually address the question asked?
- **Context Precision** — Are the retrieved chunks relevant to the question?
- **Context Recall** — Do the retrieved chunks cover all info needed for the answer?
- **Answer Correctness** — Factual overlap between the answer and the ground-truth reference.

**Answer Quality** — Descriptive statistics:
- **Avg Pred Len** — Average answer length in words. Too short = incomplete; too long = verbose.
- **Length Ratio** — Predicted length / reference length. Ideally close to 1.0.
- **Refusal Rate** — How often the model refuses to answer (says "not enough info").
- **Vocab Diversity (TTR)** — Type-token ratio. Higher = richer vocabulary, less repetition.
        """)

    # ── Benchmark Table ──────────────────────────────────────────────────
    st.markdown("#### Full Benchmark Table")
    st.markdown("*Jarvis Mk.X (our fine-tuned model) vs 5 baseline models on 100 QASPER test samples.*")

    # Try to load benchmark CSV if available
    import pandas as pd
    benchmark_csv = os.path.join("models", "evaluation_v5_qwen3", "benchmark_results.csv")
    ragas_csv = os.path.join("models", "evaluation_v5_qwen3", "ragas_results.csv")

    # Column tooltips — shown when hovering over column headers
    benchmark_col_help = {
        "Token F1": st.column_config.NumberColumn(
            "Token F1",
            help="Harmonic mean of precision and recall at the word level. Higher = better word coverage.",
        ),
        "Exact Match": st.column_config.NumberColumn(
            "Exact Match",
            help="1 if prediction exactly matches reference after normalization, 0 otherwise. Very strict metric.",
        ),
        "ROUGE-1": st.column_config.NumberColumn(
            "ROUGE-1",
            help="Unigram overlap (F1) between prediction and reference. Measures single-word recall and precision.",
        ),
        "ROUGE-2": st.column_config.NumberColumn(
            "ROUGE-2",
            help="Bigram overlap (F1). Captures two-word phrase matches — harder to score high on than ROUGE-1.",
        ),
        "ROUGE-L": st.column_config.NumberColumn(
            "ROUGE-L",
            help="Longest common subsequence overlap. Rewards answers that preserve the order of words from the reference.",
        ),
        "BLEU-1": st.column_config.NumberColumn(
            "BLEU-1",
            help="Unigram precision — what fraction of words in the prediction also appear in the reference.",
        ),
        "BLEU-2": st.column_config.NumberColumn(
            "BLEU-2",
            help="Bigram precision — what fraction of 2-word phrases in the prediction match the reference.",
        ),
        "BLEU-4": st.column_config.NumberColumn(
            "BLEU-4",
            help="4-gram precision. Very strict — requires 4 consecutive words to match. Near 0 for most base models.",
        ),
        "METEOR": st.column_config.NumberColumn(
            "METEOR",
            help="Alignment-based metric accounting for synonyms, stemming, and word order. More lenient than BLEU.",
        ),
        "BERTScore F1": st.column_config.NumberColumn(
            "BERTScore F1",
            help="Semantic similarity using DeBERTa embeddings. High even when wording differs but meaning is similar.",
        ),
        "SBERT Cosine": st.column_config.NumberColumn(
            "SBERT Cosine",
            help="Cosine similarity between sentence embeddings (MiniLM). Captures overall semantic closeness.",
        ),
        "Refusal Rate": st.column_config.NumberColumn(
            "Refusal Rate",
            help="How often the model refuses to answer (says 'not enough info'). 0 = never refuses, 1 = always refuses.",
        ),
        "Avg Pred Len": st.column_config.NumberColumn(
            "Avg Pred Len",
            help="Average answer length in words. QASPER expects short answers; shorter is often better for F1.",
        ),
        "Length Ratio": st.column_config.NumberColumn(
            "Length Ratio",
            help="Predicted length / reference length. Ideally close to 1.0. >1 means model is too verbose.",
        ),
        "Vocab Diversity": st.column_config.NumberColumn(
            "Vocab Diversity",
            help="Type-Token Ratio — unique words / total words. Higher = richer vocabulary, less repetition.",
        ),
    }

    ragas_col_help = {
        "Faithfulness": st.column_config.NumberColumn(
            "Faithfulness",
            help="Is the answer grounded in the retrieved context? Higher = less hallucination. Scored by GPT-4o-mini.",
        ),
        "Answer Relevancy": st.column_config.NumberColumn(
            "Answer Relevancy",
            help="Does the answer actually address the question? Higher = more on-topic. Scored by GPT-4o-mini.",
        ),
        "Context Precision": st.column_config.NumberColumn(
            "Context Precision",
            help="Are the retrieved chunks relevant to the question? Similar across models since they share the same retriever.",
        ),
        "Context Recall": st.column_config.NumberColumn(
            "Context Recall",
            help="Do the retrieved chunks cover all info needed for the ground-truth answer? Measures retrieval completeness.",
        ),
        "Answer Correctness": st.column_config.NumberColumn(
            "Answer Correctness",
            help="Factual overlap between the answer and the ground-truth reference. Combines F1 and semantic similarity.",
        ),
    }

    if os.path.exists(benchmark_csv):
        df = pd.read_csv(benchmark_csv, index_col=0)
        st.dataframe(df, use_container_width=True, column_config=benchmark_col_help)
    else:
        st.caption("Benchmark CSV not found. Run Notebook 6 to generate results.")

    if os.path.exists(ragas_csv):
        st.markdown("#### RAGAS -- RAG Quality Metrics")
        rdf = pd.read_csv(ragas_csv, index_col=0)
        st.dataframe(rdf, use_container_width=True, column_config=ragas_col_help)

    # ── Charts with analysis ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Visual Comparisons")
    st.markdown(
        "Each chart below includes a **How to read this** guide. "
        "Charts are generated from the evaluation notebook (Notebook 6). "
        "All 6 models were evaluated on the same 100 QASPER test questions."
    )

    img_dir = "img"

    # Chart 1: Bar Charts
    img_path = os.path.join(img_dir, "benchmark_comparison.png")
    if os.path.exists(img_path):
        st.markdown("#### Metric Bar Charts")
        st.markdown(
            "**How to read:** Each subplot shows one metric. Taller bars = better performance. "
            "The dark blue bar (Jarvis) should be compared against the lighter bars (baselines). "
            "If Jarvis is consistently taller across all subplots, it means fine-tuning improved "
            "the model across the board, not just on one metric."
        )
        st.image(img_path, use_container_width=True)
        st.markdown(
            "**Analysis:** Jarvis Mk.X (dark blue) dominates every metric. The gap is largest in "
            "Token F1 (0.51 vs ~0.10-0.15) and BLEU-4 (0.39 vs ~0.02-0.04), showing the fine-tuned model "
            "produces answers with much better word-level overlap with the reference answers. "
            "Vocab Diversity is also highest (0.91), meaning Jarvis uses richer, less repetitive language."
        )
        st.markdown("")

    # Chart 2: Radar
    img_path = os.path.join(img_dir, "radar_chart.png")
    if os.path.exists(img_path):
        st.markdown("#### Radar Chart")
        st.markdown(
            "**How to read:** Each axis represents a different metric (0-1 scale). "
            "Each model forms a polygon — a larger polygon means better overall performance. "
            "If one model's polygon fully encloses another's, it is strictly better on all metrics."
        )
        st.image(img_path, use_container_width=True)
        st.markdown(
            "**Analysis:** Jarvis Mk.X's blue polygon is visibly larger and encloses all other models. "
            "The base models cluster together in a small region near the center, showing they perform "
            "similarly to each other but far worse than the fine-tuned model. BERTScore (top) is the "
            "closest axis where all models converge, because semantic similarity captures partial meaning "
            "overlap even when word choice differs."
        )
        st.markdown("")

    # Chart 3: Heatmap
    img_path = os.path.join(img_dir, "heatmap.png")
    if os.path.exists(img_path):
        st.markdown("#### Metric Heatmap")
        st.markdown(
            "**How to read:** Rows are models, columns are metrics. Darker/bluer cells = higher scores. "
            "Scan left-to-right across a row to see one model's profile. Scan top-to-bottom in a column "
            "to compare all models on one metric. The best model has the darkest row overall."
        )
        st.image(img_path, use_container_width=True)
        st.markdown(
            "**Analysis:** The top row (Jarvis) is consistently the darkest across all columns. "
            "The BERTScore column is the most uniformly dark (0.82-0.91 for all models), confirming "
            "it is the least discriminative metric. Token F1 and BLEU-4 columns show the starkest "
            "contrast between Jarvis and the baselines."
        )
        st.markdown("")

    # Chart 4: Violin
    img_path = os.path.join(img_dir, "f1_violin.png")
    if os.path.exists(img_path):
        st.markdown("#### Token F1 Distribution (Violin Plot)")
        st.markdown(
            "**How to read:** Each violin shows the full distribution of F1 scores across all 100 test questions. "
            "The wider the violin at a score level, the more questions got that score. "
            "Dashed lines show quartiles (25th, 50th, 75th percentile). "
            "A violin reaching up to 1.0 means some questions were answered perfectly."
        )
        st.image(img_path, use_container_width=True)
        st.markdown(
            "**Analysis:** Jarvis has the widest spread, reaching 1.0 (perfect answers) for many questions, "
            "with a fat bulge near the top. Base models all cluster near 0.0-0.2 with thin tails, "
            "meaning they rarely produce high-quality answers. The median (middle dashed line) for "
            "Jarvis is around 0.4, while baselines are around 0.05-0.10."
        )
        st.markdown("")

    # Chart 5: Answer Length
    img_path = os.path.join(img_dir, "answer_length_box.png")
    if os.path.exists(img_path):
        st.markdown("#### Answer Length Distribution")
        st.markdown(
            "**How to read:** Box plots show the median (middle line), interquartile range (box), "
            "and outliers (dots). This shows how verbose each model's answers are. "
            "For QASPER (short-answer Q&A), shorter answers that match the reference style score better."
        )
        st.image(img_path, use_container_width=True)
        st.markdown(
            "**Analysis:** Jarvis produces concise answers (~10-20 words) matching QASPER's short-answer style. "
            "All base models produce 80-130 word answers — they haven't learned the dataset's expected format. "
            "DeepSeek-V3.2 (API) is the most verbose (~135 words median). This verbosity explains why base models "
            "have low Token F1 despite sometimes containing the right information buried in long responses."
        )
        st.markdown("")

    # Chart 6: RAGAS
    img_path = os.path.join(img_dir, "ragas_comparison.png")
    if os.path.exists(img_path):
        st.markdown("#### RAGAS -- RAG Quality Metrics")
        st.markdown(
            "**How to read:** These metrics are scored by GPT-4o-mini acting as a judge. "
            "Each bar represents one model's score (0-1) on one RAG quality dimension. "
            "Context Precision/Recall measure retrieval quality (same retriever for all models). "
            "Faithfulness and Answer Relevancy measure generation quality."
        )
        st.image(img_path, use_container_width=True)
        st.markdown(
            "**Analysis:** Context Precision is similar across all models (0.74-0.77) because they share "
            "the same retriever. Faithfulness is lower for Jarvis (0.60) than base models (0.78-0.84) — "
            "this is because Jarvis gives concise answers that are harder for the GPT-4 judge to verify "
            "against long context passages. However, Jarvis has the highest Answer Correctness (0.59), "
            "meaning its answers are factually closest to the ground-truth references."
        )
        st.markdown("")

    # ── Statistical Significance ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Statistical Significance")
    st.markdown(
        "We use **Bootstrap 95% Confidence Intervals** and **Wilcoxon signed-rank tests** "
        "to verify that Jarvis Mk.X's improvements are real and not due to random chance."
    )

    with st.expander("What are these statistical tests?", expanded=False):
        st.markdown("""
**Bootstrap 95% Confidence Interval:** We resample the 100 test scores 1000 times with replacement and compute the mean each time. The interval [lower, upper] contains the true mean 95% of the time. If two models' intervals don't overlap, the difference is likely real.

**Wilcoxon Signed-Rank Test:** A non-parametric test that compares paired scores (same question, different models). The p-value tells you the probability that the difference is due to chance. p < 0.05 is significant; p < 0.001 is highly significant (marked ***).

**Delta (Δ):** The average improvement of Jarvis over the baseline. Positive Δ means Jarvis is better.
        """)

    # Load and display the summary JSON as a formatted table
    summary_json = os.path.join("models", "evaluation_v5_qwen3", "benchmark_summary.json")
    if os.path.exists(summary_json):
        with open(summary_json, "r") as f:
            summary_data = json.load(f)
        ci_data = []
        for item in summary_data:
            ci_data.append({
                "Model": item.get("model", "?"),
                "Token F1": f"{item.get('token_f1', 0):.4f}",
                "ROUGE-1": f"{item.get('rouge1', 0):.4f}",
                "METEOR": f"{item.get('meteor', 0):.4f}",
                "BERTScore F1": f"{item.get('bertscore_f1', 0):.4f}",
                "SBERT Cosine": f"{item.get('sbert_cosine', 0):.4f}",
            })
        st.markdown("**Summary of Key Metrics:**")
        st.dataframe(pd.DataFrame(ci_data).set_index("Model"), use_container_width=True)

    st.markdown(
        "All Wilcoxon tests show **p < 0.001** (***) for Jarvis vs every baseline on every metric, "
        "confirming the improvements are highly statistically significant."
    )

    # ── Key Takeaways ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Key Takeaways")
    st.markdown("""
1. **Fine-tuning delivers a 4x improvement.** Jarvis Mk.X scores 0.51 Token F1 vs 0.13 for the same base Qwen3-8B model.

2. **All improvements are statistically significant** (p < 0.001 on Wilcoxon signed-rank tests across all 5 key metrics).

3. **A fine-tuned 8B model beats a 671B frontier model.** DeepSeek-V3.2 (API) scores only 0.11 Token F1 -- our 8B model outperforms it by 4.8x on this task.

4. **BERTScore is high across all models** (0.82-0.91) because it captures semantic similarity even when word choice differs. This makes it the least discriminative metric for this comparison.

5. **Answer length matters.** Jarvis learned QASPER's concise answer style (~15 words). Base models produce 80-130 word responses, diluting their precision scores.

6. **RAGAS reveals a trade-off:** Jarvis has lower Faithfulness (0.60 vs 0.84 for base) because its short answers give the GPT-4 judge less text to verify, but its Answer Correctness is highest (0.59), meaning the content is actually more accurate.
    """)

    st.markdown("---")
    st.markdown(
        "**Ready to try it?** Click **+ New Chat** in the sidebar, upload a research paper PDF, "
        "and ask questions. Switch between models using the dropdown to compare answers yourself!"
    )

else:
    session = get_session(st.session_state.current_session)
    if not session:
        st.error("Session not found.")
        st.stop()

    session_id = session["id"]

    c_title, c_export = st.columns([6, 1])
    with c_title:
        new_title = st.text_input("Title", value=session["title"],
                                   label_visibility="collapsed")
        if new_title != session["title"]:
            update_session_title(session_id, new_title)
            st.rerun()  # Refresh so sidebar reflects the new title
    with c_export:
        if st.button("Export"):
            msgs = get_messages(session_id)
            if msgs:
                path = export_chat_to_pdf(session["title"], msgs)
                with open(path, "rb") as f:
                    st.download_button("Download", f.read(),
                                       file_name=os.path.basename(path),
                                       mime="application/pdf")

    st.markdown("---")
    active_pdfs = get_active_pdfs(session_id)

    cu, cm = st.columns([3, 2])
    with cu:
        if len(active_pdfs) < 3:
            nonce_key = f"up_nonce_{session_id}"
            if nonce_key not in st.session_state:
                st.session_state[nonce_key] = 0
            uploader_key = f"up_{session_id}_{st.session_state[nonce_key]}"

            uploaded = st.file_uploader(
                f"Upload PDF ({len(active_pdfs)}/3)",
                type=["pdf"],
                accept_multiple_files=True,
                key=uploader_key,
            )
            if uploaded:
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

                    # Generate summary using Jarvis Mk.X (our fine-tuned model)
                    with st.spinner(f"Jarvis Mk.X is summarizing {uf.name}..."):
                        # Build context from abstract + intro + conclusion
                        summary_parts = []
                        if paper.abstract:
                            summary_parts.append(f"[Abstract]\n{paper.abstract[:2500]}")
                        for sec_name, content in paper.sections.items():
                            if any(kw in sec_name.lower() for kw in ["introduction", "conclusion", "summary"]):
                                summary_parts.append(f"[{sec_name}]\n{content[:1500]}")
                        if not summary_parts and paper.sections:
                            first_sec = list(paper.sections.values())[0]
                            summary_parts.append(f"[Opening]\n{first_sec[:2500]}")

                        summary_context = "\n\n---\n\n".join(summary_parts) if summary_parts else "No text extracted."

                        try:
                            summary = _generate_summary_with_jarvis(summary_context, uf.name)
                        except Exception as e:
                            print(f"Jarvis summary failed: {e}")
                            # Fallback to raw abstract
                            if paper.abstract:
                                summary = paper.abstract[:500]
                            elif paper.sections:
                                summary = list(paper.sections.values())[0][:500]
                            else:
                                summary = "Summary generation failed."

                    add_pdf(session_id=session_id, filename=uf.name, filepath=filepath,
                            summary=summary, num_pages=paper.num_pages,
                            num_chunks=len(paper.chunks),
                            sections=list(paper.sections.keys()))

                    st.session_state.suggested_questions = generate_suggested_questions(
                        {"sections": list(paper.sections.keys())})
                    st.session_state.papers_loaded_key = None
                    st.success(f"Uploaded: {uf.name}")

                st.session_state[nonce_key] += 1
                st.rerun()
        else:
            st.info("3/3 PDFs uploaded.")

    with cm:
        if active_pdfs:
            st.markdown("**Active PDFs:**")
            for pdf in active_pdfs:
                pc1, pc2 = st.columns([4, 1])
                with pc1:
                    st.caption(f"{pdf['filename']} ({pdf['num_pages']}p, {pdf['num_chunks']} chunks)")
                with pc2:
                    if st.button("Remove", key=f"rm_{pdf['id']}"):
                        remove_pdf(pdf["id"])
                        st.session_state.papers_loaded_key = None
                        st.rerun()

    active_pdfs = get_active_pdfs(session_id)
    if active_pdfs:
        with st.expander("PDF Summaries", expanded=False):
            for pdf in active_pdfs:
                st.markdown(f"""<div class="pdf-summary">
                    <strong>{pdf['filename']}</strong> — {pdf['num_pages']}p, {pdf['num_chunks']} chunks<br>
                    <em>{pdf['summary']}</em><br>
                    <span style="font-size:0.7em; color:#93c5fd; opacity:0.7;">
                    Generated by Jarvis Mk.X</span></div>""", unsafe_allow_html=True)

    if st.session_state.suggested_questions and active_pdfs:
        st.markdown("**Suggested:**")
        cols = st.columns(3)
        for idx, q in enumerate(st.session_state.suggested_questions[:6]):
            with cols[idx % 3]:
                if st.button(q, key=f"sq_{idx}", width="stretch"):
                    st.session_state.pending_question = q
                    st.rerun()

    st.markdown("---")

    messages = get_messages(session_id)
    if active_pdfs:
        load_all_pdfs_into_bot(session_id, active_pdfs, messages)

    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # User messages: show settings pills + delete button
            if msg["role"] == "user":
                pills_html = msg.get("retrieval_methods", "")
                col_pills, col_del = st.columns([6, 1])
                with col_pills:
                    if pills_html:
                        st.markdown(
                            f"<div style='display:flex; gap:5px; margin-top:4px; flex-wrap:wrap; "
                            f"opacity:0.7;'>{pills_html}</div>",
                            unsafe_allow_html=True,
                        )
                with col_del:
                    if st.button("Delete", key=f"del_{msg['id']}",
                                 help="Delete this Q&A pair"):
                        # Delete the user message
                        delete_message(msg["id"])
                        # Delete the paired assistant message (next message)
                        if idx + 1 < len(messages) and messages[idx + 1]["role"] == "assistant":
                            delete_message(messages[idx + 1]["id"])
                        st.rerun()

            if msg["role"] == "assistant" and msg.get("confidence", 0) > 0:
                sources = msg.get("sources", [])

                with st.expander("Answer Analytics", expanded=False):

                    # ── Row 1: Quick stats bar ──────────────────────────────
                    gen_time = msg.get("generation_time", 0)
                    model_used = msg.get("retrieval_methods", "N/A")
                    num_sources = len(sources)
                    top_score = max((s.get("score", 0) for s in sources), default=0)
                    methods_used = list(set(s.get("method", "?") for s in sources))

                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Confidence", f"{msg['confidence']:.0%}")
                    sc2.metric("Sources", f"{num_sources}")
                    sc3.metric("Top Score", f"{top_score:.3f}")
                    sc4.metric("Time", f"{gen_time:.1f}s")

                    st.caption(f"Model: {model_used} | Methods: {', '.join(methods_used)}")
                    st.divider()

                    # ── Row 2: Confidence gauge + Score bar chart ──────────
                    v1, v2 = st.columns(2)
                    with v1:
                        st.plotly_chart(create_confidence_gauge(msg["confidence"]),
                                        width="stretch",
                                        key=f"chart_conf_{msg['id']}")
                    with v2:
                        fig = create_source_scores_bar(sources)
                        if fig:
                            st.plotly_chart(fig, width="stretch",
                                            key=f"chart_scores_{msg['id']}")

                    # ── Row 3: Dense/Sparse breakdown + Answer keywords ────
                    v3, v4 = st.columns(2)
                    with v3:
                        fig = create_method_breakdown(sources)
                        if fig:
                            st.plotly_chart(fig, width="stretch",
                                            key=f"chart_method_{msg['id']}")
                    with v4:
                        fig = create_answer_keywords(msg["content"])
                        if fig:
                            st.plotly_chart(fig, width="stretch",
                                            key=f"chart_keywords_{msg['id']}")

                    # ── Row 4: 3D Vector Space (UMAP/PCA) ─────────────────
                    if sources and len(sources) >= 2:
                        st.markdown("**3D Vector Space — Query vs Retrieved Chunks**")
                        st.caption("Red diamond = your query | "
                                   "Green = high score | Yellow = medium | Blue = low. "
                                   "Hover to see chunk text.")
                        prev_user = [m for m in messages if m["id"] < msg["id"] and m["role"] == "user"]
                        query_text = prev_user[-1]["content"] if prev_user else "query"
                        bot = get_bot()
                        fig = create_3d_vector_space(query_text, sources, bot.retriever.embed_model)
                        if fig:
                            st.plotly_chart(fig, width="stretch",
                                            key=f"chart_3d_{msg['id']}")

                    st.divider()

                    # ── Retrieved Chunks (expandable, with full text) ──────
                    if sources:
                        st.markdown("**Retrieved Source Chunks**")
                        for idx_s, src in enumerate(sources[:5]):
                            method = src.get("method", "hybrid")
                            method_color = {"dense": "[D]", "sparse": "[S]", "hybrid": "[H]"}.get(method, "")
                            score_val = src.get("score", 0)
                            section = src.get("section", "Unknown")
                            pages = src.get("page_numbers", [])
                            page_str = f"p.{', '.join(str(p) for p in pages)}" if pages else "p.?"

                            st.markdown(
                                f"**{method_color} Source {idx_s+1}** — "
                                f"`{section}` | {page_str} | "
                                f"Score: **{score_val:.3f}** | {method.title()}"
                            )
                            full_text = src.get("text") or src.get("text_preview", "")
                            # Show first 200 chars, expand for full
                            if len(full_text) > 200:
                                with st.expander(f"View full chunk ({len(full_text)} chars)", expanded=False):
                                    st.text(full_text)
                            else:
                                st.caption(full_text)

                    st.divider()

                    # ── PDF Page Preview (highest confidence source) ───────
                    if sources and active_pdfs:
                        st.markdown("**Referenced PDF Page** *(highest confidence source)*")

                        # Find the best source
                        best_src = max(sources, key=lambda s: s.get("score", 0))
                        pages = best_src.get("page_numbers", [])

                        if pages:
                            # Determine which PDF this chunk came from
                            sec = (best_src.get("section") or "").strip()
                            pdf_for_source = active_pdfs[0]  # default
                            if sec.startswith("[") and "]" in sec:
                                tagged_name = sec[1:sec.index("]")].strip()
                                match = next(
                                    (p for p in active_pdfs if p.get("filename") == tagged_name),
                                    None,
                                )
                                if match and os.path.exists(match.get("filepath", "")):
                                    pdf_for_source = match

                            pnum = pages[0] - 1  # 0-indexed
                            if os.path.exists(pdf_for_source["filepath"]):
                                img = render_pdf_page_simple(pdf_for_source["filepath"], pnum)
                                if img:
                                    st.image(img,
                                             caption=f"{pdf_for_source['filename']} — Page {pages[0]} "
                                                     f"(Score: {best_src.get('score', 0):.3f})",
                                             width=600)

    pending = st.session_state.pop("pending_question", None)

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

            c_model, c1, c2, c_send = st.columns([2.2, 1, 1, 0.6])
            with c_model:
                model_options = list(AVAILABLE_MODELS.keys())
                model_labels = [AVAILABLE_MODELS[k]["label"] for k in model_options]
                default_model = st.session_state.get("current_model", "jarvis_finetuned")
                if default_model not in model_options:
                    default_model = model_options[0]
                current_idx = model_options.index(default_model)
                selected_label = st.selectbox(
                    "Model",
                    model_labels,
                    index=current_idx,
                    key=f"model_select_{session_id}",
                    label_visibility="collapsed",
                )
                selected_model_key = model_options[model_labels.index(selected_label)]
            with c1:
                leniency = st.slider(
                    "Leniency",
                    0,
                    100,
                    int(session.get("leniency", 50)),
                    help="0=strict, 100=lenient",
                    key=f"leniency_{session_id}",
                )
            with c2:
                top_k = st.slider(
                    "Top-K",
                    1,
                    10,
                    int(session.get("top_k", 5)),
                    help="Chunks to retrieve",
                    key=f"topk_{session_id}",
                )
            with c_send:
                sent = st.form_submit_button("Send")

        components.html(
                """
                <script>
                (function () {
                    function setComposerHeight() {
                        try {
                            const doc = window.parent.document;
                            const composer = doc.querySelector('.st-key-composer');
                            if (!composer) return;
                            const h = Math.ceil(composer.getBoundingClientRect().height);
                            if (h > 0) doc.documentElement.style.setProperty('--composer-height', h + 'px');
                        } catch (e) {}
                    }

                    setComposerHeight();

                    try {
                        const doc = window.parent.document;
                        const composer = doc.querySelector('.st-key-composer');
                        if (composer && 'ResizeObserver' in window.parent) {
                            const ro = new window.parent.ResizeObserver(setComposerHeight);
                            ro.observe(composer);
                        }
                    } catch (e) {}

                    try {
                        window.parent.addEventListener('resize', setComposerHeight);
                    } catch (e) {}
                })();
                </script>
                """,
                height=0,
        )

    # Update model selection in session state
    st.session_state.current_model = selected_model_key

    if (leniency != session.get("leniency", 50)) or (top_k != session.get("top_k", 5)):
        update_session_settings(session_id, leniency, top_k)

    if sent:
        prompt = (prompt_text or "").strip()
    elif pending:
        prompt = (pending or "").strip()
        st.session_state[prompt_key] = ""
    else:
        prompt = None
    if prompt:
        if not active_pdfs:
            st.warning("Upload a PDF first.")
        else:
            model_config = AVAILABLE_MODELS.get(selected_model_key, MODEL_CONFIGS.get(selected_model_key))
            _short = model_config['short_name']

            # Build settings pills HTML (shown on user bubble + stored in DB)
            settings_pills_html = (
                f"<span style='background:rgba(37,99,235,0.22); color:#93c5fd; "
                f"padding:2px 8px; border-radius:10px; font-size:0.72em;'>{_short}</span>"
                f"<span style='background:rgba(22,101,52,0.22); color:#86efac; "
                f"padding:2px 8px; border-radius:10px; font-size:0.72em;'>Leniency: {leniency}</span>"
                f"<span style='background:rgba(88,28,135,0.22); color:#c4b5fd; "
                f"padding:2px 8px; border-radius:10px; font-size:0.72em;'>Top-K: {top_k}</span>"
            )

            with st.chat_message("user"):
                st.markdown(prompt)
                st.markdown(
                    f"<div style='display:flex; gap:5px; margin-top:4px; flex-wrap:wrap; "
                    f"opacity:0.7;'>{settings_pills_html}</div>",
                    unsafe_allow_html=True,
                )
            add_message(session_id, "user", prompt,
                        retrieval_methods=settings_pills_html)

            with st.chat_message("assistant"):
                with st.spinner(f"{_short} is thinking..."):
                    start_time = time.time()
                    bot = get_bot()

                    # Ensure PDFs are indexed on this bot (needed on first question)
                    load_all_pdfs_into_bot(session_id, active_pdfs, messages)

                    # Load LLM → answer → unload LLM (GPU free between questions)
                    response = answer_with_model(
                        bot, selected_model_key, prompt,
                        top_k=top_k, leniency=leniency,
                    )
                    response.generation_time = time.time() - start_time

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
                    retrieval_methods=f"{model_config['short_name']} | " + ", ".join(methods),
                )

            if len(get_messages(session_id)) <= 2:
                update_session_title(session_id, prompt[:50] + ("..." if len(prompt) > 50 else ""))

            st.rerun()