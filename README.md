# Jarvis Mk.X -- Smart Research Paper Q&A Chatbot

A Retrieval-Augmented Generation (RAG) system for research paper Q&A, powered by a fine-tuned Qwen3-8B model with hybrid retrieval and cross-encoder reranking.

**AAI3008 Large Language Model -- Singapore Institute of Technology**

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Model Download](#model-download)
- [How It Works](#how-it-works)
- [Retrieval Pipeline](#retrieval-pipeline)
- [6-Model Benchmark Results](#6-model-benchmark-results)
- [Retrieval Strategy Comparison](#retrieval-strategy-comparison)
- [RAGAS Evaluation](#ragas-evaluation)
- [Application Screenshots](#application-screenshots)
- [Notebooks Overview](#notebooks-overview)
- [API Keys Required](#api-keys-required)
- [Known Limitations](#known-limitations)
- [Team](#team)v

---

## Overview

Jarvis Mk.X is a research paper Q&A chatbot that lets users upload PDF research papers and ask questions about their content. The system uses:

1. **PDF Processing** -- Extracts text from PDFs, detects section headers via font-size heuristics, and chunks text into ~512-token segments with metadata (section, page numbers, chunk type).

2. **Hybrid Retrieval** -- Combines dense semantic search (Voyage 3 Large embeddings + ChromaDB) with sparse keyword matching (BM25) to find the most relevant chunks for each question.

3. **Cross-Encoder Reranking** -- Uses BGE Reranker v2 M3 to re-score the top-15 hybrid candidates and return the best 5, improving Precision@5 by 41% over hybrid alone.

4. **Fine-Tuned LLM** -- A Qwen3-8B model fine-tuned with QLoRA on QASPER + PubMedQA datasets, achieving 4x better Token F1 than the base model on research paper Q&A.

5. **Multi-Model Comparison** -- Users can switch between 6 models (2 local + 4 API) to compare answer quality in real-time.

---

## System Architecture

```
                            JARVIS Mk.X -- SYSTEM ARCHITECTURE
    ===========================================================================

    USER INTERFACE (Streamlit)
    +-----------------------------------------------------------------------+
    |  Sidebar          |  Chat Area                                        |
    |  - Chat sessions  |  - PDF upload (up to 3)                           |
    |  - Search         |  - Model selector (6 models)                      |
    |  - New Chat       |  - Leniency / Top-K controls                      |
    |                   |  - Answer with Answer/Reason/Sources format        |
    |                   |  - Answer Analytics (charts, 3D vector space, PDF) |
    +-----------------------------------------------------------------------+
                                        |
                                        v
    ORCHESTRATION LAYER (app.py)
    +-----------------------------------------------------------------------+
    |  Session Manager (SQLite)  |  Model Router  |  Summary Generator      |
    |  - Chat history            |  - Local models|  - Jarvis summarizes    |
    |  - PDF metadata            |  - API models  |    uploaded PDFs        |
    |  - Corrections             |  - Load/Unload |                         |
    +-----------------------------------------------------------------------+
                        |                       |
              +---------+---------+    +--------+--------+
              v                   v    v                 v
    LOCAL MODELS              API MODELS            RETRIEVAL PIPELINE
    +------------------+   +------------------+   +----------------------+
    | Jarvis Mk.X      |   | Qwen3-8B Base   |   | 1. PDF Processing    |
    | (Qwen3-8B+QLoRA) |   |   (OpenRouter)  |   |    (PyMuPDF)         |
    |                  |   | Llama-3.1-8B    |   |                      |
    | DeepSeek-R1-7B   |   |   (OpenRouter)  |   | 2. Embedding         |
    | (local, no API)  |   | Mistral-7B      |   |    (Voyage 3 Large)  |
    +------------------+   |   (OpenRouter)  |   |                      |
                           | DeepSeek-V3.2   |   | 3. Hybrid Retrieval  |
                           |   (DeepSeek API)|   |    Dense + BM25      |
                           +------------------+   |                      |
                                                  | 4. BGE Reranking     |
                                                  |    (CrossEncoder)    |
                                                  +----------------------+
                                                             |
                                                             v
                                                  +----------------------+
                                                  | ChromaDB (dense)     |
                                                  | BM25Okapi (sparse)   |
                                                  | BGE v2 M3 (rerank)   |
                                                  +----------------------+
```

### Data Flow (per question)

```
User Question
    |
    v
[Question Classifier] --> conversational? --> Direct response (no retrieval)
    |                  --> meta?           --> Abstract + Intro + Conclusion context
    |                  --> application?    --> Broad context + reasoning note
    |                  --> factual?        --> Standard retrieval
    v
[Hybrid Retrieval]
    |-- Dense: Voyage embeddings -> ChromaDB cosine search (top 15)
    |-- Sparse: BM25 keyword matching (top 15)
    |-- Fusion: 0.6 * dense + 0.4 * sparse
    v
[BGE Reranker v2 M3]
    |-- Cross-encoder re-scores 15 candidates
    |-- Returns top 5
    v
[LLM Generation]
    |-- Local: Load model -> Generate -> Unload (GPU freed)
    |-- API: Send to OpenRouter/DeepSeek API
    v
[Response]
    |-- Answer: (grounded in context)
    |-- Reason: (how context supports it)
    |-- Sources: (section/page references)
```

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-PDF Support** | Upload up to 3 PDFs per chat session |
| **6 Model Selection** | Switch between Jarvis (fine-tuned), Qwen3 Base, DeepSeek-R1-7B, Llama-3.1-8B, Mistral-7B, DeepSeek-V3.2 |
| **Hybrid Retrieval + Reranking** | Dense (Voyage) + Sparse (BM25) fusion with BGE cross-encoder reranking |
| **Answer/Reason/Sources Format** | Structured responses with grounding and citations |
| **Conversation Memory** | Remembers up to 10 Q&A pairs for follow-up questions |
| **Answer Correction** | Users can correct wrong answers; bot remembers corrections |
| **PDF Summary by Jarvis** | Uploaded PDFs are summarized by the fine-tuned model |
| **Answer Analytics** | Confidence gauge, retrieval scores, dense/sparse breakdown, 3D vector space, PDF page preview |
| **Benchmark Dashboard** | Landing page shows full evaluation results with interactive tables and charts |
| **GPU-Efficient** | Local models load on-demand and unload immediately after answering; API models use zero GPU |
| **Session Management** | SQLite database stores chat history, PDFs, corrections across sessions |
| **PDF Export** | Export chat conversations as PDF documents |

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **LLM (Fine-tuned)** | Qwen3-8B + QLoRA | Research paper Q&A generation |
| **LLM (Base models)** | Qwen3-8B, DeepSeek-R1-7B | Local baseline comparison |
| **LLM (API models)** | Llama-3.1-8B, Mistral-7B, DeepSeek-V3.2 | API baseline comparison via OpenRouter |
| **Embeddings** | Voyage 3 Large (API, 1024-dim) | Dense semantic embeddings for retrieval |
| **Vector Store** | ChromaDB (in-memory, cosine) | Dense retrieval index |
| **Sparse Index** | BM25Okapi (rank-bm25) | Keyword-based retrieval |
| **Reranker** | BGE Reranker v2 M3 (CrossEncoder) | Cross-encoder reranking of candidates |
| **PDF Processing** | PyMuPDF (fitz) | Text extraction with font metadata |
| **Fine-tuning** | QLoRA (PEFT + bitsandbytes) | Parameter-efficient fine-tuning |
| **Training Data** | QASPER + PubMedQA | Research paper Q&A datasets |
| **Web Framework** | Streamlit | Chat UI with analytics |
| **Database** | SQLite | Session/chat/PDF persistence |
| **Evaluation** | RAGAS, BERTScore, ROUGE, BLEU, METEOR | Automated quality metrics |

---

## Project Structure

```
JarvisMkX/
|-- app.py                          # Streamlit web application (main entry)
|-- .env                            # API keys (VOYAGE, OPENROUTER, DEEPSEEK)
|-- Setup_Jarvis_Web.bat            # Windows setup script
|-- Run_Jarvis_Web.bat              # Windows launch script
|-- download_models.py              # Pre-download local models to cache
|-- requirements.txt                # Python dependencies
|
|-- src/
|   |-- bot.py                      # RAG pipeline + LLM generation
|   |-- processor.py                # PDF extraction + chunking
|   |-- retriever.py                # Hybrid retrieval + BGE reranking
|   |-- database.py                 # SQLite session/message management
|   |-- pdf_export.py               # Chat export to PDF
|
|-- models/
|   |-- jarvis-mkx-qwen3-8b-adapter/  # Fine-tuned QLoRA adapter weights
|   |-- evaluation_v5_qwen3/           # Benchmark results (CSVs, JSONs)
|       |-- benchmark_results.csv
|       |-- ragas_results.csv
|       |-- raw_*.json                 # Per-model predictions
|
|-- img/                            # Evaluation charts for landing page
|   |-- benchmark_comparison.png
|   |-- radar_chart.png
|   |-- heatmap.png
|   |-- f1_violin.png
|   |-- answer_length_box.png
|   |-- ragas_comparison.png
|
|-- data/
|   |-- processed_v2/               # QASPER + PubMedQA processed datasets
|   |-- uploads/                    # User-uploaded PDFs (per session)
|   |-- jarvis.db                   # SQLite database
|
|-- notebooks/
|   |-- Notebook_2_FineTuning_Qwen3_8B_colab.ipynb   # Fine-tuning pipeline
|   |-- Notebook_3_PDF_Processing.ipynb               # PDF extraction demo
|   |-- Notebook_4_Embedding_Retrieval_v3.ipynb       # Retrieval comparison
|   |-- Notebook_5_RAG_Pipeline.ipynb                 # End-to-end RAG test
|   |-- Notebook_6_Evaluation_v5_Qwen3_8B.ipynb       # 6-model benchmark
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (for local models) or API-only mode (no GPU needed)
- Windows 10/11 or Linux

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_REPO/JarvisMkX.git
cd JarvisMkX

# 2. Run the setup script (Windows)
Setup_Jarvis_Web.bat

# 3. Or manually install (any OS)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate peft bitsandbytes sentence-transformers
pip install voyageai chromadb rank-bm25 PyMuPDF
pip install streamlit plotly matplotlib scikit-learn fpdf2 numpy pandas Pillow

# 4. Set up API keys in .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env
echo "DEEPSEEK_API_KEY=your_key_here" >> .env
echo "VOYAGE_API_KEY=your_key_here" >> .env

# 5. Download the fine-tuned model adapter (see below)

# 6. Pre-download local models (optional, speeds up first load)
python download_models.py

# 7. Launch the application
streamlit run app.py
```

---

## Model Download

The fine-tuned QLoRA adapter weights for Jarvis Mk.X are available on Google Drive:

**Download:** [jarvis-mkx-qwen3-8b-adapter.zip](https://drive.google.com/file/d/1r-twTwb0-zsdhSHFqwN7gYEAfSbKyJy8/view?usp=sharing)

After downloading, extract the adapter folder to:
```
JarvisMkX/models/jarvis-mkx-qwen3-8b-adapter/
```

The folder should contain:
```
jarvis-mkx-qwen3-8b-adapter/
|-- adapter_config.json
|-- adapter_model.safetensors
|-- tokenizer.json
|-- tokenizer_config.json
|-- special_tokens_map.json
```

**Base model:** The adapter is applied on top of `Qwen/Qwen3-8B` which is downloaded automatically from HuggingFace on first launch.

### Fine-tuning Details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen3-8B (8.2B parameters) |
| Method | QLoRA (4-bit NF4 quantization + LoRA) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training data | QASPER (2,567 train) + PubMedQA (500 train) |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine scheduler) |
| Effective batch size | 16 |
| Optimizer | paged_adamw_8bit |
| Thinking mode | Disabled (enable_thinking=False) |

---

## How It Works

### 1. PDF Processing (processor.py)

```
PDF File
  |-- PyMuPDF extracts text blocks with font size metadata
  |-- Font-size heuristics detect section headers (> body font size)
  |-- Regex patterns match common section names (Introduction, Methods, etc.)
  |-- Text split into ~512-token chunks with 50-token overlap
  |-- Each chunk tagged with: section name, page numbers, chunk type
  v
ProcessedPaper(title, authors, abstract, sections, chunks[])
```

### 2. Hybrid Retrieval + Reranking (retriever.py)

```
User Query
  |
  |-- [Dense Path]  Voyage 3 Large embeddings -> ChromaDB cosine search
  |-- [Sparse Path] BM25 keyword tokenization -> BM25Okapi scoring
  |
  v
Hybrid Fusion: score = 0.6 * dense + 0.4 * sparse
  |
  |-- Top 15 candidates
  v
BGE Reranker v2 M3 (CrossEncoder)
  |
  |-- Re-scores all 15 candidates using (query, passage) pairs
  |-- Final score = 0.6 * reranker + 0.4 * hybrid
  v
Top 5 results returned with scores and metadata
```

### 3. Answer Generation (bot.py)

```
Retrieved chunks + Question + Conversation history
  |
  v
System Prompt (strict grounding rules)
  + Context from top-K chunks
  + Previous Q&A turns (up to 10 pairs)
  + Format instruction: Answer/Reason/Sources
  |
  v
LLM (Qwen3-8B + QLoRA adapter, 4-bit quantized)
  |
  v
Structured Response:
  **Answer:** [grounded in context]
  **Reason:** [how context supports it]
  **Sources:** [section/page references]
```

### 4. GPU Memory Management (app.py)

```
Idle State: 0 GB VRAM (only retriever on CPU)
  |
  v
User asks question
  |-- Load LLM into VRAM (~5 GB, 4-bit)
  |-- Generate answer (~3-10 seconds)
  |-- Unload LLM from VRAM
  v
Idle State: 0 GB VRAM (GPU freed for next model)
```

This load-on-demand architecture means:
- Switching between 6 models is instant (no VRAM conflicts)
- API models (4 of 6) use zero GPU at all times
- Local models only occupy VRAM during the few seconds of generation

---

## Retrieval Pipeline

### Why Hybrid + Reranking?

We evaluated 6 retrieval strategies and 3 reranker configurations in Notebook 4:

| Rank | Strategy | Precision@5 | MRR | NDCG@5 |
|---|---|---|---|---|
| 1 | **Hybrid + BGE Rerank** | **0.600** | **1.000** | **1.000** |
| 2 | Hybrid + Cohere Rerank | 0.525 | 1.000 | 0.868 |
| 3 | Keyword-boosted | 0.500 | 0.813 | 0.740 |
| 4 | Dense only | 0.475 | 0.760 | 0.710 |
| 5 | Metadata-filtered | 0.475 | 0.771 | 0.695 |
| 6 | Knowledge Graph | 0.425 | 0.775 | 0.659 |
| 7 | Hybrid (no rerank) | 0.425 | 0.729 | 0.631 |
| 8 | Sparse (BM25) only | 0.425 | 0.646 | 0.589 |

**Result:** Hybrid + BGE Reranking achieves perfect MRR and NDCG, and 41% better Precision@5 than hybrid alone. This is what our app uses.

---

## 6-Model Benchmark Results

All 6 models evaluated on 100 QASPER test questions (research paper Q&A):

### Main Metrics

| Model | Token F1 | ROUGE-1 | ROUGE-L | METEOR | BERTScore F1 | SBERT Cosine |
|---|---|---|---|---|---|---|
| **Qwen3-8B Finetuned (Jarvis)** | **0.5116** | **0.5364** | **0.5240** | **0.4140** | **0.9130** | **0.6743** |
| Qwen3-8B (Base) | 0.1283 | 0.1387 | 0.1148 | 0.2179 | 0.8323 | 0.4260 |
| DeepSeek-R1-7B (Base) | 0.0956 | 0.1043 | 0.0825 | 0.1802 | 0.8209 | 0.3682 |
| Llama-3.1-8B (Base) | 0.1480 | 0.1564 | 0.1350 | 0.2422 | 0.8380 | 0.4203 |
| Mistral-7B (Base) | 0.1284 | 0.1388 | 0.1157 | 0.2286 | 0.8330 | 0.4036 |
| DeepSeek-V3.2 (API) | 0.1070 | 0.1135 | 0.0932 | 0.1843 | 0.8202 | 0.4231 |

### Key Findings

1. **Fine-tuning delivers 4x improvement.** Jarvis scores 0.51 Token F1 vs 0.13 for the same base Qwen3-8B -- a 4x improvement from QLoRA fine-tuning alone.

2. **All improvements are statistically significant** (p < 0.001 on Wilcoxon signed-rank tests across all metrics).

3. **An 8B fine-tuned model beats a 671B frontier model.** DeepSeek-V3.2 (API, ~671B MoE) scores only 0.11 Token F1 -- our 8B model outperforms it by 4.8x on this task.

4. **Answer length matters.** Jarvis learned QASPER's concise answer style (~15 words). Base models produce 80-130 word responses, diluting their precision.

5. **BERTScore is uniformly high** (0.82-0.91) because it captures semantic similarity even when wording differs, making it the least discriminative metric.

### Statistical Significance

Bootstrap 95% confidence intervals and Wilcoxon signed-rank tests confirm all improvements:

| Metric | Jarvis Mean | Delta vs Base | p-value |
|---|---|---|---|
| ROUGE-1 | 0.5364 | +0.4229 | p=0.0000 *** |
| METEOR | 0.4140 | +0.2297 | p=0.0000 *** |
| BERTScore F1 | 0.9130 | +0.0929 | p=0.0000 *** |
| SBERT Cosine | 0.6743 | +0.2512 | p=0.0000 *** |

---

## RAGAS Evaluation

RAG quality metrics scored by GPT-4o-mini as an LLM judge:

| Model | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness |
|---|---|---|---|---|---|
| **Jarvis (Finetuned)** | 0.5986 | 0.3011 | 0.7600 | 0.6352 | **0.5945** |
| Qwen3-8B (Base) | 0.8425 | 0.4983 | 0.7400 | 0.6652 | 0.3578 |
| DeepSeek-R1-7B | 0.7088 | 0.0544 | 0.7500 | 0.6352 | 0.2960 |
| Llama-3.1-8B | 0.7764 | 0.4126 | 0.7600 | 0.6402 | 0.3612 |
| Mistral-7B | 0.7823 | 0.4959 | 0.7500 | 0.6452 | 0.3388 |
| DeepSeek-V3.2 (API) | 0.8388 | 0.5075 | 0.7700 | 0.6452 | 0.3260 |

**Analysis:**
- **Context Precision** is similar across all models (0.74-0.77) because they share the same retriever
- **Faithfulness** is lower for Jarvis (0.60) because its concise answers give the GPT-4 judge less text to verify
- **Answer Correctness** is highest for Jarvis (0.59), meaning its answers are factually closest to the references

---

## Application Screenshots

The application includes the following views:

### Landing Page
- Benchmark tables with hoverable column descriptions
- Chart visualizations with "How to read" guides
- Statistical significance analysis
- Available model cards

### Chat Interface
- Multi-PDF upload with Jarvis-generated summaries
- Model selector dropdown (6 models)
- Adjustable Leniency and Top-K sliders
- Suggested questions based on paper sections

### Answer Analytics (per response)
- Confidence gauge + quick stats (confidence, sources, top score, time)
- Retrieval scores bar chart (color-coded by method)
- Dense vs Sparse score breakdown
- Key terms in answer
- 3D Vector Space (PCA) with hover-able chunk previews
- Retrieved source chunks with expandable full text
- PDF page preview from highest-confidence source

---

## Notebooks Overview

| Notebook | Purpose | Key Output |
|---|---|---|
| **NB2: Fine-Tuning** | QLoRA fine-tuning Qwen3-8B on QASPER + PubMedQA | `models/jarvis-mkx-qwen3-8b-adapter/` |
| **NB3: PDF Processing** | Test and demo the PDF extraction + chunking pipeline | Verified `processor.py` |
| **NB4: Retrieval Comparison** | Compare 6 retrieval strategies + 3 rerankers | `data/retrieval_comparison.csv` |
| **NB5: RAG Pipeline** | End-to-end test: PDF -> Retrieve -> Generate | Verified `bot.py` |
| **NB6: 6-Model Evaluation** | Benchmark all 6 models on 100 QASPER test questions | `models/evaluation_v5_qwen3/` |

---

## API Keys Required

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-...    # For Qwen3, Llama, Mistral API models
DEEPSEEK_API_KEY=sk-...             # For DeepSeek-V3.2 API model
VOYAGE_API_KEY=pa-...               # For Voyage 3 Large embeddings
```

| Service | Free tier? | Get key at |
|---|---|---|
| OpenRouter | Yes (free models available) | https://openrouter.ai/keys |
| DeepSeek | Yes (free credits) | https://platform.deepseek.com/api_keys |
| Voyage AI | Yes (free tier) | https://dash.voyageai.com/api-keys |

---

## Known Limitations

1. **Jarvis produces shorter answers** than base models, which can appear less detailed for complex questions. This is a trade-off: concise answers score better on QASPER metrics but may not satisfy users wanting comprehensive explanations.

2. **PDF extraction quality depends on PDF formatting.** Scanned PDFs (images) won't work -- only text-based PDFs are supported. Some PDFs with unusual layouts may have imperfect section detection.

3. **BGE Reranker adds ~1-2s per query** due to cross-encoder inference. This is the trade-off for 41% better Precision@5.

4. **Conversation memory is per-session.** Switching to a different chat session clears the in-memory history (though it's reconstructed from the database on session load).

5. **RAGAS Faithfulness is lower for Jarvis** (0.60 vs 0.84 for base models) because the GPT-4 judge has difficulty verifying concise answers against long context passages. This is a known limitation of LLM-as-judge evaluation for short-answer Q&A.

---

## Team

**Project:** Jarvis Mk.X -- AAI3008 Large Language Model  
**Institution:** Singapore Institute of Technology  
**Course:** AAI3008 Large Language Model

---

*Built with Qwen3-8B, QLoRA, Voyage AI, ChromaDB, BGE Reranker, and Streamlit.*
