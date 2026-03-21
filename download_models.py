"""
Pre-Download Local Models for Jarvis Mk.X
==========================================
Run this ONCE to download the 2 local model weights to your HuggingFace cache.
After this, model loading takes ~30-60s (from SSD) instead of 5-10 min (download).

Only 2 models need local download — the other 4 use API:
  - Qwen3-8B (for our fine-tuned Jarvis adapter)
  - DeepSeek-R1-Distill-Qwen-7B (no API available)

Usage:
    python download_models.py
"""

from huggingface_hub import snapshot_download
import os

MODELS = [
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

print("=" * 60)
print("  Jarvis Mk.X — Pre-Download Local Models")
print("=" * 60)
print(f"\n  Downloading {len(MODELS)} models (~20 GB total)")
print(f"  Cache dir: {os.path.expanduser('~/.cache/huggingface')}")
print(f"\n  Note: Qwen3 Base, Llama 3.1, Mistral 7B, and")
print(f"  DeepSeek-V3.2 all use API — no local download needed.\n")

for i, model_name in enumerate(MODELS, 1):
    print(f"\n[{i}/{len(MODELS)}] Downloading: {model_name}")
    try:
        path = snapshot_download(
            model_name,
            ignore_patterns=["*.gguf", "*.ggml"],
        )
        print(f"  ✓ Cached at: {path}")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")

print("\n" + "=" * 60)
print("  Done! Local models cached.")
print("  Also set these environment variables before running the app:")
print("    OPENROUTER_API_KEY=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  (for Qwen3, Llama, Mistral API)")
print("    DEEPSEEK_API_KEY=sk-e5c4499fb6a04840944e7c8727a864a2    (for DeepSeek-V3.2 API)")
print("    VOYAGE_API_KEY=al-cBLokza63Waw0-eZsRvQG4fFKx-1e8xbaRWPI8DL3tq      (for Voyage 4 embeddings)")
print("=" * 60)
