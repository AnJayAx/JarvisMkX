# test_load.py — loads WITHOUT 4-bit quantization
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
print("Loading model in bfloat16 (no quantization)...")
model = AutoModelForCausalLM.from_pretrained(
    name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    max_memory={0: "23GiB", "cpu": "10GiB"},
)
print(f"✓ Loaded! GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
out = model.generate(tok("Hello", return_tensors="pt").input_ids.to("cuda"), max_new_tokens=5)
print(f"✓ Generation works: {tok.decode(out[0])}")