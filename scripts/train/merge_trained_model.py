from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "/data2/da02/models/Qwen3-4B-Instruct-2507"
adapter = "/data4/jongho/lattice/results/train/branch_policy_dpo_1500_sample_20260227_095158/final"
out = "/data4/jongho/lattice/results/train/branch_policy_dpo_1500_sample_20260227_095158/merged"

model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)

tok = AutoTokenizer.from_pretrained(adapter, use_fast=True)
tok.save_pretrained(out)
print("saved:", out)