import torch
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "LiquidAI/LFM2.5-350M"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]

rank_pattern = {}
alpha_pattern = {}

for i in range(0, 4):
    layer_type = "conv" if i in [0, 1, 3] else "full_attention"
    if layer_type == "conv":
        for m in ["in_proj", "out_proj"]:
            rank_pattern[f"layers.{i}.conv.{m}"] = 4
            alpha_pattern[f"layers.{i}.conv.{m}"] = 8
    else:
        for m in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            rank_pattern[f"layers.{i}.self_attn.{m}"] = 4
            alpha_pattern[f"layers.{i}.self_attn.{m}"] = 8
    for m in ["w1", "w2", "w3"]:
        rank_pattern[f"layers.{i}.feed_forward.{m}"] = 4
        alpha_pattern[f"layers.{i}.feed_forward.{m}"] = 8

for i in range(4, 8):
    layer_type = "conv" if i in [4, 6, 7] else "full_attention"
    if layer_type == "conv":
        for m in ["in_proj", "out_proj"]:
            rank_pattern[f"layers.{i}.conv.{m}"] = 16
            alpha_pattern[f"layers.{i}.conv.{m}"] = 32
    else:
        for m in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            rank_pattern[f"layers.{i}.self_attn.{m}"] = 16
            alpha_pattern[f"layers.{i}.self_attn.{m}"] = 32
    for m in ["w1", "w2", "w3"]:
        rank_pattern[f"layers.{i}.feed_forward.{m}"] = 16
        alpha_pattern[f"layers.{i}.feed_forward.{m}"] = 32

for i in range(8, 12):
    layer_type = "conv" if i in [9, 11] else "full_attention"
    if layer_type == "conv":
        for m in ["in_proj", "out_proj"]:
            rank_pattern[f"layers.{i}.conv.{m}"] = 32
            alpha_pattern[f"layers.{i}.conv.{m}"] = 64
    else:
        for m in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            rank_pattern[f"layers.{i}.self_attn.{m}"] = 32
            alpha_pattern[f"layers.{i}.self_attn.{m}"] = 64
    for m in ["w1", "w2", "w3"]:
        rank_pattern[f"layers.{i}.feed_forward.{m}"] = 32
        alpha_pattern[f"layers.{i}.feed_forward.{m}"] = 64

for i in range(12, 16):
    layer_type = "conv" if i in [13, 15] else "full_attention"
    if layer_type == "conv":
        for m in ["in_proj", "out_proj"]:
            rank_pattern[f"layers.{i}.conv.{m}"] = 64
            alpha_pattern[f"layers.{i}.conv.{m}"] = 128
    else:
        for m in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            rank_pattern[f"layers.{i}.self_attn.{m}"] = 64
            alpha_pattern[f"layers.{i}.self_attn.{m}"] = 128
    for m in ["w1", "w2", "w3"]:
        rank_pattern[f"layers.{i}.feed_forward.{m}"] = 64
        alpha_pattern[f"layers.{i}.feed_forward.{m}"] = 128

print("Sample patterns:")
for k in sorted(rank_pattern.keys()):
    if "out_proj" in k:
        print(f"  {k}: r={rank_pattern[k]}")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    rank_pattern=rank_pattern,
    alpha_pattern=alpha_pattern,
)

peft_model = get_peft_model(base_model, lora_config)

print("\nActual module ranks (out_proj only):")
for name, module in peft_model.named_modules():
    if hasattr(module, "r") and isinstance(module.r, dict) and "out_proj" in name:
        print(f"  {name}: r={module.r.get('default', '?')}")
