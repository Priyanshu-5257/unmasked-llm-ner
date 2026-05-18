import torch
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "LiquidAI/LFM2.5-350M"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]

rank_pattern = {
    "layers.2.self_attn.q_proj": 4,
    "layers.2.self_attn.k_proj": 4,
    "layers.2.self_attn.v_proj": 4,
    "layers.2.self_attn.out_proj": 4,
    "layers.2.conv.in_proj": 4,
    "layers.2.conv.out_proj": 4,
}

alpha_pattern = {
    "layers.2.self_attn.q_proj": 8,
    "layers.2.self_attn.k_proj": 8,
    "layers.2.self_attn.v_proj": 8,
    "layers.2.self_attn.out_proj": 8,
    "layers.2.conv.in_proj": 8,
    "layers.2.conv.out_proj": 8,
}

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

for name, module in peft_model.named_modules():
    if hasattr(module, "r") and isinstance(module.r, dict):
        if "layers.2" in name:
            print(f"  {name}: r={module.r.get('default', '?')}")
