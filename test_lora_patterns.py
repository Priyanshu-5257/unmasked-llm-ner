import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

# Use a tiny model for quick testing
model_name = "HuggingFaceTB/SmolLM2-135M"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    rank_pattern={
        "model.layers.0.self_attn.q_proj": 4,
        "model.layers.0.self_attn.v_proj": 4,
        "model.layers.10.self_attn.q_proj": 16,
        "model.layers.10.self_attn.v_proj": 16,
    },
    alpha_pattern={
        "model.layers.0.self_attn.q_proj": 8,
        "model.layers.0.self_attn.v_proj": 8,
        "model.layers.10.self_attn.q_proj": 32,
        "model.layers.10.self_attn.v_proj": 32,
    }
)

print("Applying LoRA config with rank/alpha patterns...")
model = get_peft_model(model, config)

print("\nChecking LoRA ranks per layer:")
for name, module in model.named_modules():
    if hasattr(module, "r"):
        print(f"  {name}: r={module.r}, alpha={module.lora_alpha}")

print("\nTest complete.")
