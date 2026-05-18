from transformers import AutoModel, AutoTokenizer
import torch

model_name = "LiquidAI/LFM2.5-350M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

print("Layer names and modules:")
for name, module in model.named_modules():
    if "layers." in name and ("q_proj" in name or "v_proj" in name or "in_proj" in name or "out_proj" in name or "w1" in name or "w2" in name or "w3" in name):
        print(f"  {name}")

print(f"\nTotal layers: {model.config.num_hidden_layers}")
print(f"Layer types: {model.config.layer_types}")
