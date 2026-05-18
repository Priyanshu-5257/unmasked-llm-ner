from transformers import AutoModel, AutoConfig

config = AutoConfig.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
print(f"Num layers: {config.num_hidden_layers}")
print(f"Hidden size: {config.hidden_size}")
print(f"Model type: {config.model_type}")

model = AutoModel.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
print("\nLayer types:")
for i, layer in enumerate(model.layers):
    print(f"  Layer {i}: {type(layer).__name__}")
    for name, module in layer.named_children():
        print(f"    {name}: {type(module).__name__}")
    if i >= 2:
        print("  ...")
        break
