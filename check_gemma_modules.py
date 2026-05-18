from transformers import AutoModel

model = AutoModel.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
layer = model.layers[0]
print("self_attn modules:")
for name, module in layer.self_attn.named_children():
    print(f"  {name}: {type(module).__name__}")
print("\nmlp modules:")
for name, module in layer.mlp.named_children():
    print(f"  {type(module).__name__}")
