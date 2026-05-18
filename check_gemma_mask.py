from transformers import AutoModel
import inspect

model = AutoModel.from_pretrained("google/gemma-3-270m", trust_remote_code=True)
print("Available methods with 'mask' or 'causal' in name:")
for name in dir(model):
    if 'mask' in name.lower() or 'causal' in name.lower():
        print(f"  {name}")

print("\n_forward signature:")
sig = inspect.signature(model.forward)
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param}")
