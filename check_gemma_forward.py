from transformers import AutoModel
import inspect

model = AutoModel.from_pretrained("google/gemma-3-270m", trust_remote_code=True)

# Get the source of the forward method
source = inspect.getsource(model.forward)
# Print first 100 lines
lines = source.split('\n')[:100]
for i, line in enumerate(lines):
    print(f"{i:3d}: {line}")
