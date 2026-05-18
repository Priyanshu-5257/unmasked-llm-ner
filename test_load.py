from transformers import AutoTokenizer, AutoModel
try:
    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M")
    model = AutoModel.from_pretrained("LiquidAI/LFM2.5-350M", trust_remote_code=True)
    print("Model loaded successfully")
    print(model.__class__)
except Exception as e:
    print(f"Error loading model: {e}")
