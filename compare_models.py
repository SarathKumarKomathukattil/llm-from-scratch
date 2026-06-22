import torch
from basic_gpt2_model import (
    GPTModel, generate, text_to_token_ids, token_ids_to_text, tokenizer
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Config for 124M
CONFIG_124M = {
    "vocab_size": 50257, "context_length": 1024, "emb_dim": 768,
    "n_heads": 12, "n_layers": 12, "drop_rate": 0.0, "qkv_bias": True
}
# Config for 355M
CONFIG_355M = {
    "vocab_size": 50257, "context_length": 1024, "emb_dim": 1024,
    "n_heads": 16, "n_layers": 24, "drop_rate": 0.0, "qkv_bias": True
}

# Load both models from your saved weights
model_124 = GPTModel(CONFIG_124M)
model_124.load_state_dict(torch.load('gpt2_124M.pth', map_location=device))
model_124.to(device).eval()

model_355 = GPTModel(CONFIG_355M)
model_355.load_state_dict(torch.load('gpt2_355M.pth', map_location=device))
model_355.to(device).eval()

# Same prompts through both
prompts = [
    "Every effort moves you",
    "The future of artificial intelligence is",
    "In a small town in the mountains,",
    "The most important lesson I learned was"
]

def run(model, prompt):
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=40,
        context_size=1024,
        top_k=50,
        temperature=1.0
    )
    return token_ids_to_text(token_ids, tokenizer)

for p in prompts:
    print("=" * 70)
    print(f"PROMPT: {p}")
    print(f"\n[124M]: {run(model_124, p)}")
    print(f"\n[355M]: {run(model_355, p)}")
    print()