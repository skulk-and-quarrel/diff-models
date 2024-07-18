import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def ensure_padding_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set padding token to: {tokenizer.pad_token}")
    return tokenizer

def load_models(model1_name, model2_name, device=None):
    if device is None:
        device = get_default_device()
    
    print(f"Using device: {device}")

    model1 = AutoModelForCausalLM.from_pretrained(model1_name).to(device)
    tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
    tokenizer1 = ensure_padding_token(tokenizer1)

    model2 = AutoModelForCausalLM.from_pretrained(model2_name).to(device)
    tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
    tokenizer2 = ensure_padding_token(tokenizer2)
    
    return model1, tokenizer1, model2, tokenizer2