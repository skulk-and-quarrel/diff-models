import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_models(model1_name, model2_name, device=None):
    if device is None:
        device = get_default_device()
    
    print(f"Using device: {device}")

    model1 = AutoModelForCausalLM.from_pretrained(model1_name).to(device)
    tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
    model2 = AutoModelForCausalLM.from_pretrained(model2_name).to(device)
    tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
    return model1, tokenizer1, model2, tokenizer2