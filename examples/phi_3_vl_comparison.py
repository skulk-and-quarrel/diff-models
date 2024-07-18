from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from src.utils.token_generation import generate_from_logit_diff
import torch

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def main():
    device = get_default_device()
    print(f"Using device: {device}")

    # Load model1: Phi-3-mini-128k-instruct
    model1 = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer1 = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    # Load model2: Phi-3-vision-128k-instruct
    model2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-vision-128k-instruct",
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True,
        _attn_implementation='eager'
    )
    tokenizer2 = AutoProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True)

    # Prepare input in the style of Phi-3 models
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What do giraffes look like?"},
    ]

    prompt = tokenizer1.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    generated_text = generate_from_logit_diff(model1, tokenizer1, model2, tokenizer2.tokenizer, prompt, max_new_tokens=50)
    print(generated_text)

if __name__ == "__main__":
    main()