from src.models.model_loader import load_models, get_default_device
from src.utils.token_generation import generate_from_logit_diff

def main():
    model1_name = "gpt2"
    model2_name = "distilgpt2"
    device = get_default_device()
    model1, tokenizer1, model2, tokenizer2 = load_models(model1_name, model2_name, device)

    prompt = "Once upon a time"
    generated_text = generate_from_logit_diff(model1, tokenizer1, model2, tokenizer2, prompt, max_new_tokens=50)
    print(generated_text)

if __name__ == "__main__":
    main()