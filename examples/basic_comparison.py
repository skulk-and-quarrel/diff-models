import logging
from src.models.model_loader import load_transformers_models, load_llama_models
from src.utils.token_generation import generate_from_logit_diff

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    setup_logging()

    # For transformers models
    model1_name = "gpt2"
    model2_name = "distilgpt2"
    model1, tokenizer1, model2, tokenizer2 = load_transformers_models(model1_name, model2_name)

    prompt = "Once upon a time"
    generated_text = generate_from_logit_diff(model1, tokenizer1, model2, tokenizer2, prompt, max_new_tokens=50)
    if generated_text:
        print("Transformers generated text:", generated_text)
    else:
        print("Failed to generate text with Transformers models")

    # For llama.cpp models
    model1_path = "path/to/model1.ggml"
    model2_path = "path/to/model2.ggml"
    llama_model1, llama_model2 = load_llama_models(model1_path, model2_path)

    llama_generated_text = generate_from_logit_diff(llama_model1, None, llama_model2, None, prompt, max_new_tokens=50)
    if llama_generated_text:
        print("Llama.cpp generated text:", llama_generated_text)
    else:
        print("Failed to generate text with Llama.cpp models")

if __name__ == "__main__":
    main()