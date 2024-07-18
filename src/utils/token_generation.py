from src.utils.logit_difference import create_logit_difference_generator, LogitDifferenceError
import logging

def generate_from_logit_diff(model1, tokenizer1, model2, tokenizer2, prompt, max_new_tokens=50):
    try:
        generator = create_logit_difference_generator(model1, tokenizer1, model2, tokenizer2)
        return generator.safe_generate(prompt, max_new_tokens)
    except LogitDifferenceError as e:
        logging.error(f"Failed to generate text: {str(e)}")
        return None