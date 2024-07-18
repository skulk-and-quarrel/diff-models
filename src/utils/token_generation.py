import torch
from transformers import LogitsProcessorList, AutoModelForCausalLM
from src.processors.logit_difference_processor import LogitDifferenceProcessor

def generate_from_logit_diff(model1, tokenizer1, model2, tokenizer2, prompt, max_new_tokens=50):
    try:
        # if not (isinstance(model1, AutoModelForCausalLM) and isinstance(model2, AutoModelForCausalLM)):
        #     raise ValueError("Both models must be instances of AutoModelForCausalLM")

        logit_diff_processor = LogitDifferenceProcessor(model2, tokenizer1, tokenizer2)
        
        # Encode the input without padding
        encoded_input = tokenizer1.encode_plus(
            prompt, 
            return_tensors="pt", 
            truncation=True
        )
        
        input_ids = encoded_input['input_ids'].to(model1.device)
        attention_mask = encoded_input['attention_mask'].to(model1.device)
        
        output = model1.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer1.pad_token_id,
            logits_processor=LogitsProcessorList([logit_diff_processor]),
        )
        
        return tokenizer1.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"An error occurred during generation: {str(e)}")
        return None