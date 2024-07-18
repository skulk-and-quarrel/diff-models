from transformers import LogitsProcessorList

def generate_from_logit_diff(model1, tokenizer1, model2, tokenizer2, prompt, max_length=50):
    from src.processors.logit_difference_processor import LogitDifferenceProcessor
    
    logit_diff_processor = LogitDifferenceProcessor(model2, tokenizer2)
    
    input_ids = tokenizer1.encode(prompt, return_tensors="pt")
    
    output = model1.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        num_beams=1,
        logits_processor=LogitsProcessorList([logit_diff_processor]),
    )
    
    return tokenizer1.decode(output[0], skip_special_tokens=True)