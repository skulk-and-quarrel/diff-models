from transformers import AutoModelForCausalLM, AutoTokenizer

def load_models(model1_name, model2_name):
    model1 = AutoModelForCausalLM.from_pretrained(model1_name)
    tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
    model2 = AutoModelForCausalLM.from_pretrained(model2_name)
    tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
    return model1, tokenizer1, model2, tokenizer2