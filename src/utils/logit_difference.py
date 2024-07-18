from abc import ABC, abstractmethod
import logging
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, LogitsProcessorList
from llama_cpp import Llama
import numpy as np


class LogitDifferenceError(Exception):
    """Custom exception for LogitDifferenceGenerator errors"""
    pass

class LogitDifferenceGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        pass

    def safe_generate(self, prompt: str, max_new_tokens: int) -> str:
        try:
            return self.generate(prompt, max_new_tokens)
        except Exception as e:
            logging.error(f"Error in LogitDifferenceGenerator: {str(e)}")
            raise LogitDifferenceError(f"Failed to generate text: {str(e)}")

class TransformersLogitDifferenceGenerator(LogitDifferenceGenerator):
    def __init__(self, model1: AutoModelForCausalLM, tokenizer1: PreTrainedTokenizer, 
                 model2: AutoModelForCausalLM, tokenizer2: PreTrainedTokenizer):
        self.model1 = model1
        self.tokenizer1 = tokenizer1
        self.model2 = model2
        self.tokenizer2 = tokenizer2

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        from src.processors.logit_difference_processor import LogitDifferenceProcessor
        
        logit_diff_processor = LogitDifferenceProcessor(self.model2, self.tokenizer1, self.tokenizer2)
        
        encoded_input = self.tokenizer1.encode_plus(
            prompt, 
            return_tensors="pt", 
            truncation=True
        )
        
        input_ids = encoded_input['input_ids'].to(self.model1.device)
        attention_mask = encoded_input['attention_mask'].to(self.model1.device)
        
        output = self.model1.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer1.pad_token_id,
            logits_processor=LogitsProcessorList([logit_diff_processor]),
        )
        
        generated_text = self.tokenizer1.decode(output[0], skip_special_tokens=True)
        logging.info("Generation completed successfully")
        return generated_text
    
class LlamaLogitDifferenceGenerator(LogitDifferenceGenerator):
    def __init__(self, model1: Llama, model2: Llama):
        self.model1 = model1
        self.model2 = model2
        self.vocab_mapping = self._create_vocab_mapping()

    def _create_vocab_mapping(self):
        vocab1 = self.model1.tokenizer_.vocab
        vocab2 = self.model2.tokenizer_.vocab
        
        mapping = {}
        for token, id2 in vocab2.items():
            if token in vocab1:
                id1 = vocab1[token]
                mapping[id2] = id1
        
        return mapping

    def get_logits(self, model: Llama, prompt: str):
        tokens = model.tokenize(prompt.encode("utf-8"))
        if len(tokens) > model.n_ctx():
            tokens = tokens[-model.n_ctx():]
        model.eval(tokens)
        return model.scores[-1, :]

    def calculate_logit_difference(self, prompt: str):
        logits1 = self.get_logits(self.model1, prompt)
        logits2 = self.get_logits(self.model2, prompt)
        
        aligned_logits2 = np.full_like(logits1, -np.inf)
        
        for id2, id1 in self.vocab_mapping.items():
            aligned_logits2[id1] = logits2[id2]
        
        return logits1 - aligned_logits2

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        generated_text = prompt
        for _ in range(max_new_tokens):
            logit_diff = self.calculate_logit_difference(generated_text)
            
            next_token_id = np.argmax(logit_diff)
            
            next_token = self.model1.detokenize([next_token_id]).decode("utf-8")
            
            generated_text += next_token
            
            if next_token_id == self.model1.token_eos():
                break

        logging.info("Generation completed successfully")
        return generated_text

def create_logit_difference_generator(model1, tokenizer1, model2, tokenizer2) -> LogitDifferenceGenerator:
    if isinstance(model1, AutoModelForCausalLM) and isinstance(model2, AutoModelForCausalLM):
        return TransformersLogitDifferenceGenerator(model1, tokenizer1, model2, tokenizer2)
    elif isinstance(model1, Llama) and isinstance(model2, Llama):
        return LlamaLogitDifferenceGenerator(model1, model2)
    else:
        raise ValueError("Unsupported model types")
