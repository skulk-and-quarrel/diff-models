import torch
from transformers import LogitsProcessor
from typing import Dict, Any

class LogitDifferenceProcessor(LogitsProcessor):
    def __init__(self, model2, tokenizer1, tokenizer2):
        self.model2 = model2
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():
            # Decode using tokenizer1, then encode using tokenizer2
            text = self.tokenizer1.decode(input_ids[0])
            
            encoded_input2 = self.tokenizer2.encode_plus(
                text, 
                return_tensors="pt", 
                truncation=True,
                max_length=input_ids.shape[1]
            )
            input_ids2 = encoded_input2['input_ids'].to(input_ids.device)
            
            # Create attention mask based on the actual sequence length
            attention_mask2 = torch.ones_like(input_ids2)
            
            outputs2 = self.model2(input_ids2, attention_mask=attention_mask2)
            logits2 = outputs2.logits[:, -1, :]

        # Calculate logit difference while respecting vocabulary differences
        logit_diff = self.calculate_logit_difference(scores, logits2)

        return logit_diff

    def calculate_logit_difference(self, scores1: torch.FloatTensor, scores2: torch.FloatTensor) -> torch.FloatTensor:
        vocab1 = self.tokenizer1.get_vocab()
        vocab2 = self.tokenizer2.get_vocab()

        # Initialize logit difference with negative infinity
        logit_diff = torch.full_like(scores1, float('-inf'))

        for token, idx1 in vocab1.items():
            if token in vocab2:
                idx2 = vocab2[token]
                logit_diff[0, idx1] = scores2[0, idx2] - scores1[0, idx1]

        return logit_diff