import torch
from transformers import LogitsProcessor

class LogitDifferenceProcessor(LogitsProcessor):
    def __init__(self, model2, tokenizer2):
        self.model2 = model2
        self.tokenizer2 = tokenizer2

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():
            input_ids2 = self.tokenizer2.encode(self.tokenizer2.decode(input_ids[0]), return_tensors="pt")
            outputs2 = self.model2(input_ids2)
            logits2 = outputs2.logits[:, -1, :]

        aligned_scores, aligned_logits2 = self.align_logits(scores, logits2)
        logit_diff = aligned_scores - aligned_logits2

        return logit_diff

    def align_logits(self, scores, logits2):
        vocab1 = set(range(scores.shape[-1]))
        vocab2 = set(range(logits2.shape[-1]))
        all_tokens = vocab1 | vocab2

        aligned_scores = torch.zeros(len(all_tokens), device=scores.device)
        aligned_logits2 = torch.zeros(len(all_tokens), device=logits2.device)

        for i in all_tokens:
            if i in vocab1:
                aligned_scores[i] = scores[0, i]
            if i in vocab2:
                aligned_logits2[i] = logits2[0, i]

        return aligned_scores.unsqueeze(0), aligned_logits2.unsqueeze(0)