import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.processors.logit_difference_processor import LogitDifferenceProcessor

class TestLogitDifferenceProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model1 = AutoModelForCausalLM.from_pretrained("gpt2")
        cls.tokenizer1 = AutoTokenizer.from_pretrained("gpt2")
        cls.model2 = AutoModelForCausalLM.from_pretrained("distilgpt2")
        cls.tokenizer2 = AutoTokenizer.from_pretrained("distilgpt2")
        cls.processor = LogitDifferenceProcessor(cls.model2, cls.tokenizer1, cls.tokenizer2)

    def test_logit_difference_calculation(self):
        input_text = "Hello, world!"
        input_ids = self.tokenizer1.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs1 = self.model1(input_ids)
            scores1 = outputs1.logits[:, -1, :]
            
        logit_diff = self.processor(input_ids, scores1)
        
        self.assertEqual(logit_diff.shape, scores1.shape)
        self.assertTrue(torch.isfinite(logit_diff).all())

if __name__ == '__main__':
    unittest.main()