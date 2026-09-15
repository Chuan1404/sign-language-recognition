from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from config import DEVICE


class GPTPretrainedModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load(self, model_name):
        if model_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        self.model.to(DEVICE)
        return self.model

    def predict_topk(self, text, top_k=5):
        if self.model is None:
            raise RuntimeError("Call load() before predict_topk().")

        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        next_token_logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)

        results = []
        for prob, token_id in zip(top_probs[0], top_indices[0]):
            token = self.tokenizer.decode([token_id.item()])
            results.append({"token": token, "probability": prob.item()})

        return results