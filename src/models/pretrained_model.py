from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
import torch
from config import DEVICE


class GPT_Pretrained_Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        if model_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2Model.from_pretrained('gpt2')

        model = self.model.to(DEVICE)
        model.eval()

    def predict_topk(self, text, top_k=5):
        # Convert text -> token IDs
        inputs = self.tokenizer(text, return_tensors="pt")

        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Logits của token cuối cùng
        next_token_logits = outputs.logits[:, -1, :]

        # Convert logits -> probability
        probabilities = torch.softmax(next_token_logits, dim=-1)

        # Top K
        top_probs, top_indices = torch.topk(
            probabilities,
            k=top_k,
            dim=-1
        )

        print(zip(
            top_probs[0],
            top_indices[0]
        ))

        results = []

        for prob, token_id in zip(
                top_probs[0],
                top_indices[0]
        ):
            token = self.tokenizer.decode([token_id])

            results.append({
                "token": token,
                "probability": prob.item()
            })

        return results