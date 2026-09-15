import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPTPretrainedModel:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = None
        self.model = None

    def load(self):
        print(f"Loading {self.model_name}...")
        print(f"Device: {self.device}")

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_name
        )

        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        print("GPT-2 loaded successfully.")

    def predict_top5(self, text):
        # Text -> token IDs
        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Logits của token cuối cùng
        next_token_logits = outputs.logits[:, -1, :]

        # Logits -> probability
        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        # Lấy top 5
        top_probs, top_token_ids = torch.topk(
            probabilities,
            k=5,
            dim=-1
        )

        results = []

        for prob, token_id in zip(
            top_probs[0],
            top_token_ids[0]
        ):
            token = self.tokenizer.decode([token_id])

            results.append({
                "token": token,
                "probability": prob.item()
            })

        return results