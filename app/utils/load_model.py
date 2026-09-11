from sympy.printing.pytorch import torch

from src.models.SLT_model import ISLR_V1


def load_model(model_path, num_classes=100):
    print("Loading model...")

    model_kwargs = dict(
        num_classes=num_classes,
    )
    model = ISLR_V1(**model_kwargs)

    checkpoint = torch.load(
        model_path,
        map_location="cuda"
    )
    model.load_state_dict(checkpoint["model"])
    model = model.cuda()
    model.eval()

    return model

def predict(model, features, video_mask):
    with torch.no_grad():
            logits, loss = model(features, video_mask=video_mask)
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class
