import os, json

class Vocabulary:
    def __init__(self, label_path):
        self.label_path = label_path
        self.label2idx = {}
        self.idx2label = {}

        with open(os.path.join(label_path), "r") as f:
            self.label2idx = json.load(f)
            self.idx2label = {v: k for k, v in self.label2idx.items()}

