from torch.utils.data import Dataset


class GlossClassificationDataset(Dataset):

    def __init__(self, base_dataset, tokenizer, gloss2idx):
        self.base_dataset = base_dataset
        self.tokenizer    = tokenizer
        self.gloss2idx    = gloss2idx

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        feature, text_ids = self.base_dataset[idx]
        gloss = self.tokenizer.decode(text_ids, skip_special_tokens=True).strip().lower()
        label = self.gloss2idx[gloss]
        return feature, label
