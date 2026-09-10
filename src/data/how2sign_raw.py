import pandas as pd
from torch.utils.data import Dataset, DataLoader


class How2SignDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, sep="\t")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # VIDEO_ID - cmG4MzqyjE
        # VIDEO_NAME - cmG4MzqyjE - 5 - rgb_front
        # SENTENCE_ID - cmG4MzqyjE_4
        # SENTENCE_NAME - cmG4MzqyjE_4 - 5 - rgb_front
        # START 20.44
        # END25.48
        # SENTENCE I want you
        # Name: 1672, dtype: object
        # Output shape: tensor([22514, 1672])

        video_id = row["VIDEO_ID"]
        video_name = row["SENTENCE_NAME"]
        sentence = row["SENTENCE"]

        return video_name, sentence