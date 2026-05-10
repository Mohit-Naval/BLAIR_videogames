# blair/dataset.py
import pandas as pd
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class ContrastivePairDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length=512):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["review"], row["metadata"]

    def collate_fn(self, batch):
        reviews, metas = zip(*batch)
        enc_review = self.tokenizer(
            list(reviews), max_length=self.max_length,
            padding=True, truncation=True, return_tensors="pt"
        )
        enc_meta = self.tokenizer(
            list(metas), max_length=self.max_length,
            padding=True, truncation=True, return_tensors="pt"
        )
        return enc_review, enc_meta
