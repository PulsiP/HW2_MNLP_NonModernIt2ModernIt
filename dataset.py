import pandas as pd
from datasets import Dataset

class OldItalianDataset(Dataset):
    
    def __init__(self, path):
        self.path=path
        self.dataset = pd.read_csv(path)  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self.idx=idx
        row = self.dataset.iloc[idx]
        return row["Sentence"]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]