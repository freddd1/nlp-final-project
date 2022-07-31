from torch.utils.data import Dataset
import transformers
from typing import List, Tuple


class TitlesDataset(Dataset):
    def __init__(self, titles: List[str], labels: List[int]):
        self.labels = labels
        self.titles = titles

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> dict:
        titles = self.titles[idx]
        labels = self.labels[idx]
        return dict(titles=titles, labels=labels)