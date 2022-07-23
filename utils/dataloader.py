import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from typing import Callable


def create_tokenizer_dataloader(
        df: pd.DataFrame, tokenizer_function: Callable,
        batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    """
    The function receives df with 2 columns = ['labels', 'title']
    Where label is int label and title is free text (string).
    It will tokenize the data and create pytorch dataloader
    """
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenizer_function)
    tokenized_dataset = tokenized_dataset.remove_columns(["title"])
    tokenized_dataset.set_format("torch")
    return DataLoader(tokenized_dataset, shuffle=shuffle, batch_size=batch_size)
