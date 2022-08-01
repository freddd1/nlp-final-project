import pandas as pd
from torch.utils.data import DataLoader
from typing import List
import itertools

def combine_scraped_data(path_to_data: str, path_to_metadata: str, is_train=True) -> None:
    output_name = 'train' if 'train' in path_to_data else 'test'

    # load
    df = pd.read_excel(path_to_data)
    metadata = pd.read_csv(path_to_metadata)

    # process
    df = pd.merge(df, metadata, left_on=df.iloc[:, 0], right_on=metadata.iloc[:, 0])
    df = df.drop(columns=['key_0', 'Unnamed: 0_x', 'Unnamed: 0_y'])

    # save
    df.to_excel(f'data/{output_name}_processed.xlsx')


def clean_scraped_data(path_to_data: str) -> pd.DataFrame:
    df = pd.read_excel(path_to_data)
    df = df[['occupation', 'vacancyTitle']].rename(columns={'occupation': 'labels', 'vacancyTitle': 'title'})
    df.loc[:, 'title'] = df.title.apply(clean_str)
    return df


def clean_str(s: str) -> str:
    s = s.lower()
    return s


def labels_indexes_mapping(df: pd.DataFrame) -> (dict, dict):
    assert 'labels' in df.columns, "there is no labels column in the df"
    label_to_idx = {l: i for i, l in enumerate(df.labels.unique())}
    idx_to_label = {i: l for i, l in enumerate(df.labels.unique())}
    return label_to_idx, idx_to_label


def create_titles_corpus(dl: DataLoader) -> List:
    titles = []
    titles.extend([batch['titles'] for batch in dl])
    return list(itertools.chain(*titles))