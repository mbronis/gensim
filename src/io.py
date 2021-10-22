from abc import abstractmethod

import pandas as pd

from src.transformers import PassThroughMixin


class DataLoader(PassThroughMixin):
    """Base class for data loaders with interface for Pipeline usage"""

    def __init__(self, file_name: str) -> None:
        self.file_name = file_name

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.load()


class CsvDataLoader(DataLoader):
    """Loading data from local csv"""

    def __init__(self, file_name: str = 'polish_sentiment_dataset.csv'):
        self.folder = './data/'
        self.file_name = file_name

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.folder + self.file_name)
