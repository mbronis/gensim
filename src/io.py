import os
import pickle
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

class DataSaver(PassThroughMixin):
    """Base class for data saver with interface for Pipeline usage"""

    def __init__(self, save_dir: str, file_name: str) -> None:
        self.save_dir = os.path.join('./data', save_dir)
        self.save_path = os.path.join(self.save_dir, file_name)

    @abstractmethod
    def save(self, save_obj) -> None:
        pass

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        save_obj = (X, y) if y else X
        self.save(save_obj)

        return X


class DataPickler(DataSaver):
    """Class for pickling pipeline results"""

    def save(self, save_obj) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        with open(self.save_path, 'wb') as f:
            pickle.dump(save_obj, f)
