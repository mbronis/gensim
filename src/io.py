import os
import pickle
from abc import abstractmethod
from typing import List

import pandas as pd

from src.transformers import PassThroughMixin
from src.exceptions import BadPickleLoaderDataType


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


class PickleLoader(DataLoader):
    """Loading preprocessed data from pickle"""
    file_name = {
        'data': 'df_prepro.pkl',
        'meta': 'df_prepro_meta.pkl'
    }
    folder = './data/'

    def __init__(self, save_dir: str = None, data_type: str = 'data'):
        self.save_dir = save_dir or PickleLoader.get_latest_dir()
        if data_type not in PickleLoader.file_name:
            raise BadPickleLoaderDataType(data_type)
        self.data_type = data_type

    def load(self) -> pd.DataFrame:
        file_name = PickleLoader.file_name[self.data_type]
        path = os.path.join(PickleLoader.folder, self.save_dir, file_name)
        return pd.read_pickle(path)

    @staticmethod
    def list_dir() -> List[str]:
        """Returns existing save directories"""
        dirs = [os.path.basename(p[0]) for p in os.walk(PickleLoader.folder)][1:]
        return dirs

    @staticmethod
    def get_latest_dir() -> List[str]:
        """Returns the latest save directory"""
        return PickleLoader.list_dir()[-1]

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
            pickle.dump(save_obj, f, protocol=4)
